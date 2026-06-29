from datetime import UTC, datetime
from types import SimpleNamespace

from PIL import Image

from biolit.label_studio import BoundingBox, ManualCropAnnotation, ValidationAnnotation
from pipelines.run import (
    Runtime,
    _classify_crops,
    _crop_new_images,
    _sync_manual_crops,
    _sync_validations,
    crop_from_label_studio,
)


def test_crop_from_label_studio_uses_percentages():
    image = Image.new("RGB", (200, 100), "blue")
    box = BoundingBox(
        x=25,
        y=20,
        width=50,
        height=60,
        original_width=200,
        original_height=100,
    )

    assert crop_from_label_studio(image, box).size == (100, 60)


class FakeRepository:
    def __init__(self, record):
        self.record = record
        self.finalized = []
        self.transitions = []

    def get_image(self, image_id):
        return self.record if image_id == self.record["id_image"] else None

    def finalize(self, image_id, **values):
        self.finalized.append((image_id, values))
        self.record["status"] = "validated" if values["identifiable"] else "rejected"

    def transition(self, image_id, **values):
        self.transitions.append((image_id, values))
        self.record.update(values)

    def fail(self, image_id, error):
        raise AssertionError(f"Erreur inattendue pour {image_id}: {error}")

    def claim(self, current_status, claimed_status, limit):
        if self.record["status"] != current_status:
            return []
        self.record["status"] = claimed_status
        return [self.record.copy()]


class FakeLabelStudio:
    def __init__(self, *, validations=(), crops=()):
        self.validations = list(validations)
        self.crops = list(crops)
        self.deleted = []
        self.manual_tasks = []
        self.validation_tasks = []

    def validation_annotations(self):
        return self.validations

    def manual_crop_annotations(self):
        return self.crops

    def delete_task(self, task_id):
        self.deleted.append(task_id)

    def push_manual_crop(self, image_id, image_url):
        self.manual_tasks.append((image_id, image_url))

    def push_validation(self, **task):
        self.validation_tasks.append(task)


class FakeStorage:
    def __init__(self):
        self.deleted = []

    def load_image(self, uri):
        return Image.new("RGB", (200, 100), "blue")

    def upload_image(self, image_id, kind, image):
        return f"s3://temporary/processing/{image_id}/{kind}.jpg"

    def delete(self, uri):
        self.deleted.append(uri)

    def public_url(self, uri):
        return f"https://temporary.example/{uri.rsplit('/', 1)[-1]}"


def test_validation_writes_final_result_and_deletes_task():
    repository = FakeRepository(
        {
            "id_image": "42",
            "id_observation": 42,
            "status": "awaiting_validation",
            "predicted_name": "Asterias rubens",
            "predicted_rank": "species",
        }
    )
    annotation = ValidationAnnotation(
        task_id=7,
        image_id="42",
        decision="Prédiction correcte",
        corrected_name=None,
        annotator="Ada",
        annotated_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    labels = FakeLabelStudio(validations=[annotation])
    runtime = Runtime(repository, FakeStorage(), labels)

    _sync_validations(runtime)

    assert repository.finalized[0][1]["scientific_name"] == "Asterias rubens"
    assert repository.finalized[0][1]["annotator"] == "Ada"
    assert labels.deleted == [7]


def test_manual_crop_is_routed_to_classification_and_original_is_deleted():
    original_uri = "s3://temporary/processing/43/original.jpg"
    repository = FakeRepository(
        {
            "id_image": "43",
            "id_observation": 43,
            "status": "awaiting_manual_crop",
            "original_s3_uri": original_uri,
        }
    )
    annotation = ManualCropAnnotation(
        task_id=8,
        image_id="43",
        bounding_box=BoundingBox(25, 20, 50, 60, 200, 100),
        no_species=False,
        annotator="Ada",
        annotated_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    labels = FakeLabelStudio(crops=[annotation])
    storage = FakeStorage()
    runtime = Runtime(repository, storage, labels)

    _sync_manual_crops(runtime)

    assert repository.record["status"] == "pending_classification"
    assert repository.record["crop_source"] == "human"
    assert repository.record["original_s3_uri"] is None
    assert storage.deleted == [original_uri]
    assert labels.deleted == [8]


def test_missing_ml_crop_is_sent_to_manual_crop_project():
    record = {
        "id_image": "44",
        "id_observation": 44,
        "status": "pending_crop",
        "source_url": "https://images.example/44.jpg",
    }
    repository = FakeRepository(record)
    labels = FakeLabelStudio()
    crop_service = SimpleNamespace(
        process=lambda _: [
            SimpleNamespace(
                image_id="44",
                original=Image.new("RGB", (20, 20)),
                crop=None,
                label=None,
                confidence=None,
            )
        ]
    )
    runtime = Runtime(repository, FakeStorage(), labels, crop=crop_service)

    _crop_new_images(runtime)

    assert repository.record["status"] == "awaiting_manual_crop"
    assert labels.manual_tasks[0][0] == "44"


def test_classification_is_sent_to_validation_project():
    crop_uri = "s3://temporary/processing/45/crop.jpg"
    record = {
        "id_image": "45",
        "id_observation": 45,
        "status": "pending_classification",
        "crop_s3_uri": crop_uri,
    }
    repository = FakeRepository(record)
    labels = FakeLabelStudio()
    classifier = SimpleNamespace(
        process=lambda _: [
            SimpleNamespace(
                name="Asterias rubens",
                rank="species",
                score=0.91,
                margin=0.4,
                details={"species_top3": []},
            )
        ]
    )
    runtime = Runtime(repository, FakeStorage(), labels, classification=classifier)

    _classify_crops(runtime)

    assert repository.record["status"] == "awaiting_validation"
    assert repository.record["predicted_name"] == "Asterias rubens"
    assert labels.validation_tasks[0]["image_id"] == "45"
