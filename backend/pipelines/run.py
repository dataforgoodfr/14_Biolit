"""Un cycle de production : API → crop → classification → validation → nettoyage."""

import json
from dataclasses import dataclass
from typing import Any

import structlog
from PIL import Image

from biolit.database import Repository
from biolit.export_api import fetch_observations
from biolit.label_studio import BoundingBox, LabelStudioGateway
from biolit.settings import PIPELINE_BATCH_SIZE, PROCESSING_STALE_MINUTES
from biolit.storage import TemporaryStorage

LOGGER = structlog.get_logger()


@dataclass
class Runtime:
    repository: Repository
    storage: TemporaryStorage
    label_studio: LabelStudioGateway
    crop: Any | None = None
    classification: Any | None = None

    @classmethod
    def from_environment(cls) -> "Runtime":
        return cls(
            repository=Repository.from_environment(),
            storage=TemporaryStorage.from_environment(),
            label_studio=LabelStudioGateway.from_environment(),
        )

    def initialize(self) -> None:
        self.repository.create_schema()
        recovered = self.repository.recover_stale_claims(PROCESSING_STALE_MINUTES)
        if recovered:
            LOGGER.warning("stale_images_requeued", count=recovered)
        self.storage.ensure_bucket()
        self.label_studio.ensure_projects()
        if self.crop is None:
            from ml.crop_inference import CropService

            self.crop = CropService()
        if self.classification is None:
            from ml.classification import ClassificationService

            self.classification = ClassificationService()


def crop_from_label_studio(image: Image.Image, box: BoundingBox) -> Image.Image:
    """Convertit les coordonnées en pourcentage Label Studio en pixels."""

    width, height = image.size
    left = max(0, int(box.x / 100 * width))
    top = max(0, int(box.y / 100 * height))
    right = min(width, left + int(box.width / 100 * width))
    bottom = min(height, top + int(box.height / 100 * height))
    if right <= left or bottom <= top:
        raise ValueError("Le crop Label Studio est vide.")
    return image.crop((left, top, right, bottom)).convert("RGB")


def run_cycle(runtime: Runtime) -> None:
    """Exécute un cycle idempotent sur un lot limité d'images."""

    runtime.initialize()
    observations = fetch_observations()
    summary = runtime.repository.ingest(observations)
    LOGGER.info(
        "ingestion_completed",
        observations=summary.observations,
        images=summary.images_discovered,
    )

    _sync_validations(runtime)
    _sync_manual_crops(runtime)
    _crop_new_images(runtime)
    _classify_crops(runtime)
    _cleanup_finalized_assets(runtime)


def _sync_validations(runtime: Runtime) -> None:
    for annotation in runtime.label_studio.validation_annotations():
        image = runtime.repository.get_image(annotation.image_id)
        if image is None:
            runtime.label_studio.delete_task(annotation.task_id)
            continue
        if image["status"] in {"validated", "rejected"}:
            runtime.label_studio.delete_task(annotation.task_id)
            continue
        if image["status"] != "awaiting_validation":
            continue

        if annotation.decision == "Prédiction correcte":
            scientific_name = image["predicted_name"]
            taxon_rank = image["predicted_rank"]
            identifiable = scientific_name is not None
        elif annotation.decision == "Corriger l'espèce":
            scientific_name = annotation.corrected_name
            taxon_rank = "species"
            identifiable = scientific_name is not None
            if not identifiable:
                runtime.repository.fail(annotation.image_id, "Correction sans nom scientifique")
                continue
        else:
            scientific_name = None
            taxon_rank = None
            identifiable = False

        runtime.repository.finalize(
            annotation.image_id,
            scientific_name=scientific_name,
            taxon_rank=taxon_rank,
            identifiable=identifiable,
            annotator=annotation.annotator,
            source="label_studio_validation",
            validated_at=annotation.annotated_at,
        )
        runtime.label_studio.delete_task(annotation.task_id)


def _sync_manual_crops(runtime: Runtime) -> None:
    for annotation in runtime.label_studio.manual_crop_annotations():
        image = runtime.repository.get_image(annotation.image_id)
        if image is None:
            runtime.label_studio.delete_task(annotation.task_id)
            continue
        if image["status"] in {"validated", "rejected"}:
            runtime.label_studio.delete_task(annotation.task_id)
            continue
        if image["status"] != "awaiting_manual_crop":
            continue

        try:
            if annotation.no_species and annotation.bounding_box is None:
                runtime.repository.finalize(
                    annotation.image_id,
                    scientific_name=None,
                    taxon_rank=None,
                    identifiable=False,
                    annotator=annotation.annotator,
                    source="label_studio_crop",
                    validated_at=annotation.annotated_at,
                )
            else:
                original = runtime.storage.load_image(image["original_s3_uri"])
                crop = crop_from_label_studio(original, annotation.bounding_box)
                crop_uri = runtime.storage.upload_image(annotation.image_id, "crop", crop)
                runtime.repository.transition(
                    annotation.image_id,
                    status="pending_classification",
                    crop_s3_uri=crop_uri,
                    crop_source="human",
                )
                runtime.storage.delete(image["original_s3_uri"])
                runtime.repository.transition(annotation.image_id, original_s3_uri=None)
            runtime.label_studio.delete_task(annotation.task_id)
        except Exception as error:
            runtime.repository.fail(annotation.image_id, error)


def _crop_new_images(runtime: Runtime) -> None:
    images = runtime.repository.claim("pending_crop", "cropping", PIPELINE_BATCH_SIZE)
    if not images:
        return
    try:
        results = runtime.crop.process(images)
    except Exception as error:
        for image in images:
            runtime.repository.fail(image["id_image"], error)
        return

    for result in results:
        uploaded_uri = None
        try:
            if result.crop is not None:
                uploaded_uri = runtime.storage.upload_image(result.image_id, "crop", result.crop)
                runtime.repository.transition(
                    result.image_id,
                    status="pending_classification",
                    crop_s3_uri=uploaded_uri,
                    crop_source="ml",
                    crop_label=result.label,
                    crop_confidence=result.confidence,
                )
            else:
                uploaded_uri = runtime.storage.upload_image(
                    result.image_id,
                    "original",
                    result.original,
                )
                runtime.label_studio.push_manual_crop(
                    result.image_id,
                    runtime.storage.public_url(uploaded_uri),
                )
                runtime.repository.transition(
                    result.image_id,
                    status="awaiting_manual_crop",
                    original_s3_uri=uploaded_uri,
                )
        except Exception as error:
            if uploaded_uri:
                runtime.storage.delete(uploaded_uri)
            runtime.repository.fail(result.image_id, error)


def _classify_crops(runtime: Runtime) -> None:
    records = runtime.repository.claim(
        "pending_classification",
        "classifying",
        PIPELINE_BATCH_SIZE,
    )
    if not records:
        return

    valid_records = []
    images = []
    for record in records:
        try:
            images.append(runtime.storage.load_image(record["crop_s3_uri"]))
            valid_records.append(record)
        except Exception as error:
            runtime.repository.fail(record["id_image"], error)

    try:
        predictions = runtime.classification.process(images)
    except Exception as error:
        for record in valid_records:
            runtime.repository.fail(record["id_image"], error)
        return

    for record, prediction in zip(valid_records, predictions, strict=True):
        try:
            runtime.label_studio.push_validation(
                image_id=record["id_image"],
                image_url=runtime.storage.public_url(record["crop_s3_uri"]),
                predicted_name=prediction.name,
                predicted_rank=prediction.rank,
                score=prediction.score,
            )
            runtime.repository.transition(
                record["id_image"],
                status="awaiting_validation",
                predicted_name=prediction.name,
                predicted_rank=prediction.rank,
                prediction_score=prediction.score,
                prediction_margin=prediction.margin,
                prediction_details=json.dumps(prediction.details, ensure_ascii=False),
            )
        except Exception as error:
            runtime.repository.fail(record["id_image"], error)


def _cleanup_finalized_assets(runtime: Runtime) -> None:
    for image in runtime.repository.finalized_assets():
        try:
            runtime.storage.delete(image["original_s3_uri"])
            runtime.storage.delete(image["crop_s3_uri"])
            runtime.repository.clear_assets(image["id_image"])
        except Exception as error:
            LOGGER.warning(
                "temporary_asset_cleanup_failed",
                image_id=image["id_image"],
                error=str(error),
            )


def main() -> None:
    run_cycle(Runtime.from_environment())


if __name__ == "__main__":
    main()
