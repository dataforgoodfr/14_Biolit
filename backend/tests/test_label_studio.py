from biolit.label_studio import LabelStudioGateway
from biolit.settings import CROP_PROJECT, VALIDATION_PROJECT


def gateway(tasks):
    label_studio = LabelStudioGateway(None, "http://label-studio")
    label_studio._project_ids = {CROP_PROJECT: 7, VALIDATION_PROJECT: 8}
    label_studio._tasks = lambda title: tasks[label_studio._project_ids[title]]
    return label_studio


def test_reads_manual_bounding_box():
    task = {
        "id": 11,
        "data": {"id_image": "42"},
        "annotations": [
            {
                "created_at": "2026-06-01T10:00:00Z",
                "created_username": "Ada, ada@example.org",
                "result": [
                    {
                        "from_name": "crop",
                        "type": "rectanglelabels",
                        "original_width": 200,
                        "original_height": 100,
                        "value": {"x": 10, "y": 20, "width": 50, "height": 60},
                    }
                ],
            }
        ],
    }

    annotations = gateway({7: [task]}).manual_crop_annotations()

    assert annotations[0].image_id == "42"
    assert annotations[0].bounding_box.width == 50
    assert annotations[0].annotator == "Ada"


def test_reads_corrected_validation():
    task = {
        "id": 12,
        "data": {"id_image": "42"},
        "annotations": [
            {
                "created_at": "2026-06-01T10:00:00Z",
                "completed_by": 3,
                "result": [
                    {
                        "from_name": "decision",
                        "value": {"choices": ["Corriger l'espèce"]},
                    },
                    {
                        "from_name": "corrected_species",
                        "value": {"text": ["Asterias rubens"]},
                    },
                ],
            }
        ],
    }

    annotations = gateway({8: [task]}).validation_annotations()

    assert annotations[0].decision == "Corriger l'espèce"
    assert annotations[0].corrected_name == "Asterias rubens"
