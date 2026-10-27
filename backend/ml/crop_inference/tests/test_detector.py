"""
Tests unitaires de la détection ONNX, sans modèle : une fausse session onnxruntime
renvoie une sortie YOLO construite à la main (1, 4 + nb_classes, 8400).
"""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ml.crop_inference.detector import class_names, detect, list_images
from ml.crop_inference.model_loader import load_model_weights

NAMES = "{0: 'plant', 1: 'animal'}"


def _fake_session(boxes: list[tuple[float, float, float, float, float, int]]):
    """boxes : (cx, cy, w, h, score, class_id) dans le repère letterbox 640x640."""
    preds = np.zeros((1, 6, 8400), dtype=np.float32)
    for i, (cx, cy, w, h, score, cls) in enumerate(boxes):
        preds[0, :4, i] = (cx, cy, w, h)
        preds[0, 4 + cls, i] = score
    return SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="images", shape=[1, 3, 640, 640])],
        run=lambda _outputs, _feeds: [preds],
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={"names": NAMES}),
    )


def test_class_names_from_metadata():
    assert class_names(_fake_session([])) == {0: "plant", 1: "animal"}


def test_detect_keeps_best_box_and_maps_it_to_original_image():
    # 1280x640 → échelle 0.5, image 640x320 collée à top=160 dans le carré 640x640
    session = _fake_session([(100, 300, 40, 40, 0.5, 0), (320, 320, 100, 50, 0.9, 1)])
    det = detect(session, Image.new("RGB", (1280, 640)), conf=0.4)
    assert det.class_id == 1 and det.class_name == "animal"
    assert det.confidence == pytest.approx(0.9)
    assert det.bbox_xyxy == pytest.approx([540, 270, 740, 370])


def test_detect_clips_box_to_image():
    session = _fake_session([(0, 160, 200, 200, 0.9, 0)])
    det = detect(session, Image.new("RGB", (1280, 640)), conf=0.4)
    x1, y1, x2, y2 = det.bbox_xyxy
    assert 0 <= x1 <= x2 <= 1280 and 0 <= y1 <= y2 <= 640


def test_detect_returns_none_below_threshold():
    session = _fake_session([(320, 320, 100, 100, 0.3, 0)])
    assert detect(session, Image.new("RGB", (640, 480)), conf=0.4) is None


def test_list_images_filters_extensions(tmp_path):
    for name in ["b.JPG", "a.png", "c.jpeg", "notes.txt"]:
        (tmp_path / name).touch()
    assert [p.name for p in list_images(str(tmp_path))] == ["a.png", "b.JPG", "c.jpeg"]
    assert list_images(str(tmp_path / "a.png")) == [tmp_path / "a.png"]


def test_load_model_weights_local_and_errors(tmp_path):
    model = tmp_path / "model.onnx"
    model.touch()
    assert load_model_weights({"model": {"source": "local", "path": str(model)}}) == str(model)
    with pytest.raises(FileNotFoundError):
        load_model_weights({"model": {"source": "local", "path": str(tmp_path / "absent.onnx")}})
    with pytest.raises(ValueError):
        load_model_weights({"model": {"source": "s3"}})
