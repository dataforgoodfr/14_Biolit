import torch
import yaml
import unicodedata
import numpy as np
from pathlib import Path
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

# Fix PyTorch / Ultralytics / NumPy
torch.use_deterministic_algorithms(False)
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **kwargs, weights_only=False)

# np.trapz supprimé en NumPy 2.0, ultralytics 8.0.81 l'utilise encore
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# Helpers
def sanitize_filenames(img_dir: str) -> None:
    """Renomme les fichiers avec accents → ASCII safe (cv2.imread échoue sur macOS sinon)"""
    for f in Path(img_dir).iterdir():
        if f.is_file():
            nfkd = unicodedata.normalize("NFKD", f.name)
            ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
            if ascii_name != f.name:
                new_path = f.parent / ascii_name
                f.rename(new_path)
                print(f"Renommé : {f.name} → {ascii_name}")


def clean_non_images(img_dir: str) -> list[Path]:
    """Déplace les fichiers non-image (.DS_Store, etc.) dans un sous-dossier temporaire.
    Retourne la liste des fichiers déplacés pour restauration ultérieure.
    """
    img_path = Path(img_dir)
    non_images = [
        f for f in img_path.iterdir()
        if f.is_file() and (f.name.startswith(".") or f.suffix.lower() not in VALID_IMG_EXT)
    ]

    if not non_images:
        return []

    tmp = img_path / "_non_images"
    tmp.mkdir(exist_ok=True)

    for f in non_images:
        f.rename(tmp / f.name)
        print(f"Déplacé (non-image) : {f.name}")

    return non_images


def restore_non_images(img_dir: str) -> None:
    """Restaure les fichiers non-image déplacés par clean_non_images."""
    tmp = Path(img_dir) / "_non_images"
    if not tmp.exists():
        return
    for f in tmp.iterdir():
        f.rename(tmp.parent / f.name)
    tmp.rmdir()


def load_config(path: str = "configs/autodistill_boostrap.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Pipeline
def label(cfg: dict) -> None:
    print("[1/2] Pseudo-labeling GroundingDINO")

    ontology = CaptionOntology(cfg["ontology"])
    base_model = GroundingDINO(ontology=ontology)

    # Écarte les fichiers non-image avant le labeling
    clean_non_images(cfg["img_path"])

    try:
        base_model.label(
            input_folder=cfg["img_path"],
            output_folder=cfg["labeled_output"],
        )
    finally:
        restore_non_images(cfg["img_path"])

    print(f"Labels générés → {cfg['labeled_output']}")


def train(cfg: dict) -> None:
    print("[2/2] Training YOLOv8 (bootstrap)")
    print(f"GPU dispo : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device : {torch.cuda.get_device_name(0)}")

    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    model = YOLOv8(cfg["model"])
    model.yolo.train(
        data=cfg["data_yaml"],
        epochs=cfg["epochs"],
        device=device,
        project=cfg["project"],
        name=cfg["name"],
    )


# Main
if __name__ == "__main__":
    cfg = load_config()
    sanitize_filenames(cfg["img_path"])
    label(cfg)
    train(cfg)
