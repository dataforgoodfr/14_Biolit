import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


# Configuration de base TODO: Ajouter dans config.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_SMALL_MODELS = os.environ.get("USE_SMALL_MODELS", "0") == "1"
ASSETS_DIR = Path(__file__).parent / "assets"

# Récupération du modèle TODO: Ajouter dans config.py
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny" if USE_SMALL_MODELS else "IDEA-Research/grounding-dino-base"
gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).eval().to(device)

# Code détection avec GroundingDINO TODO: mettre dans detect.py
def _detect_gdino(image: Image.Image, labels: str, threshold: float) -> tuple[Image.Image, list[dict], dict, list]:
    text = labels.strip().rstrip(".")
    candidate_labels = [part.strip() for part in text.split(",") if part.strip()]
    text_prompt = ". ".join(candidate_labels) + "."

    inputs = gdino_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    outputs = gdino_model(**inputs)
    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        target_sizes=[(image.height, image.width)],
        threshold=threshold,
        text_threshold=threshold,
    )[0]

    return results

# Code de crop et annotation TODO: mettre dans img_utils
def crop_and_anotate(image, results, threshold):
    annotations = []
    crops = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"], strict=True):
        x_min, y_min, x_max, y_max = box.tolist()
        annotations.append(
            {
                "bbox": {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min},
                "score": round(float(score), 3),
                "label": label,
            }
        )
        # Crop de la région détectée
        crop = image.crop((x_min, y_min, x_max, y_max))
        crops.append((crop, f"{label} {float(score):.1%}"))

    return image, annotations, {"score_threshold": (threshold, 1.0)}, crops


# Script d'inférence TODO: mettre dans main.py
@torch.inference_mode()
def run_detection():
    # Partie détection
    detection = _detect_gdino(
        image = Image.open(f"{ASSETS_DIR}/Biolit-Espece (12).webp"),
        labels="animal, plant",
        threshold=0.4
    )

    # Partie crop
    crop_anotation = crop_and_anotate(
        image = Image.open(f"{ASSETS_DIR}/Biolit-Espece (12).webp"),
        results=detection,
        threshold=0.4
    )

    # Afficher les infos et sauvegarder l'image TODO: afficher dans un DataFrame pour évaluer le modèle
    print("Bbox : ", crop_anotation[1][0]["bbox"])
    print("Score : ", crop_anotation[1][0]["score"])
    print("Label : ", crop_anotation[1][0]["label"])
    # print(crop_anotation[2])
    crop_anotation[3][0][0].save("Biolit-Espece_(12)_crop.webp","WEBP")

# def run_detection(image: Image.Image, labels: str, threshold: float) -> tuple | None:
#     if image is None or not labels.strip():
#         return None, []
#     image, annotations, score_threshold, crops = _detect_gdino(image, labels, threshold)
    # return (image, annotations, score_threshold), crops



# Exécuter la fonction
if __name__ == "__main__":
    run_detection()
#     results = run_detection(
#         image = Image.open(f"{ASSETS_DIR}/Biolit-Espece (12).webp"),
#         labels="animal, plant",
#         threshold=0.4
#     )

#     print(results)