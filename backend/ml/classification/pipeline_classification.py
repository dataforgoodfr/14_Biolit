"""
Classification taxonomique des crops (point d'entrée du pipeline principal)
==========================================================================
Appelé par pipelines/run.py après la détection YOLO (ml/crop_inference) :

    df_taxonomy = flow_ml_classification(crops_images, df_crops)
"""

from typing import Optional

import polars as pl
import structlog
from PIL import Image

from .classifier_infer import load_model, predict_batch

LOGGER = structlog.get_logger()


def flow_ml_classification(
    crops_images: dict[str, Image.Image],
    df_crops: pl.DataFrame,
    thresholds: Optional[dict] = None,
) -> pl.DataFrame:
    """
    Classification taxonomique pure — pas de S3, pas de DB.

    Args:
        crops_images: dict { id_crops → PIL.Image } produit par flow_ml_crops
        df_crops: DataFrame avec colonnes id_crops, id_observation, regne, confiance, path_s3
        thresholds: surcharge des seuils par niveau (None = seuils calibrés du modèle)

    Retourne:
        DataFrame avec une ligne par crop : infos YOLO + prédiction taxonomique
    """
    if not crops_images:
        LOGGER.info("flow_ml_classification: aucun crop à classifier")
        return pl.DataFrame()

    LOGGER.info("flow_ml_classification start", n_crops=len(crops_images))

    model = load_model()

    id_crops_list = list(crops_images.keys())
    images_list = [crops_images[k] for k in id_crops_list]

    results = predict_batch(images_list, model, thresholds=thresholds)

    meta_by_id = {row["id_crops"]: row for row in df_crops.to_dicts()}

    rows = []
    for id_crops, pred in zip(id_crops_list, results):
        meta = meta_by_id.get(id_crops, {})
        rows.append({
            "id_crops": id_crops,
            "id_observation": meta.get("id_observation"),
            "regne_yolo": meta.get("regne"),
            "confiance_yolo": meta.get("confiance"),
            "path_s3": meta.get("path_s3"),
            "best_level": pred["best_level"],
            "best_label": pred["best_label"],
            "best_score": pred["best_score"],
            "path": pred["path"],
            "margin": pred["margin"],
            "regne": pred.get("regne"),
            "phylum": pred.get("phylum"),
            "classe": pred.get("classe"),
            "ordre": pred.get("ordre"),
            "famille": pred.get("famille"),
            "species_name": pred.get("species_name"),
        })

    df = pl.DataFrame(rows)
    LOGGER.info("flow_ml_classification done", n_predictions=len(df))
    return df
