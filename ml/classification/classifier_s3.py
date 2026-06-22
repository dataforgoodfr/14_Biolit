"""
classifier_s3.py — Lecture des images croppées depuis S3
=========================================================
Ce module gère la récupération des images croppées depuis le bucket S3
pour alimenter le pipeline de classification.

"""

import io
import os
from pathlib import Path
from typing import Optional

import boto3
import polars as pl
import structlog
from botocore.exceptions import ClientError
from dotenv import load_dotenv, find_dotenv
from PIL import Image

LOGGER = structlog.get_logger()
load_dotenv(find_dotenv())


def create_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        endpoint_url=os.getenv("aws_url"),
    )

# Configuration S3
BUCKET_NAME = "biolit-uploads"
DEFAULT_RUN_NAME = "latest"


# ════════════════════════════════════════════════════════════════════════════
# CLIENT S3
# ════════════════════════════════════════════════════════════════════════════

def get_s3_client():
    """Retourne un client S3 configuré."""
    return create_s3_client()


# ════════════════════════════════════════════════════════════════════════════
# LISTE DES CROPS DISPONIBLES
# ════════════════════════════════════════════════════════════════════════════

def list_available_crops(
    run_name: str = DEFAULT_RUN_NAME,
    bucket: str = BUCKET_NAME,
    client: Optional[boto3.client] = None
) -> list[str]:
    """
    Liste tous les crops disponibles pour un run donné.

    Retourne une liste de clés S3 (object_name) pour les crops.
    """
    if client is None:
        client = get_s3_client()

    prefix = f"{run_name}/crops/"
    crops = []

    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith((".jpg", ".jpeg", ".png")):
                    crops.append(key)
        LOGGER.info("%d crops trouvés pour run=%s", len(crops), run_name)
    except ClientError as e:
        LOGGER.error("Erreur list_available_crops: %s", e)
        return []

    return crops


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES IMAGES DEPUIS S3
# ════════════════════════════════════════════════════════════════════════════

def load_image_from_s3(
    key: str,
    bucket: str = BUCKET_NAME,
    client: Optional[boto3.client] = None
) -> Optional[Image.Image]:
    """
    Charge une image depuis S3 et retourne un objet PIL Image.
    """
    if client is None:
        client = get_s3_client()

    try:
        response = client.get_object(Bucket=bucket, Key=key)
        image_data = response["Body"].read()
        img = Image.open(io.BytesIO(image_data))
        return img.convert("RGB")
    except ClientError as e:
        LOGGER.warning("Impossible de charger %s: %s", key, e)
        return None
    except Exception as e:
        LOGGER.warning("Erreur traitement %s: %s", key, e)
        return None


def load_crops_from_s3(
    run_name: str = DEFAULT_RUN_NAME,
    bucket: str = BUCKET_NAME,
    limit: Optional[int] = None,
    client: Optional[boto3.client] = None
) -> pl.DataFrame:
    """
    Charge les métadonnées des crops depuis S3.

    Retourne un DataFrame Polars avec :
        - id_observation : ID de l'observation source
        - id_crops : ID unique du crop
        - regne : classe détectée par YOLO
        - confiance : score de confiance YOLO
        - path_s3 : chemin S3 complet
        - image : objet PIL Image (lazy loading)
    """
    if client is None:
        client = get_s3_client()

    crops = list_available_crops(run_name, bucket, client)

    if limit:
        crops = crops[:limit]

    rows = []
    for key in crops:
        # Parse: run_name/crops/{id_observation}_{regne}_{confiance}.jpg
        filename = Path(key).stem  # sans extension
        parts = filename.split("_")

        if len(parts) >= 3:
            id_observation = parts[0]
            regne = parts[1]
            try:
                confiance = float(parts[2])
            except ValueError:
                confiance = 0.0
        else:
            id_observation = filename
            regne = "unknown"
            confiance = 0.0

        rows.append({
            "id_observation": id_observation,
            "id_crops": filename,
            "regne": regne,
            "confiance": confiance,
            "path_s3": f"s3://{bucket}/{key}",
            "s3_key": key,
        })

    df = pl.DataFrame(rows)
    LOGGER.info("%d crops chargés depuis S3", len(df))
    return df


def load_crops_with_images(
    run_name: str = DEFAULT_RUN_NAME,
    bucket: str = BUCKET_NAME,
    limit: Optional[int] = None,
    client: Optional[boto3.client] = None
) -> list[tuple[Image.Image, dict]]:
    """
    Charge les crops avec leurs images PIL.

    Retourne une liste de tuples (image, metadata) où metadata est un dict
    avec les métadonnées du crop.
    """
    df = load_crops_from_s3(run_name, bucket, limit, client)

    results = []
    for row in df.iter_rows(named=True):
        img = load_image_from_s3(row["s3_key"], bucket, client)
        if img is not None:
            results.append((img, row))

    LOGGER.info("%d images chargées avec succès", len(results))
    return results

