from io import BytesIO
from urllib.parse import urlparse

import boto3
import structlog
from botocore.client import Config
from PIL import Image

from biolit.settings import (
    doris_object_key,
    env_bool,
    label_studio_presigned_url_expires,
    s3_access_key_id,
    s3_bucket_name,
    s3_endpoint_url,
    s3_region,
    s3_secret_access_key,
)

__all__ = [
    "create_s3_client",
    "doris_object_key",
    "load_image_from_s3",
    "parse_s3_uri",
    "public_url_for_s3_uri",
    "s3_bucket_name",
    "upload_image_s3",
    "upload_parquet_s3",
]

LOGGER = structlog.get_logger()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


# =============================================
# UPLOADS BOTO3 (vers Cellar)
# =============================================


def upload_parquet_s3(client, df, bucket_name: str, object_name: str):
    """Upload un DataFrame Polars (Parquet) vers Cellar via boto3."""
    buffer = BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)
    client.put_object(
        Body=buffer,
        Bucket=bucket_name,
        Key=object_name,
        ContentLength=buffer.getbuffer().nbytes,
    )
    LOGGER.info("Parquet uploaded", path=f"s3://{bucket_name}/{object_name}")


def upload_image_s3(client, pil_img: Image.Image, bucket_name: str, object_name: str):
    """Upload une image PIL (JPEG) vers Cellar via boto3."""
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    buffer.seek(0)
    client.put_object(
        Body=buffer,
        Bucket=bucket_name,
        Key=object_name,
        ContentType="image/jpeg",
        ContentLength=buffer.getbuffer().nbytes,
    )
    LOGGER.info("Image uploaded", key=object_name)


# =============================================
# FONCTIONS POUR BOTO3 (Checks/Lectures)
# =============================================


def create_s3_client():
    """
    Crée un client boto3 compatible Cellar, Scaleway ou MinIO/local.
    """
    options = {
        "aws_access_key_id": s3_access_key_id(),
        "aws_secret_access_key": s3_secret_access_key(),
        "region_name": s3_region(),
        "config": Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            # Cellar rejette l'encodage aws-chunked (checksum CRC32 par défaut
            # de botocore >=1.36) -> MissingContentLength. On le désactive.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
        "verify": env_bool("S3_VERIFY_SSL", True),
    }
    endpoint = s3_endpoint_url()
    if endpoint:
        options["endpoint_url"] = endpoint

    return boto3.client("s3", **options)


def public_url_for_s3_uri(uri: str, expires_in: int | None = None, client=None) -> str:
    """
    Convertit une URI s3://bucket/key en URL présignée lisible par Label Studio.
    """
    if not uri.startswith("s3://"):
        return uri

    bucket, key = parse_s3_uri(uri)
    expires = expires_in or label_studio_presigned_url_expires()
    s3_client = client or create_s3_client()
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def _check_file_existence_s3(client, bucket_name: str, key: str) -> bool:
    """Vérifie si un fichier existe dans S3 avec boto3."""
    try:
        client.head_object(Bucket=bucket_name, Key=key)
        LOGGER.info("File exists:", key=key)
        return True
    except Exception as e:
        LOGGER.info("File does not exist:", key=key, error=str(e))
        return False


def _read_file_s3(client, bucket_name: str, key: str) -> bytes:
    """Lit un fichier depuis S3 avec boto3."""
    obj = client.get_object(Bucket=bucket_name, Key=key)
    LOGGER.info("Fichier Lu :", key=key)
    return obj["Body"].read()


def load_image_from_s3(s3_client, bucket_name: str, object_key: str) -> Image.Image:
    """Charge une image depuis S3 et retourne un PIL.Image."""
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    image_data = response["Body"].read()
    return Image.open(BytesIO(image_data)).convert("RGB")
