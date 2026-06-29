"""Stockage temporaire des originaux et crops sur une API compatible S3."""

import os
from io import BytesIO
from urllib.parse import urlparse

import boto3
import botocore.exceptions
from dotenv import load_dotenv
from PIL import Image

from biolit.settings import S3_BUCKET

load_dotenv()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"URI S3 invalide : {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


class TemporaryStorage:
    """Gère uniquement les fichiers temporaires du cycle de validation."""

    def __init__(self, client, public_client, bucket: str = S3_BUCKET):
        self.client = client
        self.public_client = public_client
        self.bucket = bucket

    @classmethod
    def from_environment(cls) -> "TemporaryStorage":
        access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
        endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
        public_endpoint = os.getenv("S3_PUBLIC_ENDPOINT_URL", endpoint)
        region = os.getenv("AWS_REGION", "us-east-1")
        if not access_key or not secret_key:
            raise ValueError("Les identifiants S3/MinIO sont absents.")

        options = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": region,
        }
        return cls(
            boto3.client("s3", endpoint_url=endpoint, **options),
            boto3.client("s3", endpoint_url=public_endpoint, **options),
        )

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except botocore.exceptions.ClientError as error:
            code = str(error.response.get("Error", {}).get("Code"))
            if code not in {"404", "NoSuchBucket"}:
                raise
            self.client.create_bucket(Bucket=self.bucket)

    def upload_image(self, image_id: str, kind: str, image: Image.Image) -> str:
        if kind not in {"original", "crop"}:
            raise ValueError(f"Type d'image temporaire inconnu : {kind}")
        key = f"processing/{image_id}/{kind}.jpg"
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="image/jpeg",
        )
        return f"s3://{self.bucket}/{key}"

    def load_image(self, uri: str) -> Image.Image:
        bucket, key = parse_s3_uri(uri)
        response = self.client.get_object(Bucket=bucket, Key=key)
        return Image.open(BytesIO(response["Body"].read())).convert("RGB")

    def delete(self, uri: str | None) -> None:
        if not uri:
            return
        bucket, key = parse_s3_uri(uri)
        self.client.delete_object(Bucket=bucket, Key=key)

    def public_url(self, uri: str, expires_in: int = 604_800) -> str:
        bucket, key = parse_s3_uri(uri)
        return self.public_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
