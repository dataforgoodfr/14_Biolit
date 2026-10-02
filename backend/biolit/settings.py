"""Configuration runtime partagee par le pipeline et le deploiement."""

import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_S3_BUCKET = "biolit-uploads"
DEFAULT_DORIS_KEY = "doris_data.csv"
DEFAULT_CROPS_PROJECT = "Biolit Crops"
DEFAULT_NO_CROPS_PROJECT = "Biolit No Crops"
DEFAULT_S3_REGION = "fr-par"
DEFAULT_LABEL_STUDIO_URL_EXPIRES = 604_800


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def require_env(*names: str) -> str:
    value = first_env(*names)
    if value:
        return value
    joined = " or ".join(names)
    raise ValueError(f"Missing required environment variable: {joined}")


def postgres_url() -> str:
    return require_env("POSTGRES_URL", "POSTGRESQL_ADDON_URI")


def s3_endpoint_url() -> str | None:
    endpoint = first_env(
        "S3_ENDPOINT_URL",
        "AWS_ENDPOINT_URL",
        "AWS_S3_ENDPOINT",
        "aws_url",
        "CELLAR_ADDON_HOST",
    )
    if endpoint and not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    return endpoint


def s3_bucket_name() -> str:
    return (
        first_env(
            "BIOLIT_S3_BUCKET",
            "AWS_STORAGE_BUCKET_NAME",
            "S3_BUCKET",
            "aws_storage_bucket_name",
        )
        or DEFAULT_S3_BUCKET
    )


def s3_region() -> str:
    return (
        first_env("AWS_REGION", "AWS_DEFAULT_REGION", "aws_region")
        or DEFAULT_S3_REGION
    )


def s3_access_key_id() -> str:
    return require_env("AWS_ACCESS_KEY_ID", "aws_access_key_id", "CELLAR_ADDON_KEY_ID")


def s3_secret_access_key() -> str:
    return require_env(
        "AWS_SECRET_ACCESS_KEY",
        "aws_secret_access_key",
        "CELLAR_ADDON_KEY_SECRET",
    )


def doris_object_key() -> str:
    return os.getenv("DORIS_S3_KEY", DEFAULT_DORIS_KEY)


def label_studio_url() -> str:
    return require_env("LABEL_STUDIO_URL")


def label_studio_api_key() -> str:
    return require_env("LABEL_STUDIO_API_KEY_DATAFORGOOD")


def label_studio_crops_project() -> str:
    return os.getenv("LABEL_STUDIO_CROPS_PROJECT", DEFAULT_CROPS_PROJECT)


def label_studio_no_crops_project() -> str:
    return os.getenv("LABEL_STUDIO_NO_CROPS_PROJECT", DEFAULT_NO_CROPS_PROJECT)


def label_studio_presigned_url_expires() -> int:
    return env_int(
        "LABEL_STUDIO_PRESIGNED_URL_EXPIRES",
        DEFAULT_LABEL_STUDIO_URL_EXPIRES,
    )
