import requests
import polars as pl
import structlog
import os
from dotenv import load_dotenv
from minio import Minio
from io import BytesIO
import json

LOGGER = structlog.get_logger()
load_dotenv()

def _upload_photos_minio(df: pl.DataFrame):
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")

    # Config MinIO
    client = Minio(
        "minio:9000",
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    LOGGER.info("Connected to S3")

    bucket_name = "crops-data"

    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        LOGGER.info(f"Bucket created: {bucket_name}")
    else:
        LOGGER.info(f"Bucket already exists: {bucket_name}")

    for row in df.to_dicts():
            id_obs = row["id_observation"]
            url = row["photos"]

            filename = url.split("/")[-1]
            object_name = f"{id_obs}/{filename}"

            try:
                client.stat_object(bucket_name, object_name)
                LOGGER.info(f"Skipped (already exists): {object_name}")
                continue
            except Exception:
                pass

            response = requests.get(url)
            if response.status_code == 200:
                data = BytesIO(response.content)
                client.put_object(
                    bucket_name,
                    object_name,
                    data,
                    length=len(response.content),
                    content_type="image/jpeg"
                )
                LOGGER.info(f"Uploaded: {object_name}")
            else:
                LOGGER.warning(f"Failed to fetch: {url}")

def _get_label_studios_info_minio():
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")

    # Config MinIO
    client = Minio(
        "minio:9000",
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    LOGGER.info("Connected to S3")

    bucket_name = "label-data"

    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        LOGGER.info(f"Bucket created: {bucket_name}")
    else:
        LOGGER.info(f"Bucket already exists: {bucket_name}")

    objects = client.list_objects(bucket_name, recursive=True)
    all_annotations = []

    for obj in objects:
        LOGGER.info(f"Reading object: {obj.object_name}")

        response = client.get_object(bucket_name, obj.object_name)

        try:
            content = response.read().decode("utf-8")
            data = json.loads(content)

            all_annotations.append(data)

            LOGGER.info(f"Loaded OK: {obj.object_name}")
        except Exception as e:
            LOGGER.warning(f"Not JSON or failed: {obj.object_name} - {e}")

        finally:
            response.close()
            response.release_conn()

    return all_annotations


def annotations_to_polars(all_annotations):
    rows = []

    for ann in all_annotations:
        annotation_id = ann.get("id")

        task = ann.get("task", {})
        task_id = task.get("id")
        image = task.get("data", {}).get("image")

        for res in ann.get("result", []):
            value = res.get("value", {})

            row = {
                "task_id": task_id,
                "annotation_id": annotation_id,
                "type": res.get("type"),
                "from_name": res.get("from_name"),
                "label": value.get("choices"),
                "image": image,
            }

            rows.append(row)
    df = pl.DataFrame(rows)
    LOGGER.info(df)
    return df