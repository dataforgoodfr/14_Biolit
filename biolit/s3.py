import boto3
import os
from botocore.exceptions import ClientError
import structlog
from dotenv import load_dotenv
LOGGER = structlog.get_logger()
load_dotenv()


def test_permissions(bucket_name):

    ACCESS_KEY = os.getenv("aws_access_key_id")
    SECRET_KEY = os.getenv("aws_secret_access_key")

    ENDPOINT_URL = "https://s3.fr-par.scw.cloud"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL
    )

    LOGGER.info(f"\n🔎 Bucket: {bucket_name}")

    # 1. List bucket
    try:
        s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        LOGGER.info("✅ ListBucket: OK")
    except ClientError as e:
        LOGGER.info(f"❌ ListBucket: {e.response['Error']['Code']}")

    # 2. Get bucket ACL
    try:
        s3.get_bucket_acl(Bucket=bucket_name)
        LOGGER.info("✅ GetBucketAcl: OK")
    except ClientError as e:
        LOGGER.info(f"❌ GetBucketAcl: {e.response['Error']['Code']}")