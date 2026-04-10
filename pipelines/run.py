from biolit.export_api import fetch_biolit_from_api, adapt_api_to_dataframe
from biolit.postgres import (
    prepare_dataframe_for_postgres,
    insert_dataframe,
    get_engine,
    insert_enriched_dataframe,
    create_table,
    create_enriched_table,
    load_observations_from_db_for_S3,
)
from biolit.geoloc import geoloc_enrichie_data_biolit_db
from biolit.minio import _upload_photos_minio
import structlog
from dotenv import load_dotenv

LOGGER = structlog.get_logger()
load_dotenv()

def run_pipeline():
    # -------------------------
    # 1. INGESTION API
    # -------------------------
    LOGGER.info("Fetching data...")
    data = fetch_biolit_from_api()

    LOGGER.info("Transforming...")
    df = adapt_api_to_dataframe(data)

    LOGGER.info("Preparing for Postgres...")
    df = prepare_dataframe_for_postgres(df)

    LOGGER.info("Creating table if not exists...")
    create_table()

    LOGGER.info("Loading into Postgres...")
    insert_dataframe(df)

    # -------------------------
    # 2. ENRICHISSEMENT GEOLOC
    # -------------------------
    LOGGER.info("Starting geolocation enrichment...")
    engine = get_engine()

    df_geo = geoloc_enrichie_data_biolit_db(engine)

    LOGGER.info("Creating enriched table if not exists...")
    create_enriched_table(engine)

    LOGGER.info("Saving enriched data into Postgres...")
    insert_enriched_dataframe(df_geo, engine)

    LOGGER.info("DONE ✅")

    # -------------------------
    # 3. INSERTION DES IMAGES DANS MINIO
    # -------------------------
    LOGGER.info("Connection to minio...")
    df_minio = load_observations_from_db_for_S3(engine)
    _upload_photos_minio(df_minio)
    LOGGER.info("DONE ✅")

    # -------------------------
    # 4. ENVOIE DES CROPS A LABEL STUDIO
    # -------------------------

    # -------------------------
    # 5. RECUPERATION DES INFOS DEPUIS LABEL STUDIO
    # -------------------------

if __name__ == "__main__":
    run_pipeline()