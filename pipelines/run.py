from biolit.export_api import load_biolit_from_api
from biolit.postgres import prepare_dataframe_for_postgres, insert_dataframe
from biolit.geoloc import geoloc_enrichie_data_biolit
import structlog

LOGGER = structlog.get_logger()


def run_pipeline():
    LOGGER.info("Fetching data...")
    data = load_biolit_from_api()

    LOGGER.info("Preparing for Postgres...")
    df = prepare_dataframe_for_postgres(data)

    LOGGER.info("Loading into Postgres...")
    insert_dataframe(df)

    LOGGER.info("DONE ✅")

    LOGGER.info("Enrichment Geoloc")
    geoloc_enrichie_data_biolit()


if __name__ == "__main__":
    run_pipeline()