from biolit.export_api import fetch_biolit_from_api, adapt_api_to_dataframe
from biolit.create_table import (
    prepare_dataframe_for_postgres,
    insert_dataframe,
    get_engine,
    insert_enriched_dataframe,
    create_table,
    create_enriched_table,
)
from biolit.geoloc import geoloc_enrichie_data_biolit_db


def run_pipeline():
    # -------------------------
    # 1. INGESTION API
    # -------------------------
    print("Fetching data...")
    data = fetch_biolit_from_api()

    print("Transforming...")
    df = adapt_api_to_dataframe(data)

    print("Preparing for Postgres...")
    df = prepare_dataframe_for_postgres(df)

    print("Creating table if not exists...")
    create_table()

    print("Loading into Postgres...")
    insert_dataframe(df)

    # -------------------------
    # 2. ENRICHISSEMENT GEOLOC
    # -------------------------
    print("Starting geolocation enrichment...")
    engine = get_engine()

    df_geo = geoloc_enrichie_data_biolit_db(engine)

    print("Creating enriched table if not exists...")
    create_enriched_table(engine)

    print("Saving enriched data into Postgres...")
    insert_enriched_dataframe(df_geo, engine)

    print("DONE ✅")


if __name__ == "__main__":
    run_pipeline()