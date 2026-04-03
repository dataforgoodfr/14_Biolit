from biolit.export_api import fetch_biolit_from_api, adapt_api_to_dataframe
from biolit.postgres import prepare_dataframe_for_postgres, insert_dataframe


def run_pipeline():
    print("Fetching data...")
    data = fetch_biolit_from_api()

    print("Transforming...")
    df = adapt_api_to_dataframe(data)

    print("Preparing for Postgres...")
    df = prepare_dataframe_for_postgres(df)

    print("Loading into Postgres...")
    insert_dataframe(df)

    print("DONE ✅")


if __name__ == "__main__":
    run_pipeline()