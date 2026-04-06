import requests
import polars as pl
import structlog
import re
import os
from dotenv import load_dotenv
load_dotenv()

LOGGER = structlog.get_logger()

# ------------------------------
# FETCH API
# ------------------------------
def fetch_biolit_from_api():

    url = os.getenv("BIOLIT_API_URL")


    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    print(f"{len(data)} observations récupérées")
    return data

# ------------------------------
# RENAME OF COLUMNS
# ------------------------------


def normalize_column_name(col: str) -> str:
    """Convertit les noms API en snake_case propre FR"""
    col = col.lower()
    col = col.replace("-", "_")
    col = col.replace(" ", "_")
    col = col.replace("é", "e").replace("è", "e").replace("à", "a")
    col = col.replace("ù", "u").replace("ô", "o")
    col = re.sub(r"[^a-z0-9_]", "", col)
    return col


COLUMN_MAPPING = {
    "id": "id_observation",
    "date": "date_observation",
    "link": "lien_observation",
    "author": "observateur",
    "_url_sortie": "url_sortie",
    "espece-identifiee": "espece_identifiee",
    "heure-debut": "heure_debut",
    "heure-fin": "heure_fin",
    "latitude": "latitude",
    "longitude": "longitude",
    "photos": "photos",
    "relais": "relais",
    "espece_id": "id_espece",
    "espece": "nom_scientifique",
    "common": "nom_commun",
    "categorie-programme": "categorie_programme",
    "programme": "programme",
}


# ------------------------------
# ADAPT API -> PARQUET
# ------------------------------
def adapt_api_to_dataframe(data: list) -> pl.DataFrame:
    rows = []

    for item in data:
        new_row = {}

        for key, value in item.items():
            # mapping si connu, sinon normalisation auto
            new_key = COLUMN_MAPPING.get(key, normalize_column_name(key))
            new_row[new_key] = value

        rows.append(new_row)

    df = pl.DataFrame(rows)

    return df


# ------------------------------
# LOAD (Fetch + Adapt)
# ------------------------------
def load_biolit_from_api() -> pl.DataFrame:
    raw_data = fetch_biolit_from_api()
    df = adapt_api_to_dataframe(raw_data)
    return df

