import polars as pl
import requests

###Test export from API

def fetch_biolit_api(per_page: int = 1000):
    all_data = []
    page = 1
    print("Téléchargement des données depuis l'API Biolit...")
    while True:
        url = f"https://biolit.fr/wp-json/biolitapi/v1/observations/all?per_page={per_page}&page={page}"
        r = requests.get(url)
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        page += 1
        print(f"Page {page} téléchargée, total observations : {len(all_data)}")

    if not all_data:
        return pl.DataFrame([])
    df = pl.DataFrame(all_data)
    print(df.head())
    print(df.shape)
    return df


def adapt_api_to_parquet_schema(df):
    return (
        df.rename({
            "id": "id",
            "link": "lien",
            "author": "auteur",
            "date": "date",
            "heure-debut": "heure-de-debut",
            "heure-fin": "heure-de-fin",
            "latitude": "latitude",
            "longitude": "longitude",
            "photos": "images",
            "espece": "nom_scientifique",
            "common": "nom_commun",
        })
        .with_columns([
            pl.col("lien").str.split("/").list.get(-1).alias("titre"),  # dernier segment du lien
            pl.lit("").alias("validee"),
            pl.lit("TBD").alias("espece_identifiable_?"),
            pl.lit("API").alias("protocole"),
        ])
    )
def load_biolit_from_api() -> pl.DataFrame:
    df_api = fetch_biolit_api()
    if df_api.is_empty():
        return df_api
    print(adapt_api_to_parquet_schema(df_api).head())
    print(adapt_api_to_parquet_schema(df_api).columns)
    return adapt_api_to_parquet_schema(df_api)

def format_observations_from_api():
### A faire plus tard
    return 


if __name__ == "__main__":
    print("Script lancé")
    load_biolit_from_api()