import os
import polars as pl
from sqlalchemy import create_engine, text
import pandas as pd
import structlog
from dotenv import load_dotenv
LOGGER = structlog.get_logger()
load_dotenv()

# -------------------------
# Connexion DB
# -------------------------

def get_engine():
    postgres_url = os.getenv("POSTGRES_URL")

    if not postgres_url:
        raise ValueError("Missing DATABASE_URL")
    return create_engine(postgres_url)

# -------------------------
# Création de la table (si besoin)
# -------------------------
def create_table():
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS observations (
                id_observation BIGINT PRIMARY KEY,
                date_observation TIMESTAMP,
                lien_observation TEXT,
                observateur TEXT,
                url_sortie TEXT,
                espece_identifiee TEXT,
                heure_debut TIME,
                heure_fin TIME,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                photos TEXT,
                relais BIGINT,
                id_espece BIGINT,
                nom_scientifique TEXT,
                nom_commun TEXT,
                categorie_programme BIGINT,
                programme TEXT,
                validee TEXT
            );
        """))

def create_enriched_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS observations_enriched (
                id_observation BIGINT PRIMARY KEY,
                nearest_commune TEXT,
                code_insee TEXT,
                distance_commune_m DOUBLE PRECISION,
                code_postal TEXT,
                reg_nom TEXT,
                dep_nom TEXT,
                distance_to_coast DOUBLE PRECISION,
                is_coastal BOOLEAN
            );
        """))


# -------------------------
# Préparation des données
# -------------------------
def prepare_dataframe_for_postgres(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([

        # -------------------------
        # IDs
        # -------------------------
        pl.col("id_observation")
        .cast(pl.Int64),

        pl.col("id_espece")
        .cast(pl.Float64, strict=False)
        .fill_nan(None)
        .cast(pl.Int64, strict=False),

        pl.col("categorie_programme")
        .cast(pl.Float64, strict=False)
        .fill_nan(None)
        .cast(pl.Int64, strict=False),

        pl.col("relais")
        .cast(pl.Utf8)
        .replace("", None)
        .cast(pl.Float64, strict=False)
        .fill_nan(None)
        .cast(pl.Int64, strict=False),

        # -------------------------
        # Coordonnées
        # -------------------------
        pl.col("latitude")
        .cast(pl.Utf8)
        .str.strip_chars()
        .cast(pl.Float64, strict=False),

        pl.col("longitude")
        .cast(pl.Utf8)
        .str.strip_chars()
        .cast(pl.Float64, strict=False),

        # -------------------------
        # Dates
        # -------------------------
        pl.col("date_observation")
        .str.strptime(pl.Datetime, strict=False),

        pl.col("heure_debut")
        .str.strptime(pl.Time, strict=False),

        pl.col("heure_fin")
        .str.strptime(pl.Time, strict=False),
    ])

# -------------------------
# Insert avec sécurité (UPSERT)
# -------------------------

def insert_dataframe(df: pl.DataFrame):
    engine = get_engine()

    rows = df.to_dicts()

    with engine.begin() as conn:
        for row in rows:
            conn.execute(text("""
                INSERT INTO observations (
                    id_observation,
                    date_observation,
                    lien_observation,
                    observateur,
                    url_sortie,
                    espece_identifiee,
                    heure_debut,
                    heure_fin,
                    latitude,
                    longitude,
                    photos,
                    relais,
                    id_espece,
                    nom_scientifique,
                    nom_commun,
                    categorie_programme,
                    programme,
                    validee
                ) VALUES (
                    :id_observation,
                    :date_observation,
                    :lien_observation,
                    :observateur,
                    :url_sortie,
                    :espece_identifiee,
                    :heure_debut,
                    :heure_fin,
                    :latitude,
                    :longitude,
                    :photos,
                    :relais,
                    :id_espece,
                    :nom_scientifique,
                    :nom_commun,
                    :categorie_programme,
                    :programme,
                    :validee
                )
                ON CONFLICT (id_observation) DO NOTHING
            """), row)

def insert_enriched_dataframe(df: pd.DataFrame, engine):
    pl_df = pl.from_pandas(df)
    rows = pl_df.to_dicts()

    with engine.begin() as conn:
        for row in rows:
            conn.execute(text("""
                INSERT INTO observations_enriched (
                    id_observation,
                    nearest_commune,
                    code_insee,
                    distance_commune_m,
                    code_postal,
                    reg_nom,
                    dep_nom,
                    distance_to_coast,
                    is_coastal
                ) VALUES (
                    :id_observation,
                    :nearest_commune,
                    :code_insee,
                    :distance_commune_m,
                    :code_postal,
                    :reg_nom,
                    :dep_nom,
                    :distance_to_coast,
                    :is_coastal
                )
                ON CONFLICT (id_observation) DO NOTHING
            """), row)

def insert_no_crops_dataframe(df: pl.Dataframe, engine):
    rows = df.to_dicts()

    with engine.begin() as conn:
        for row in rows:
            conn.execute(text("""
                INSERT INTO ml_no_crops (
                    run_name,
                    id_observation,
                    path_s3
                ) VALUES (
                    :run_name,
                    :id_observation,
                    :path_s3
                )
                ON CONFLICT (id_observation) DO NOTHING
            """), row)

def insert_crops_dataframe(df: pl.DataFrame, engine):
    rows = df.to_dicts()

    with engine.begin() as conn:
        for row in rows:
            conn.execute(text("""
                INSERT INTO ml_crops (
                    run_name,
                    id_crops,
                    regne,
                    confiance,
                    path_s3
                ) VALUES (
                    :run_name,
                    :id_crops,
                    :regne,
                    :confiance,
                    :path_s3
                )
                ON CONFLICT (id_crops) DO NOTHING
            """), row)

def load_observations_from_db(engine) -> pl.DataFrame:
    query = """
        SELECT *
        FROM observations
    """

    return pl.read_database(query, engine)

def load_observations_from_db_for_ML(engine) -> pl.DataFrame:
    query = """
        SELECT id_observation, photos, latitude, longitude
        FROM observations
        LIMIT 100
    """
    return pl.read_database(query, engine)

def load_observations_from_crops_for_Label_Studio(engine) -> pl.DataFrame:
    query = """
        SELECT *
        FROM ml_crops
    """
    return pl.read_database(query, engine)