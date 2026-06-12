import os
import polars as pl
from sqlalchemy import create_engine, text
import pandas as pd
import structlog
from typing import Dict
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

def insert_no_crops_dataframe(df: pl.DataFrame, engine):
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
        SELECT
            id_observation,
            photos,
            latitude,
            longitude,
            relais,
            nearest_commune,
            reg_nom,
            dep_nom,
            validee
        FROM observations
        LEFT JOIN observations_enriched
        USING (id_observation)
        WHERE LOWER(validee) = 'false'
        AND photos LIKE 'https:%'
        AND id_observation NOT IN (
            SELECT DISTINCT CAST(split_part(id_crops, '_', 1) AS BIGINT) FROM ml_crops
            UNION
            SELECT DISTINCT CAST(id_observation AS BIGINT) FROM ml_no_crops
        )
        LIMIT 20
    """
    return pl.read_database(query, engine)

def load_observations_from_crops_for_Label_Studio(engine) -> pl.DataFrame:
    query = """
        SELECT *
        FROM ml_crops
    """
    return pl.read_database(query, engine)

def create_taxonomy_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml_taxonomy (
                id_crops        TEXT PRIMARY KEY,
                run_name        TEXT,
                id_observation  TEXT,
                regne           TEXT,
                confiance       DOUBLE PRECISION,
                path_s3         TEXT,
                best_level      TEXT,
                best_label      TEXT,
                best_score      DOUBLE PRECISION,
                phylum          TEXT,
                classe          TEXT,
                ordre           TEXT,
                famille         TEXT,
                species_name    TEXT
            );
        """))

def insert_taxonomy_predictions(df: pl.DataFrame, engine) -> None:
    rows = df.to_dicts()
    with engine.begin() as conn:
        for row in rows:
            conn.execute(text("""
                INSERT INTO ml_taxonomy (
                    id_crops, run_name, id_observation,
                    regne, confiance, path_s3,
                    best_level, best_label, best_score,
                    phylum, classe, ordre, famille, species_name
                ) VALUES (
                    :id_crops, :run_name, :id_observation,
                    :regne_yolo, :confiance_yolo, :path_s3,
                    :best_level, :best_label, :best_score,
                    :phylum, :classe, :ordre, :famille, :species_name
                )
                ON CONFLICT (id_crops) DO UPDATE SET
                    run_name     = EXCLUDED.run_name,
                    best_level   = EXCLUDED.best_level,
                    best_label   = EXCLUDED.best_label,
                    best_score   = EXCLUDED.best_score
            """), row)
    LOGGER.info("ml_taxonomy: %d lignes insérées", len(rows))


def get_observation_image_path(engine,
                               id_obs:int,
                               table_name:str) -> Dict :

    """
    Récupère le path S3 associé à une observation
    puis retourne :
    {
        "bucket": ...,
        "object_key": ...
    }
    """

    query=f"""
      SELECT path_s3 FROM {table_name}
        WHERE  CAST(id_observation AS BIGINT)={id_obs};
    """
    df = pl.read_database(query=query, connection=engine)

    if df.is_empty():
        LOGGER.info(f"No IMAGE for this {id_obs}")
        return None

    # récupération string réelle
    s3_path = df["path_s3"][0]

    # suppression prefix s3://
    sans_prefix = s3_path[5:]

    # split bucket / key
    bucket_name, rest = sans_prefix.split("/", 1)

    return {
            "bucket": bucket_name,
            "rest": rest }

def create_db_finale_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS db_finale (
                id_observation BIGINT PRIMARY KEY,
                nom_scientifique TEXT,
                validee  BOOLEAN,
                identifiable BOOLEAN,
                annotateur  TEXT,
                source  TEXT
            );
        """))

def create_taxonomy_queue_table(engine):

    with engine.begin() as conn:

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS taxonomy_queue (
                id_crops TEXT PRIMARY KEY,
                id_observation BIGINT NOT NULL,
                task_created_date TIMESTAMP,
                crop_index INTEGER,
                x FLOAT,
                y FLOAT,
                width FLOAT,
                height FLOAT,
                label TEXT,
                original_width INTEGER,
                original_height INTEGER,
                annotator TEXT,
                annotated_at TIMESTAMP
            );
        """))

def insert_db_finale_dataframe(df, engine ):
        """
        Insert les observations finales dans db_finale.
        """

        if df.is_empty():
            LOGGER.info("Aucune donnée à insérer dans db_finale")
            return

        rows = df.to_dicts()

        query = text("""
            INSERT INTO db_finale (
                id_observation,
                nom_scientifique,
                validee,
                identifiable,
                annotateur,
                source
            )
            VALUES (
                :id_observation,
                :nom_scientifique,
                :validee,
                :identifiable,
                :annotateur,
                :source
            )
            ON CONFLICT (id_observation)
            DO NOTHING
        """)

        with engine.begin() as conn:
            conn.execute(query, rows)

        LOGGER.info(
            "Insertion db_finale terminée",
            rows_inserted=len(rows)
        )


def insert_taxonomy_queue_dataframe( df,engine ):
        """
        Insert les crops manuels à envoyer
        vers ML taxonomy.
        """

        if df.is_empty():
            LOGGER.info("Aucune donnée à insérer dans taxonomy_queue")
            return

        rows = df.to_dicts()

        query = text("""
            INSERT INTO taxonomy_queue (
                id_crops,
                id_observation,
                task_created_date,
                crop_index,
                x,
                y,
                width,
                height,
                label,
                original_width,
                original_height,
                annotator,
                annotated_at
            )
            VALUES (
                :id_crops,
                :id_observation,
                :task_created_date,
                :crop_index,
                :x,
                :y,
                :width,
                :height,
                :label,
                :original_width,
                :original_height,
                :annotator,
                :annotated_at
            )
            ON CONFLICT (id_crops)
            DO NOTHING
        """)

        with engine.begin() as conn:
            conn.execute(query, rows)

        LOGGER.info(
            "Insertion taxonomy_queue terminée",
            rows_inserted=len(rows)
        )

def prepare_db_finale_dataframe(df: pl.DataFrame ) -> pl.DataFrame:
        """
        Prépare le dataframe avant insertion
        dans Bd_finale.
        """

        return (
            df
            .select([
                "id_observation",
                "nom_scientifique",
                "validee",
                "identifiable",
                "annotateur",
                "source"
            ])
            .with_columns([
                pl.col("id_observation").cast(pl.Int64),
                pl.col("nom_scientifique").cast(pl.Utf8),
                pl.col("validee").cast(pl.Boolean),
                pl.col("identifiable").cast(pl.Boolean),
                pl.col("annotateur").cast(pl.Utf8),
                pl.col("source").cast(pl.Utf8),
            ])
        )