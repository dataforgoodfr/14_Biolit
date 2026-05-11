import polars as pl
from dotenv import load_dotenv
load_dotenv()
import os 

FORCE_REPROCESS = os.getenv("FORCE_REPROCESS", "false").lower() == "true"

# =========================
# HELPERS DB
# =========================
def get_already_cropped_observations(engine) -> pl.DataFrame:
    """
    Récupère les identifiants des observations déjà passées par l'étape de crop (ML1).
    On prend en compte à la fois :
    - les observations avec crops détectés (ml_crops)
    - les observations sans détection (ml_no_crops)
    """
    query = """
        SELECT DISTINCT CAST(split_part(id_crops,'_',1) AS BIGINT) AS id_observation FROM ml_crops
        UNION
        SELECT DISTINCT CAST(id_observation AS BIGINT) FROM ml_no_crops
    """
    return pl.read_database(query, engine)


def get_already_classified_observations(engine) -> pl.DataFrame:
    """
    Récupère les identifiants des observations déjà passées par l'étape de classification (ML2).

    """
    query = "SELECT DISTINCT id_observation FROM ml_taxonomy"
    return pl.read_database(query, engine)


# =========================
# FILTERS
# =========================
def filter_observations_for_crop(df: pl.DataFrame, engine) -> pl.DataFrame:
    """
    Filtre les observations à envoyer au modèle de crop (ML1).
    Logique :
    - Si FORCE_REPROCESS = True → aucun filtrage
    - Sinon → on exclut les observations déjà traitées (présentes dans ml_crops ou ml_no_crops)
    """
    if FORCE_REPROCESS:
        return df

    processed = get_already_cropped_observations(engine)

    if processed.is_empty():
        return df

    return df.filter(~pl.col("id_observation").is_in(processed["id_observation"]))


def filter_crops_for_classification(df_crops: pl.DataFrame, engine) -> pl.DataFrame:
    """
    Filtre les crops à envoyer au modèle de classification (ML2).

    Logique :
    - Si FORCE_REPROCESS = True → aucun filtrage
    - Sinon → on exclut les observations déjà classifiées
    """
    if FORCE_REPROCESS:
        return df_crops

    classified = get_already_classified_observations(engine)

    if classified.is_empty():
        return df_crops

    return df_crops.filter(~pl.col("id_observation").is_in(classified["id_observation"]))


def filter_processed_no_crop_annotations( df: pl.DataFrame, engine) -> pl.DataFrame:
        """
        Filtre les annotations NO_CROP déjà traitées.

        Vérifie :
        - bd_finale (id_observation)
        - taxonomy_queue (crop_id)
        """
        # =========================
        # Récupération BD finale
        # =========================
        query_finale = """
            SELECT id_observation
            FROM bd_finale
        """

        df_finale = pl.read_database(
            query=query_finale,
            connection=engine
        )

        # =========================
        # Récupération taxonomy queue
        # =========================
        query_taxo = """
            SELECT crop_id
            FROM taxonomy_queue
        """

        df_taxo = pl.read_database(
            query=query_taxo,
            connection=engine
        )

        # =========================
        # Filtre observations finales
        # =========================
        if not df_finale.is_empty():

            df = df.filter(
                ~pl.col("id_observation").is_in(
                    df_finale["id_observation"]
                )
            )

        # =========================
        # Filtre crops déjà envoyés ML
        # =========================
        if (
            "crop_id" in df.columns
            and not df_taxo.is_empty()
        ):

            df = df.filter(
                ~pl.col("crop_id").is_in(
                    df_taxo["crop_id"]
                )
            )

        return df
    
def filter_processed_crop_annotations( df: pl.DataFrame,engine):
        query = """
            SELECT id_observation
            FROM bd_finale
        """

        existing = pl.read_database(
            query=query,
            connection=engine
        )

        if existing.is_empty():
            return df

        return df.filter(
            ~pl.col("id_observation").is_in(
                existing["id_observation"]
            )
        )