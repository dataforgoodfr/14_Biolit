from biolit.export_api import fetch_biolit_from_api, adapt_api_to_dataframe
from biolit.create_table import (
    get_engine,
    create_table,
    create_enriched_table,
    create_db_finale_table,
    create_taxonomy_queue_table,
    prepare_dataframe_for_postgres,
    prepare_db_finale_dataframe,
    insert_dataframe,
    insert_enriched_dataframe,
    insert_crops_dataframe,
    insert_no_crops_dataframe,
    insert_db_finale_dataframe,
    insert_taxonomy_queue_dataframe,
    load_observations_from_db_for_ML
)
from biolit.geoloc import geoloc_enrichie_data_biolit_db
from biolit.flow_gatekeeper import filter_observations_for_crop
from biolit.label_studio import (
    push_tasks_label_studio_no_crops,
    push_tasks_label_studio_crops,
    extract_crops_data_from_label_studio,
    extract_no_crops_data_from_label_studio
)
from biolit.s3 import create_s3_client, doris_object_key, s3_bucket_name
from ml.crop_inference.predict import flow_ml_crops
from ml.classification.pipeline_classification import flow_ml_classification
import datetime
import io
from dataclasses import dataclass
import structlog
import polars as pl
from biolit.settings import label_studio_crops_project, label_studio_no_crops_project

LOGGER = structlog.get_logger()


@dataclass(frozen=True)
class RuntimeConfig:
    bucket_name: str
    doris_key: str
    crops_project: str
    no_crops_project: str

    @classmethod
    def from_environment(cls) -> "RuntimeConfig":
        return cls(
            bucket_name=s3_bucket_name(),
            doris_key=doris_object_key(),
            crops_project=label_studio_crops_project(),
            no_crops_project=label_studio_no_crops_project(),
        )


def _warn_if_doris_missing(s3_client, config: RuntimeConfig) -> None:
    try:
        s3_client.head_object(Bucket=config.bucket_name, Key=config.doris_key)
        LOGGER.info(
            "Fichier DORIS present sur S3",
            bucket=config.bucket_name,
            key=config.doris_key,
        )
    except Exception:
        LOGGER.warning(
            "Fichier DORIS introuvable sur S3 - l'enrichissement DORIS sera ignore",
            bucket=config.bucket_name,
            key=config.doris_key,
        )


def _enrich_taxonomy_with_doris(
    df_taxonomy: pl.DataFrame,
    s3_client,
    config: RuntimeConfig,
) -> pl.DataFrame:
    try:
        doris_bytes = s3_client.get_object(
            Bucket=config.bucket_name, Key=config.doris_key
        )["Body"].read()
        # doris_data.csv est en réalité un fichier Parquet (magic PAR1)
        df_doris = pl.read_parquet(io.BytesIO(doris_bytes))
        LOGGER.info("Fichier DORIS charge", species=len(df_doris))

        df_doris = df_doris.with_columns(
            pl.col("nom_scientifique").str.to_lowercase()
        )
        df_taxonomy = df_taxonomy.with_columns(
            pl.col("species_name").str.to_lowercase()
        )

        df_taxonomy = df_taxonomy.join(
            df_doris,
            left_on="species_name",
            right_on="nom_scientifique",
            how="left",
        )
        LOGGER.info("Enrichissement Doris applique")
    except Exception as e:
        LOGGER.warning("Erreur enrichissement Doris - continuation sans", error=str(e))
    return df_taxonomy


def run_pipeline():
    dossier_inference = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    config = RuntimeConfig.from_environment()
    s3_client = create_s3_client()
    LOGGER.info(dossier_inference)

    # -------------------------
    # 0. VERIFICATION DU FICHIER DORIS SUR S3 (Cellar)
    # -------------------------
    _warn_if_doris_missing(s3_client, config)

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
    LOGGER.info("Geoloc Enrichment DONE")

    # -------------------------
    # 3. FLOW ML CROPS
    # -------------------------
    LOGGER.info("Creating tables for ML if not exist...")
    create_db_finale_table(engine)
    create_taxonomy_queue_table(engine)

    LOGGER.info("Recuperation des donnees a traiter pour le ML")
    df_ml = load_observations_from_db_for_ML(engine)
    # On filtre le df avec toutes les images qui sont deja passees dans le flow
    df_ml_to_process = filter_observations_for_crop(df_ml, engine)
    nb_to_process = len(df_ml_to_process)

    LOGGER.info("Nombre d'observations a traiter", value=nb_to_process)

    if nb_to_process == 0:
        LOGGER.info("Aucune nouvelle observation a traiter -> arret du pipeline")
        return

    LOGGER.info("Lancement du Flow de ML Crop")
    config_name = "ml/crop_inference/config.yaml"
    df_crops, df_no_crops, crops_images = flow_ml_crops(
        df_ml_to_process, config_name, dossier_inference
    )
    LOGGER.info("Cropping des images realisees")
    LOGGER.info("Crops uploades sur S3")

    LOGGER.info("Enregistrement des observations traitees dans Postgres")
    insert_crops_dataframe(df_crops, engine)
    insert_no_crops_dataframe(df_no_crops, engine)
    LOGGER.info("Table de Crops et No Crops mises a jours")

    # -------------------------
    # 4. PASSAGE ML TAXONOMIE EXPORT VERS LABEL STUDIO
    # -------------------------

    # --- ENVOI DES NO CROPS VERS LABEL STUDIO ---
    if len(df_no_crops) > 0:
        LOGGER.info("Envoi des observations sans crops vers Label Studio...")
        # df_no_crops ne contient que run_name/id_observation/path_s3 :
        # jointure avec df_ml_to_process pour récupérer relais, reg_nom,
        # nearest_commune, dep_nom, latitude, longitude (attendus par LS)
        df_no_crops = df_no_crops.with_columns(
            pl.col("id_observation").cast(pl.Int64)
        ).join(df_ml_to_process, on="id_observation")
        push_tasks_label_studio_no_crops(config.no_crops_project, df_no_crops)
        LOGGER.info(
            f"{len(df_no_crops)} observations sans crops envoyees "
            "vers Label Studio"
        )
    else:
        LOGGER.info("Aucune observation sans crop a envoyer vers Label Studio")

    # --- ENVOI DES CROPS VERS LABEL STUDIO (avec classification taxonomique) ---
    if len(crops_images) > 0:
        LOGGER.info("Lancement du Flow de Classification Taxonomique")
        df_taxonomy = flow_ml_classification(crops_images, df_crops)

        # --- ENRICHISSEMENT AVEC LIENS DORIS (CSV sur Cellar via boto3) ---
        df_taxonomy = _enrich_taxonomy_with_doris(df_taxonomy, s3_client, config)

        df_taxonomy = df_taxonomy.with_columns(
            pl.col("id_observation").cast(pl.Int64)
        ).join(df_ml_to_process, on="id_observation")
        push_tasks_label_studio_crops(config.crops_project, df_taxonomy)
        LOGGER.info("Classification taxonomique DONE")
    else:
        LOGGER.info("Aucun crop a classifier -> skip taxonomie")

    # -------------------------
    # 6. RECUPERATION DES INFOS DEPUIS LABEL STUDIO
    # -------------------------
    LOGGER.info("Recuperation des annotations realisees depuis le dernier run...")
    data_label_studio_crops = extract_crops_data_from_label_studio(
        config.crops_project, datetime.datetime(2025, 1, 1), datetime.datetime(2027, 1, 1)
    )
    LOGGER.info("Data collected from label studio projet Crops")
    data_label_studio_no_crops = extract_no_crops_data_from_label_studio(
        config.no_crops_project, datetime.datetime(2025, 1, 1), datetime.datetime(2027, 1, 1)
    )
    LOGGER.info("Data collected from label studio projet No Crops")

    # Insertion des donnees recuperes dans les tables postgresql
    data_label_studio_crops_filtered = prepare_db_finale_dataframe(
        data_label_studio_crops
    )
    insert_db_finale_dataframe(data_label_studio_crops_filtered, engine)
    LOGGER.info(
        "Insertion db_finale terminee projet crops",
        rows_inserted=len(data_label_studio_crops_filtered)
    )
    data_label_studio_no_crops_filtered = prepare_db_finale_dataframe(
        data_label_studio_no_crops
    )
    LOGGER.info(
        "Insertion db_finale terminee projet no crops",
        rows_inserted=len(data_label_studio_no_crops_filtered)
    )
    insert_db_finale_dataframe(data_label_studio_no_crops_filtered, engine)

    # Enregistrement donnees de crops pour reentrainnement
    insert_taxonomy_queue_dataframe(data_label_studio_no_crops, engine)
    LOGGER.info(
        "Stockage donnees pour reentrainement projet, "
        "nombre de lignes stockees",
        rows_inserted=len(data_label_studio_no_crops)
    )

    # -------------------------
    # 7. CLEANING
    # -------------------------
    LOGGER.info("Cleaning des taches annotees depuis le precedent flow...")
    LOGGER.info("Cleaning du S3...")
    LOGGER.info("Cleaning de LabelStudio...")

    LOGGER.info("Fin du Flow: succes")


if __name__ == "__main__":
    run_pipeline()
