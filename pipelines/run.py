from biolit.export_api import fetch_biolit_from_api, adapt_api_to_dataframe
from biolit.create_table import (
    prepare_dataframe_for_postgres,
    insert_dataframe,
    insert_enriched_dataframe,
    get_engine,
    create_table,
    create_enriched_table,
    load_observations_from_db_for_ML,
    insert_crops_dataframe,
    insert_no_crops_dataframe,
    create_Bd_finale_table,
    create_taxonomy_queue_table,
    insert_taxonomy_queue_dataframe,
    insert_bd_finale_dataframe,
    prepare_bd_finale_dataframe
    
)
from biolit.geoloc import geoloc_enrichie_data_biolit_db
from biolit.flow_gatekeeper import(
    filter_observations_for_crop
)
from biolit.label_studio import (
    push_tasks_label_studio_no_crops,
    push_tasks_label_studio_crops,
    extract_crop_data_from_label_studio,
    extract_no_crops_data_from_label_studio
)
from biolit.s3 import create_s3_client, upload_parquet_s3
from biolit.label_studio_postprocessing import (
    process_no_crop_annotations,
    process_crop_annotations
)
from ml.crop_inference.predict import flow_ml_crops
from ml.classification.pipeline_classification import flow_ml_classification
import datetime
import structlog
import polars as pl
from dotenv import load_dotenv

LOGGER = structlog.get_logger()
load_dotenv()

def run_pipeline():
    dossier_inference = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    LOGGER.info(dossier_inference)

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
    create_Bd_finale_table(engine)
    create_taxonomy_queue_table(engine)
    
    # df_geo = geoloc_enrichie_data_biolit_db(engine)

    LOGGER.info("Creating enriched table if not exists...")
    create_enriched_table(engine)

    LOGGER.info("Saving enriched data into Postgres...")
    # insert_enriched_dataframe(df_geo, engine)
    LOGGER.info("Geoloc Enrichment DONE ✅")

    # # -------------------------
    # # 3. FLOW ML CROPS
    # # -------------------------
    LOGGER.info("Récupération des données à traiter pour le ML")
    df_ml = load_observations_from_db_for_ML(engine)
    df_ml_to_process = filter_observations_for_crop(df_ml, engine)
    nb_to_process = len(df_ml_to_process)

    LOGGER.info(
        "Nombre d'observations à traiter",
        value=nb_to_process
    )

    if nb_to_process == 0:
        LOGGER.info("Aucune nouvelle observation à traiter → arrêt du pipeline ✅")
        return

    LOGGER.info("Lancement du Flow de ML Crop")
    config_name="ml/crop_inference/config.yaml"
    df_crops, df_no_crops= flow_ml_crops(df_ml_to_process, config_name, dossier_inference)
    LOGGER.info("Cropping des images réalisées")
    LOGGER.info("Crops uploadés sur S3")
    LOGGER.info(f"this is df_crops {df_crops}")

    # LOGGER.info("Enregistrement des observations traitées dans Postgres")
    # insert_crops_dataframe(df_crops, engine)
    # insert_no_crops_dataframe(df_no_crops, engine)
    # LOGGER.info("Table de Crops et No Crops mises à jours")

    # # # -------------------------
    # # # 5. PASSAGE ML TAXONOMIE
    # # # -------------------------
    LOGGER.info("Connection to Label Studio projet No Crops...")
    data_label_studio_No_crop=extract_no_crops_data_from_label_studio("Biolit No Crops")
    
    df_label_crop_ml,db_final_crop_invalide=process_no_crop_annotations(data_label_studio_No_crop,
                                                                  dossier_inference,
                                                                  engine)
    
    
    LOGGER.info(f"data collected from {df_label_crop_ml}")
    
    df_bd_final = prepare_bd_finale_dataframe(db_final_crop_invalide)
    
    LOGGER.info(f"data collected from {df_bd_final}")
    
    insert_bd_finale_dataframe(df_bd_final, engine)
    insert_taxonomy_queue_dataframe(df_label_crop_ml,engine)
    
    if len(crops_images) > 0:
        LOGGER.info("Lancement du Flow de Classification Taxonomique")
        df_taxonomy = flow_ml_classification(crops_images, df_crops)

        s3_client = create_s3_client()
        parquet_key = f"{dossier_inference}/taxonomy/predictions.parquet"
        upload_parquet_s3(s3_client, df_taxonomy, "biolit-uploads", parquet_key)

        push_tasks_label_studio_crops("Biolit Crops", df_taxonomy)
        LOGGER.info("Classification taxonomique DONE ✅")
    else:
        LOGGER.info("Aucun crop à classifier → skip taxonomie ✅")

    # # -------------------------
    # 6. ENVOIE DES IMAGES A LABEL STUDIO
    # -------------------------
    LOGGER.info("Connection to Label Studio...")
    if len(df_no_crops) == 0:
        LOGGER.info("Aucune image à envoyer à Label Studio → skip ✅")
    else:
        df_no_crops = df_no_crops.with_columns(
            pl.col("id_observation").cast(pl.Int64)
        ).join(
            df_ml_to_process, on="id_observation"
        )

        push_tasks_label_studio_no_crops("Biolit No Crops", df_no_crops)
        LOGGER.info("LABEL STUDIO DONE ✅")

    # """
    # # -------------------------
    # # 7. ENVOIE DES CROPS A LABEL STUDIO
    # # -------------------------
    # LOGGER.info("Connection to Label Studio...")
    # push_tasks_label_studio_crops("Biolit Crops", df)
    # LOGGER.info("LABEL STUDIO DONE ✅")

    # -------------------------
    # 8. RECUPERATION DES INFOS DEPUIS LABEL STUDIO
    # -------------------------
    # data_label_studio_crop=extract_crop_data_from_label_studio("Taxonomie")
    # LOGGER.info(f"data collected from label studio {data_label_studio_crop}")
    # cro_bd_finale=process_crop_annotations(data_label_studio_crop,dossier_inference,engine)
    # LOGGER.info(f"data collected from {cro_bd_finale}")

if __name__ == "__main__":
    run_pipeline()