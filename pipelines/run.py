from biolit.export_api import fetch_biolit_from_api, adapt_api_to_dataframe
from biolit.create_table import (
    prepare_dataframe_for_postgres,
    insert_dataframe,
    get_engine,
    create_table,
    load_observations_from_db_for_ML,
    insert_crops_dataframe,
    insert_no_crops_dataframe,
)

from biolit.flow_gatekeeper import(
    filter_observations_for_crop,
    filter_crops_for_classification
)
from biolit.label_studio import (
    push_tasks_label_studio_no_crops,
)
from ml.crop_inference.predict import flow_ml_crops
import datetime
import structlog
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
    """
    df_geo = geoloc_enrichie_data_biolit_db(engine)

    LOGGER.info("Creating enriched table if not exists...")
    create_enriched_table(engine)

    LOGGER.info("Saving enriched data into Postgres...")
    insert_enriched_dataframe(df_geo, engine)
    LOGGER.info("Geoloc Enrichment DONE ✅")
    """

    # -------------------------
    # 3. ENREGISTREMENT TABLE METABASE
    # -------------------------

    # -------------------------
    # 4. FLOW ML CROPS
    # -------------------------
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
    df_crops, df_no_crops = flow_ml_crops(df_ml_to_process, config_name, dossier_inference)
    LOGGER.info("Cropping des images réalisées")
    LOGGER.info("Crops uploadés sur S3")

    LOGGER.info(df_crops)
    LOGGER.info(df_no_crops)
    insert_crops_dataframe(df_crops, engine)
    insert_no_crops_dataframe(df_no_crops, engine)
    LOGGER.info("Table de Crops et No Crops mises à jours")

    # -------------------------
    # 5. PASSAGE ML TAXONOMIE
    # -------------------------
    LOGGER.info("passage à l'identification par le ML2")
    df_crops_to_classify=filter_crops_for_classification(df_crops,engine)
    nb_to_classify = len(df_crops_to_classify)

    LOGGER.info(
        "Nombre d'observations non classifiées",
        value=nb_to_classify
    )

    if nb_to_classify == 0:
        LOGGER.info("Aucun crop à classifier → skip ML taxonomy ✅")

    # -------------------------
    # 6. ENVOIE DES IMAGES A LABEL STUDIO
    # -------------------------
    LOGGER.info("Connection to Label Studio...")
    if len(df_no_crops) == 0:
        LOGGER.info("Aucune image à envoyer à Label Studio → skip ✅")
    else:
        push_tasks_label_studio_no_crops("Biolit No Crops", df_no_crops)
        LOGGER.info("LABEL STUDIO DONE ✅")



    # -------------------------
    # 7. ENVOIE DES CROPS A LABEL STUDIO
    # -------------------------
    """
    LOGGER.info("Connection to Label Studio...")
    push_tasks_label_studio("Biolit Crops", df)
    LOGGER.info("LABEL STUDIO DONE ✅")

    # -------------------------
    # 8. RECUPERATION DES INFOS DEPUIS LABEL STUDIO
    # -------------------------
    """

if __name__ == "__main__":
    run_pipeline()