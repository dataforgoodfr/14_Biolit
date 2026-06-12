import datetime

import polars as pl
from PIL import Image


from biolit.minio import (
    create_minio_client,
    load_image_from_s3_mino,
    upload_crop_image
)

from biolit.create_table import (
    get_observation_image_path
)
from biolit.flow_gatekeeper import(
    filter_processed_no_crop_annotations,
    filter_processed_crop_annotations
)
import structlog
LOGGER = structlog.get_logger()

def crop_image_from_url( image: Image.Image, x: float, y: float, width: float, height: float, original_width: int, original_height: int ) -> Image.Image:
        """
        Crop une image à partir des coordonnées Label Studio (en %).
        """

        # -------------------------
        # Conversion % → pixels
        # -------------------------
        x_px = int(x / 100 * original_width)
        y_px = int(y / 100 * original_height)
        w_px = int(width / 100 * original_width)
        h_px = int(height / 100 * original_height)

        # -------------------------
        # Sécurité coordonnées
        # -------------------------
        x_px = max(0, x_px)
        y_px = max(0, y_px)

        # -------------------------
        # Crop
        # -------------------------
        crop_box = (
            x_px,
            y_px,
            x_px + w_px,
            y_px + h_px
        )

        cropped_image = image.crop(crop_box)

        return cropped_image


def process_no_crop_annotations(df: pl.DataFrame, run_name: str, engine ):

        LOGGER.info("Starting process_no_crop_annotations")

        # -------------------------
        # Extraction datetime depuis run_name
        # -------------------------
        run_time = datetime.datetime.strptime(
            run_name.replace("run_", ""),
            "%Y%m%d_%H%M%S"
        ).replace(tzinfo=datetime.timezone.utc)

        # -------------------------
        # Filtrage tâches annotées
        # avant le run courant
        # -------------------------
        df = df.with_columns(
            pl.col("annotated_at").str.to_datetime(
                format="%Y-%m-%dT%H:%M:%S%.fZ",
                time_zone="UTC"
            )
        )

        data_filtered = df.filter(
            pl.col("task_created_date") < run_time
        )
        data_process=filter_processed_no_crop_annotations(data_filtered,engine)

        LOGGER.info(f"Nombre annotations à traiter : {len(data_process)}")

        # -------------------------
        # Init client S3/MinIO
        # -------------------------
        client = create_minio_client()

        rows_db = []

        # -------------------------
        # Traitement crops
        # -------------------------
        for row in data_process.iter_rows(named=True):

            # =========================
            # CAS NON IDENTIFIABLE
            # =========================
            if row["label"] == "Pas d'espèce":

                rows_db.append({
                    "id_crops" : row["id_crops"],
                    "id_observation": row["id_observation"],
                    "label": row["label"],
                    "validee": False,
                    "identifiable": False,
                    "nom_scientifique" : False,
                    "source": "labelstudio_no_crop",
                    "annotateur" :row["annotator"]
                })

                continue

            # =========================
            # Récupération image source
            # =========================
            image_path = get_observation_image_path(
                engine,
                row["id_observation"],
                "ml_no_crops"
            )
            LOGGER.info(f"bucket name {image_path["bucket"]} and object key {image_path["rest"]}")
            image = load_image_from_s3_mino(
                client,
                bucket_name=image_path["bucket"],
                object_key=image_path["rest"]
            )

            # =========================
            # Crop image
            # =========================
            cropped_image = crop_image_from_url(
                image=image,
                x=row["x"],
                y=row["y"],
                width=row["width"],
                height=row["height"],
                original_width=row["original_width"],
                original_height=row["original_height"]
            )

            # =========================
            # Upload crop S3
            # =========================
            object_name = (
                f"{run_name}/crops_labelstudio/"
                f"{row['id_crops']}.jpg"
            )

            upload_crop_image(
                client=client,
                pil_img=cropped_image,
                bucket_name=image_path["bucket"],
                object_name=object_name
            )

        # -------------------------
        # DF finale directe
        # -------------------------
        df_db_final = pl.DataFrame(rows_db)

        # -------------------------
        # DF à envoyer ML taxonomy
        # = tout sauf Pas d'espèce
        # -------------------------
        ids_final_db = (
            df_db_final["crop_id"].to_list()
            if len(df_db_final) > 0
            else []
        )

        df_ml = data_process.filter(
            ~pl.col("crop_id").is_in(ids_final_db)
        )

        LOGGER.info(
            f"Images vers ML taxonomy : {len(df_ml)}"
        )

        LOGGER.info(
            f"Images vers DB finale : {len(df_db_final)}"
        )

        return df_ml, df_db_final



def process_crop_annotations( df: pl.DataFrame,run_name: str,engine):
        """
        Traitement des annotations du projet Label Studio CROPS.

        Objectif :
        - filtrer les annotations déjà traitées
        - préparer les données pour insertion
        dans Bd_finale
        """

        LOGGER.info("Starting process_crop_annotations")

        # -------------------------
        # Extraction datetime depuis run_name
        # -------------------------
        run_time = datetime.datetime.strptime(
            run_name.replace("run_", ""),
            "%Y%m%d_%H%M%S"
        ).replace(tzinfo=datetime.timezone.utc)

        # -------------------------
        # Conversion datetime
        # -------------------------
        df = df.with_columns(
            pl.col("annotated_at").str.to_datetime(
                format="%Y-%m-%dT%H:%M:%S%.fZ",
                time_zone="UTC"
            )
        )

        # -------------------------
        # Filtrage temporel
        # -------------------------
        data_filtered = df.filter(
            pl.col("task_created_date") < run_time
        )

        # -------------------------
        # Filtrage déjà traités
        # -------------------------
        data_process = filter_processed_crop_annotations(
            data_filtered,
            engine
        )

        LOGGER.info(
            f"Nombre annotations crop à traiter : {len(data_process)}"
        )

        rows_db = []

        # -------------------------
        # Préparation DB finale
        # -------------------------
        for row in data_process.iter_rows(named=True):

            # =========================
            # Prediction correcte
            # =========================
            if row["decision"] == "Prédiction correcte":
                nom_scientifique = row["espece_pred"]
                validee = True

            # =========================
            # Espèce corrigée
            # =========================
            elif row["decision"] == "Corriger l'espèce":

                nom_scientifique = row["espece_corigée"]
                validee = True

            # =========================
            # Cas fallback
            # =========================
            else:

                nom_scientifique = None
                validee = None

            rows_db.append({

                "id_observation": row["id_observation"],
                "nom_scientifique": nom_scientifique,
                "validee": validee,
                "identifiable": True,
                "annotateur": row["annotator"],
                "source": "labelstudio_crop"
            })

        df_db_final = pl.DataFrame(rows_db)

        LOGGER.info(
            f"Observations prêtes pour DB finale : {len(df_db_final)}"
        )

        return df_db_final