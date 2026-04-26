"""
db.py — Logging des prédictions dans PostgreSQL
================================================
Utilisé par pipeline_classification.py pour persister les résultats.
En mode local (--no-db), les fonctions ne sont pas appelées.
"""

import os
import structlog
import polars as pl
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

LOGGER = structlog.get_logger()
load_dotenv()


def get_engine():
    url = os.getenv("POSTGRES_URL")
    if not url:
        raise ValueError(
            "POSTGRES_URL manquant dans .env — "
            "utilise --no-db pour tester sans PostgreSQL"
        )
    return create_engine(url)


def insert_taxonomy_predictions(df: pl.DataFrame, run_name: str) -> None:
    """
    Insère les prédictions de classification dans la table ml_taxonomy.

    Colonnes attendues dans df :
        id_observation, id_crops, regne_yolo, confiance_yolo,
        best_label, path_s3 (optionnel)

    Colonnes nullable (mises à NULL si absentes) :
        latitude, longitude, lien_doris
    """
    engine = get_engine()

    rows = []
    for row in df.iter_rows(named=True):
        rows.append({
            "run_name": run_name,
            "id_crops": row["id_crops"],
            "id_observation": row.get("id_observation"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "regne": row.get("regne_yolo"),
            "confiance": row.get("confiance_yolo"),
            "path_s3": row.get("path_s3"),
            "nom_scientifique": row.get("best_label"),
            "lien_doris": row.get("lien_doris"),
        })

    with engine.begin() as conn:
        for row in rows:
            conn.execute(
                text("""
                    INSERT INTO ml_taxonomy (
                        run_name, id_crops, id_observation,
                        latitude, longitude,
                        regne, confiance, path_s3,
                        nom_scientifique, lien_doris
                    ) VALUES (
                        :run_name, :id_crops, :id_observation,
                        :latitude, :longitude,
                        :regne, :confiance, :path_s3,
                        :nom_scientifique, :lien_doris
                    )
                    ON CONFLICT (id_crops) DO UPDATE SET
                        run_name        = EXCLUDED.run_name,
                        nom_scientifique = EXCLUDED.nom_scientifique,
                        confiance       = EXCLUDED.confiance,
                        lien_doris      = EXCLUDED.lien_doris
                """),
                row,
            )

    LOGGER.info("ml_taxonomy: %d lignes insérées (run=%s)", len(rows), run_name)
