"""
Pipeline complet crop → classification
====================================================================

"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import structlog
from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE,
    RESULTS_DIR,
    S3_BUCKET_NAME,
    CONFIDENCE_THRESHOLD,
    MARGIN_MIN,
)
from db import insert_taxonomy_predictions
from classifier_s3 import load_crops_with_images
from classifier_infer_v2 import load_model, predict_batch
from classifier_bioclip import BioCLIPExtractor

LOGGER = structlog.get_logger()
load_dotenv()


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ════════════════════════════════════════════════════════════════════════════

def run_classification(
    run_name: str,
    bucket: str = S3_BUCKET_NAME,
    limit: Optional[int] = None,
    threshold: float = CONFIDENCE_THRESHOLD,
    margin_min: float = MARGIN_MIN,
    device: str = None,
    no_db: bool = False,
) -> pl.DataFrame:
    """
    Exécute la classification sur les crops S3.
    
    Args:
        run_name: Nom du run (dossier S3)
        bucket: Nom du bucket S3
        limit: Limite nombre d'images (pour test)
        threshold: Seuil de confiance
        margin_min: Marge minimum
        device: Device PyTorch (cpu/cuda)
        
    Retourne:
        DataFrame Polars avec les prédictions
    """
    device = device or str(DEVICE)
    print(f"\n=== Classification: run={run_name} | device={device} ===")
    
    # 1. Charger le modèle
    print("  Chargement du modèle...")
    model = load_model()
    bioclip = BioCLIPExtractor()
    
    # 2. Charger les crops depuis S3
    print("  Chargement des crops depuis S3...")
    crops_data = load_crops_with_images(run_name, bucket, limit)
    
    if not crops_data:
        print("  ⚠ Aucun crop trouvé!")
        return pl.DataFrame()
    
    print(f"  {len(crops_data)} crops chargés")
    
    # 3. Prédiction batch
    print("  Prédiction en cours...")
    images = [img for img, _ in crops_data]
    metadata = [meta for _, meta in crops_data]
    
    results = predict_batch(
        images,
        model,
        bioclip,
        threshold=threshold,
        margin_min=margin_min
    )
    
    # 4. Construire le DataFrame résultat
    rows = []
    for meta, pred in zip(metadata, results):
        rows.append({
            "id_observation": meta["id_observation"],
            "id_crops": meta["id_crops"],
            "regne_yolo": meta["regne"],
            "confiance_yolo": meta["confiance"],
            "path_s3": meta.get("path_s3"),
            "best_level": pred["best_level"],
            "best_label": pred["best_label"],
            "best_score": pred["best_score"],
            "path": pred["path"],
            "margin": pred["margin"],
            "regne": pred.get("regne"),
            "phylum": pred.get("phylum"),
            "classe": pred.get("classe"),
            "ordre": pred.get("ordre"),
            "famille": pred.get("famille"),
            "species_name": pred.get("species_name"),
        })

    df = pl.DataFrame(rows)

    # 5. Sauvegarder en local (toujours)
    output_path = RESULTS_DIR / f"predictions_{run_name}.parquet"
    df.write_parquet(output_path)
    print(f"  Résultats → {output_path}")

    # 6. Logger dans PostgreSQL 
    if no_db:
        print("  [--no-db] Postgres ignoré — résultats dans le parquet uniquement")
    else:
        try:
            insert_taxonomy_predictions(df, run_name)
            print("  Prédictions insérées dans ml_taxonomy (PostgreSQL)")
        except Exception as e:
            print(f"  ⚠ Postgres non disponible ({e})")
            print("  → Utilise --no-db pour tester sans PostgreSQL")
    
    # Statistiques
    n_species = (df["best_level"] == "species_name").sum()
    n_family = (df["best_level"] == "famille").sum()
    n_other = len(df) - n_species - n_family
    
    print(f"\n=== Résultats ===")
    print(f"  Total: {len(df)}")
    print(f"  Espèce: {n_species} ({n_species/len(df):.1%})")
    print(f"  Famille: {n_family} ({n_family/len(df):.1%})")
    print(f"  Autre: {n_other} ({n_other/len(df):.1%})")
    
    return df




# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def flow_ml_classification(
    crops_images: dict[str, Image.Image],
    df_crops: pl.DataFrame,
    threshold: float = CONFIDENCE_THRESHOLD,
    margin_min: float = MARGIN_MIN,
    device: str = None,
) -> pl.DataFrame:
    """
    Classification taxonomique pure — pas de S3, pas de DB.

    Args:
        crops_images: dict { id_crops → PIL.Image } produit par flow_ml_crops
        df_crops: DataFrame avec colonnes id_crops, id_observation, regne, confiance, path_s3
        threshold: seuil de confiance BioCLIP
        margin_min: marge minimum entre top-1 et top-2

    Retourne:
        DataFrame avec les prédictions taxonomiques
    """
    if not crops_images:
        LOGGER.info("flow_ml_classification: aucun crop à classifier")
        return pl.DataFrame()

    device = device or str(DEVICE)
    LOGGER.info("flow_ml_classification start", n_crops=len(crops_images), device=device)

    model = load_model()
    bioclip = BioCLIPExtractor()

    id_crops_list = list(crops_images.keys())
    images_list = [crops_images[k] for k in id_crops_list]

    results = predict_batch(images_list, model, bioclip, threshold=threshold, margin_min=margin_min)

    meta_by_id = {row["id_crops"]: row for row in df_crops.to_dicts()}

    rows = []
    for id_crops, pred in zip(id_crops_list, results):
        meta = meta_by_id.get(id_crops, {})
        rows.append({
            "id_crops": id_crops,
            "id_observation": meta.get("id_observation"),
            "regne_yolo": meta.get("regne"),
            "confiance_yolo": meta.get("confiance"),
            "path_s3": meta.get("path_s3"),
            "best_level": pred["best_level"],
            "best_label": pred["best_label"],
            "best_score": pred["best_score"],
            "path": pred["path"],
            "margin": pred["margin"],
            "regne": pred.get("regne"),
            "phylum": pred.get("phylum"),
            "classe": pred.get("classe"),
            "ordre": pred.get("ordre"),
            "famille": pred.get("famille"),
            "species_name": pred.get("species_name"),
        })

    df = pl.DataFrame(rows)
    LOGGER.info("flow_ml_classification done", n_predictions=len(df))
    return df


def main():
    parser = argparse.ArgumentParser(description="Pipeline classification")
    parser.add_argument("--run_name", type=str, help="Nom du run S3")
    parser.add_argument("--classify_only", action="store_true",
                        help="Classification uniquement (skip crop_inference)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite nombre d'images")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--margin", type=float, default=MARGIN_MIN)
    parser.add_argument("--device", type=str, default=None,
                        help="Device PyTorch (cpu/cuda)")
    parser.add_argument("--no-db", action="store_true",
                        help="Ne pas écrire dans PostgreSQL (mode test local)")
    args = parser.parse_args()
    
    if args.run_name:
        # Classification depuis S3
        run_classification(
            args.run_name,
            limit=args.limit,
            threshold=args.threshold,
            margin_min=args.margin,
            device=args.device,
            no_db=args.no_db,
        )
    else:
        print("Usage:")
        print("  python pipeline_classification.py --run_name <run>")
        print("  python pipeline_classification.py --source <dossier> --local")
        print("  python pipeline_classification.py --run_name <run> --classify_only")


if __name__ == "__main__":
    main()