"""
Le DataFrame produit a toujours les mêmes colonnes :
    image_path   : chemin absolu vers l'image
    source_image : nom du fichier
    id_n1        : extrait du nom de fichier
    species_name, regne, phylum, classe, ordre, famille, sous_famille
"""

import re
import pandas as pd
from pathlib import Path

from config import IMAGES_DIR, DATA_DIR, TAXONOMY_FILE, TAXONOMY_LEVELS


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _parse_id(filename: str) -> str | None:
    """Extrait l'id_n1 depuis {id_n1}_{nom}_{idx}.{ext}"""
    m = re.match(r'^(\d+)_', filename)
    return m.group(1) if m else None


def _parse_nom_commun(filename: str) -> str | None:
    """
    Extrait le nom commun normalisé depuis {id_n1}_{nom_commun}_{idx}.{ext}
    """
    stem  = Path(filename).stem       # "1234_fucus_spiralis_42"
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    # Tout entre le premier segment (id) et le dernier (idx pandas)
    return "_".join(parts[1:-1])


def _normalize_nom(nom: str) -> str:
    
    return str(nom).strip().replace(" ", "_").replace("/", "-")


def build_dataset(
    images_dir: Path = None,
    observations_csv: Path = None,
    taxonomy_path: Path = None,
) -> pd.DataFrame:
    """
    Construit le DataFrame labellisé depuis un dossier d'images.

    Paramètres :
        images_dir       : dossier contenant les images (défaut: IMAGES_DIR/identifiable)
        export_biolit : CSV avec ID - N1 + Nom scientifique
        taxonomy_path    : parquet taxref 

    Retourne un DataFrame avec image_path + colonnes taxonomiques.
    """
    images_dir       = Path(images_dir) if images_dir else IMAGES_DIR / "identifiable"
    observations_csv = Path(observations_csv) if observations_csv else DATA_DIR / "export_biolit.csv"
    taxonomy_path    = Path(taxonomy_path) if taxonomy_path else TAXONOMY_FILE

    # ── Collecter les images ──────────────────────────────────────────────────
    image_paths = [p for p in images_dir.iterdir()
                   if p.suffix.lower() in VALID_EXTS]
    print(f"{len(image_paths)} images trouvées dans {images_dir}")

    df = pd.DataFrame({
        "image_path":   [str(p) for p in image_paths],
        "source_image": [p.name for p in image_paths],
    })
    df["id_n1"]      = df["source_image"].apply(_parse_id)
    df["nom_commun"] = df["source_image"].apply(_parse_nom_commun)

    # ── Jointure observations → nom scientifique ──────────────────────────────
    obs = pd.read_csv(observations_csv)
    obs["id_n1"]      = obs["ID - N1"].astype(str).str.strip()
    obs["_species"]   = obs["Nom scientifique - observation"].str.strip()
    obs["nom_commun"] = obs["Nom commun - observation"].apply(
        lambda x: _normalize_nom(x) if pd.notna(x) else None
    )

    obs_valid = obs[obs["_species"].notna()][["id_n1", "nom_commun", "_species"]]

    # Jointure composite id_n1 + nom_commun — gère les doublons d'id_n1
    merged = df.merge(obs_valid, on=["id_n1", "nom_commun"], how="left")

    # Fallback id_n1 seul pour les cas non matchés
    # (ex: nom commun manquant ou légèrement différent)
    no_match = merged["_species"].isna()
    if no_match.any():
        obs_id_only = obs_valid.drop_duplicates("id_n1")[["id_n1", "_species"]]
        fallback    = df[no_match][["id_n1"]].merge(obs_id_only, on="id_n1", how="left")
        merged.loc[no_match, "_species"] = fallback["_species"].values
        n_fallback = no_match.sum() - merged["_species"].isna().sum()
        if n_fallback > 0:
            print(f"  ⚠ {n_fallback} images matchées via id_n1 seul (nom commun non trouvé)")

    df = merged.rename(columns={"_species": "Nom scientifique - observation"})

    n_with_species = df["Nom scientifique - observation"].notna().sum()
    print(f"Observations matchées : {n_with_species}/{len(df)}")

    # ── Jointure taxonomie ────────────────────────────────────────────────────
    tax = pd.read_parquet(taxonomy_path)
    tax.columns = tax.columns.str.strip().str.lower().str.replace(" ", "_")
    for col in tax.select_dtypes("object").columns:
        tax[col] = tax[col].str.strip()

    tax["_key"] = tax["species_name"].str.lower()
    df["_key"]  = df["Nom scientifique - observation"].str.lower()

    df = df.merge(tax[["_key"] + TAXONOMY_LEVELS], on="_key", how="left")
    df.drop(columns=["_key"], inplace=True)

    n_matched = df["species_name"].notna().sum()
    no_obs    = df["Nom scientifique - observation"].isna().sum()
    no_tax    = (df["Nom scientifique - observation"].notna() & df["species_name"].isna()).sum()

    print(f"Taxonomie matchée    : {n_matched}/{len(df)}")
    print(f"\nDétail des {len(df) - n_matched} images non utilisables :")
    print(f"  - Sans observation CSV    : {no_obs}")
    print(f"  - Observation sans taxref : {no_tax}")

    unmatched = (df[df["species_name"].isna()]["Nom scientifique - observation"]
                 .dropna().unique())
    if len(unmatched):
        print(f"\nEspèces absentes du taxref ({len(unmatched)}) :")
        for u in sorted(unmatched):
            print(f"  - {u}")

    if n_matched > 0:
        print(f"\nDistribution par règne :")
        print(df.groupby("regne").size().sort_values(ascending=False).to_string())
        print(f"\nTop 10 espèces :")
        print(df.groupby("species_name").size().sort_values(ascending=False).head(10).to_string())
        rare = (df.groupby("species_name").size() < 5).sum()
        print(f"\nEspèces avec < 5 images : {rare} / {df['species_name'].nunique()}")

    return df


if __name__ == "__main__":
    df = build_dataset()
    out = DATA_DIR / "classify_dataset.csv"
    df.to_csv(out, index=False)
    print(f"\nDataset sauvegardé : {out}")
    print(df[["source_image", "species_name", "regne", "famille"]].head(10).to_string())