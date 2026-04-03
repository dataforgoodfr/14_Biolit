import argparse
import unicodedata
import yaml
import cv2
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import get_logger

logger = get_logger("build_dataset")

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def sanitize_filenames(img_dir: Path):
    """Renomme les fichiers avec accents en ASCII (cv2.imread plante sur macOS sinon)."""
    for f in img_dir.iterdir():
        if not f.is_file():
            continue
        clean = "".join(
            c for c in unicodedata.normalize("NFKD", f.name)
            if not unicodedata.combining(c)
        )
        if clean != f.name:
            f.rename(f.parent / clean)
            logger.info("Renommé : %s → %s", f.name, clean)


def clean_non_images(img_dir: Path):
    """Écarte les fichiers non-image et les images corrompues dans un dossier tmp."""
    tmp = img_dir.parent / "_non_images_tmp"
    tmp.mkdir(exist_ok=True)
    removed = 0
    for f in img_dir.iterdir():
        if not f.is_file():
            continue
        ok = f.suffix.lower() in VALID_IMG_EXT and cv2.imread(str(f)) is not None
        if not ok:
            f.rename(tmp / f.name)
            logger.warning("Écarté : %s", f.name)
            removed += 1
    if removed:
        logger.info("%d fichier(s) non-image écartés dans _non_images_tmp/", removed)


def load_config(path: str = "configs/build_dataset.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config chargée : %s", path)
    return cfg


# PIPELINE TELECHARGEMENT DATASET

def load(path: str, use_cols: list) -> pd.DataFrame:
    """Charge le CSV, garde uniquement les observations validées et identifiables."""
    logger.info("Chargement : %s", path)
    df = pd.read_csv(path, usecols=use_cols)
    df = df[df["validee - observation"] == "TRUE"].reset_index(drop=True)
    logger.info("%d observations valides et identifiables", len(df))
    return df


def explode_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par image."""
    df = (
        df.assign(**{"images - observation": df["images - observation"].str.strip().str.split("|")})
        .explode("images - observation")
        .rename(columns={"images - observation": "image_url"})
        .assign(image_url=lambda d: d["image_url"].str.strip())
    )
    df = df[df["image_url"].notna() & (df["image_url"] != "")].reset_index(drop=True)
    logger.info("%d URLs d'images après explosion", len(df))
    return df


def make_filename(row: pd.Series, idx: int, valid_exts: set) -> str:
    id_n1 = str(row["ID - N1"])
    nom   = str(row["Nom commun - observation"]) if pd.notna(row["Nom commun - observation"]) else "inconnu"
    nom   = nom.strip().replace(" ", "_").replace("/", "-")
    url   = row["image_url"] if isinstance(row["image_url"], str) else ""
    ext   = url.split(".")[-1].split("?")[0].lower() if url else "jpg"
    ext   = ext if ext in valid_exts else "jpg"
    return f"{id_n1}_{nom}_{idx}.{ext}"


def download(df: pd.DataFrame, images_dir: Path, timeout: int, max_workers: int, valid_exts: set) -> pd.DataFrame:
    """Télécharge chaque image en parallèle. Retourne les lignes réussies."""
    images_dir.mkdir(parents=True, exist_ok=True)

    def _download_one(row, idx):
        filename = make_filename(row, idx, valid_exts)
        dest = images_dir / filename
        if dest.exists():
            logger.debug("Skip (déjà présente) : %s", filename)
            return {**row, "filename": filename, "filepath": str(dest)}
        try:
            r = requests.get(row["image_url"], timeout=timeout)
            r.raise_for_status()
            dest.write_bytes(r.content)
            logger.debug("OK  %s", filename)
            return {**row, "filename": filename, "filepath": str(dest)}
        except Exception as e:
            logger.warning("ÉCHEC  %s  →  %s", row["image_url"], e)
            return None

    futures = {}
    rows_ok = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in df.iterrows():
            futures[executor.submit(_download_one, row, idx)] = idx
        for future in tqdm(as_completed(futures), total=len(futures), desc="Téléchargement"):
            result = future.result()
            if result is not None:
                rows_ok.append(result)

    downloaded = pd.DataFrame(rows_ok).reset_index(drop=True)
    logger.info("Téléchargement terminé — %d/%d images récupérées", len(downloaded), len(df))
    return downloaded


# MAIN

def build_dataset(cfg: dict, limit: int = None):
    logger.info("════════ START ════════")

    images_dir = Path(cfg["images_dir"])
    use_cols   = list(cfg["columns"].values())

    df = load(cfg["data_path"], use_cols)
    df = explode_urls(df)

    if limit:
        df = df.head(limit)
        logger.info("Mode test : limité à %d images", limit)

    df_ok = download(df, images_dir, cfg["timeout"], cfg["max_workers"], set(cfg["valid_extensions"]))

    logger.info(
        "Terminé — %d images | %d espèces | dossier : %s",
        len(df_ok),
        df_ok["Nom commun - observation"].nunique(),
        images_dir.resolve(),
    )

    logger.info("Nettoyage du dossier images…")
    sanitize_filenames(images_dir)
    clean_non_images(images_dir)

    logger.info("════════ END ════════")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build_dataset.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limite le nombre d'images téléchargées (test)")
    args = parser.parse_args()

    build_dataset(load_config(args.config), limit=args.limit)
