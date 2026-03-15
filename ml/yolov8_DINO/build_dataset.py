import logging
import sys
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# LOGGING
LOG_FILE = Path("logs/build_dataset.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("biolit")
logger.setLevel(logging.INFO)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
)

_file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%Y-%m-%d %H:%M:%S")
)

logger.addHandler(_stream_handler)
logger.addHandler(_file_handler)


# CONFIG
DATA_PATH  = "dataset_biolit/export_biolit.csv"
BASE_DIR   = Path("dataset_biolit")
IMAGES_DIR = BASE_DIR / "images"
TIMEOUT    = 10
VALID_EXTS = {"jpg", "jpeg", "png", "webp"}

USE_COLS = [
    "ID - N1",
    "images - observation",
    "Nom commun - observation",
    "validee - observation",
    "espece identifiable ? - observation",
]


# PIPELINE
def load(path: str) -> pd.DataFrame:
    """Charge le CSV, garde uniquement les observations validées et identifiables."""
    logger.info("Chargement : %s", path)
    df = pd.read_csv(path, usecols=USE_COLS)

    df = df[
        (df["validee - observation"] == "TRUE") &
        (df["espece identifiable ? - observation"] == "Identifiable")
    ].reset_index(drop=True)

    logger.info("%d observations valides et identifiables", len(df))
    return df


def explode_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par image."""
    df = (
        df.assign(**{"images - observation": df["images - observation"].str.strip().str.split("|")})
        .explode("images - observation")
        .rename(columns={"images - observation": "image_url"})
        .assign(image_url=lambda d: d["image_url"].str.strip())
        .reset_index(drop=True)
    )
    logger.info("%d URLs d'images après explosion", len(df))
    return df


def make_filename(row: pd.Series, idx: int) -> str:
    id_n1 = str(row["ID - N1"])
    nom   = str(row["Nom commun - observation"]) if pd.notna(row["Nom commun - observation"]) else "inconnu"
    nom   = nom.strip().replace(" ", "_").replace("/", "-")
    ext   = row["image_url"].split(".")[-1].split("?")[0].lower()
    ext   = ext if ext in VALID_EXTS else "jpg"
    return f"{id_n1}_{nom}_{idx}.{ext}"


def download(df: pd.DataFrame) -> pd.DataFrame:
    """
    Télécharge chaque image.
    Retourne uniquement les lignes dont le téléchargement a réussi.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    rows_ok = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Téléchargement"):
        filename = make_filename(row, idx)
        dest     = IMAGES_DIR / filename

        if dest.exists():
            logger.debug("Skip (déjà présente) : %s", filename)
            rows_ok.append({**row, "filename": filename, "filepath": str(dest)})
            continue

        try:
            r = requests.get(row["image_url"], timeout=TIMEOUT)
            r.raise_for_status()
            dest.write_bytes(r.content)
            rows_ok.append({**row, "filename": filename, "filepath": str(dest)})
            logger.debug("OK  %s", filename)
        except Exception as e:
            logger.warning("ÉCHEC  %s  →  %s", row["image_url"], e)

    downloaded = pd.DataFrame(rows_ok).reset_index(drop=True)
    logger.info(
        "Téléchargement terminé — %d/%d images récupérées",
        len(downloaded), len(df),
    )
    return downloaded


def save_metadata(df: pd.DataFrame) -> None:
    cols = ["ID - N1", "Nom commun - observation", "image_url", "filename", "filepath"]
    dest = BASE_DIR / "metadata.csv"
    df[cols].to_csv(dest, index=False)
    logger.info("metadata.csv : %d lignes → %s", len(df), dest)


# MAIN
def build_dataset():
    logger.info("════════ START ════════")

    df       = load(DATA_PATH)
    df       = explode_urls(df)
    df_ok    = download(df)

    save_metadata(df_ok)

    logger.info(
        "Terminé — %d images | %d espèces | dossier : %s",
        len(df_ok),
        df_ok["Nom commun - observation"].nunique(),
        IMAGES_DIR.resolve(),
    )
    logger.info("════════ END ════════")


if __name__ == "__main__":
    build_dataset()
