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
_stream_handler.setLevel(logging.INFO)
_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
)

_file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
)

logger.addHandler(_stream_handler)
logger.addHandler(_file_handler)


# CONFIGS
DATA_PATH   = "dataset_biolit/export_biolit.csv"
BASE_DIR    = Path("dataset_biolit")
TIMEOUT     = 10
VALID_EXTS  = {"jpg", "jpeg", "png", "webp"}

USE_COLS = [
    "ID - N1",
    "images - observation",
    "Nom commun - observation",
    "validee - observation",
    "espece identifiable ? - observation",
]

IDENTIFIABLE_VALUES = ["Identifiable", "Non identifiable"]

SUBFOLDER_MAP = {
    "Identifiable":     "identifiable",
    "Non identifiable": "non_identifiable",
}


# FONCTION UTILS
def load_and_filter(path: str) -> pd.DataFrame:
    """Charge le CSV et filtre les observations validées."""
    logger.info("Chargement du CSV : %s", path)
    data = pd.read_csv(path, usecols=USE_COLS)
    total   = len(data)
    filtered = data[data["validee - observation"] == "TRUE"]
    logger.info(
        "CSV chargé — %d lignes totales, %d après filtrage (validée=TRUE)",
        total, len(filtered),
    )
    return filtered


def explode_image_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Split les URLs multiples (séparées par |) en une ligne par image."""
    logger.debug("Explosion des URLs multiples...")
    result = (
        df.copy()
        .assign(**{
            "images - observation": df["images - observation"]
            .str.strip()
            .str.split("|")
        })
        .explode("images - observation")
        .rename(columns={"images - observation": "image_url"})
        .assign(image_url=lambda d: d["image_url"].str.strip())
        .reset_index(drop=True)
    )
    logger.info("Explosion terminée — %d lignes (une par image)", len(result))
    return result


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne, renomme et enrichit les colonnes utiles."""
    logger.debug("Préparation du dataset...")
    dataset = (
        df[[
            "ID - N1",
            "Nom commun - observation",
            "espece identifiable ? - observation",
            "image_url",
        ]]
        .copy()
        .rename(columns={"espece identifiable ? - observation": "identifiable"})
        .pipe(lambda d: d[d["identifiable"].isin(IDENTIFIABLE_VALUES)])
        .reset_index(drop=True)
    )

    dataset["filename"]  = [_make_filename(row, i) for i, row in dataset.iterrows()]
    dataset["subfolder"] = dataset["identifiable"].map(SUBFOLDER_MAP)
    dataset["filepath"]  = (
        str(BASE_DIR / "images") + "/" + dataset["subfolder"] + "/" + dataset["filename"]
    )

    counts = dataset["subfolder"].value_counts()
    logger.info(
        "Dataset préparé — identifiable: %d | non_identifiable: %d | total: %d",
        counts.get("identifiable", 0),
        counts.get("non_identifiable", 0),
        len(dataset),
    )
    return dataset


def _make_filename(row: pd.Series, idx: int) -> str:
    id_n1 = str(row["ID - N1"])
    nom   = str(row["Nom commun - observation"]) if pd.notna(row["Nom commun - observation"]) else "inconnu"
    nom   = nom.strip().replace(" ", "_").replace("/", "-")
    ext   = row["image_url"].split(".")[-1].split("?")[0].lower()
    ext   = ext if ext in VALID_EXTS else "jpg"
    return f"{id_n1}_{nom}_{idx}.{ext}"


def create_folder_structure() -> None:
    """Crée l'arborescence du dataset."""
    logger.debug("Création de l'arborescence des dossiers...")
    for sub in [
        "images/identifiable", "images/non_identifiable",
        "labels/identifiable", "labels/non_identifiable",
    ]:
        (BASE_DIR / sub).mkdir(parents=True, exist_ok=True)
    logger.info("Arborescence créée dans : %s", BASE_DIR.resolve())


def download_images(dataset: pd.DataFrame) -> list[dict]:
    """Télécharge les images, skippe celles déjà présentes."""
    failed  = []
    skipped = 0
    success = 0

    logger.info("Démarrage du téléchargement (%d images)...", len(dataset))

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Téléchargement"):
        dest = Path(row["filepath"])
        if dest.exists():
            skipped += 1
            logger.debug("Skipped (déjà présente) : %s", dest.name)
            continue
        try:
            response = requests.get(row["image_url"], timeout=TIMEOUT)
            response.raise_for_status()
            dest.write_bytes(response.content)
            success += 1
            logger.debug("OK  %s", dest.name)
        except Exception as e:
            failed.append({"url": row["image_url"], "erreur": str(e)})
            logger.warning("ÉCHEC  %s  →  %s", row["image_url"], e)

    logger.info(
        "Téléchargement terminé — succès: %d | skipped: %d | échecs: %d",
        success, skipped, len(failed),
    )
    return failed


def save_metadata(dataset: pd.DataFrame) -> None:
    dest = BASE_DIR / "metadata.csv"
    dataset.to_csv(dest, index=False)
    logger.info("Métadonnées sauvegardées : %s (%d lignes)", dest, len(dataset))


def save_yaml(dataset: pd.DataFrame) -> None:
    classes = sorted(dataset["Nom commun - observation"].dropna().unique().tolist())
    yaml_content = (
        f"# Dataset Biolit - config YOLO\n"
        f"path: {BASE_DIR.resolve()}\n"
        f"train: images/identifiable\n"
        f"val: images/identifiable   # à splitter train/val/test\n\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    dest = BASE_DIR / "data.yaml"
    dest.write_text(yaml_content)
    logger.info("data.yaml sauvegardé : %d classes", len(classes))


def print_report(dataset: pd.DataFrame, failed: list[dict]) -> None:
    counts = dataset["subfolder"].value_counts()
    report = f"""
    Dataset sauvegardé dans : {BASE_DIR.resolve()}

    images/identifiable/ -> {counts.get('identifiable', 0):>5} images
    images/non_identifiable/ -> {counts.get('non_identifiable', 0):>5} images
    labels/ -> prêt pour annotation YOLO
    metadata.csv -> {len(dataset):>5} lignes
    data.yaml ->  {dataset['Nom commun - observation'].nunique():>5} classes

    Succès  : {len(dataset) - len(failed)}
    Échecs  : {len(failed)}
    """
    print(report)
    logger.info("=== RAPPORT FINAL ===")
    logger.info("identifiable: %d | non_identifiable: %d", counts.get("identifiable", 0), counts.get("non_identifiable", 0))
    logger.info("Total lignes metadata: %d | Classes: %d", len(dataset), dataset["Nom commun - observation"].nunique())
    logger.info("Succès: %d | Échecs: %d", len(dataset) - len(failed), len(failed))
    logger.info("Logs sauvegardés dans : %s", LOG_FILE.resolve())


# RUNNER L'EXTRACT
def build_dataset():
    logger.info("════════ START build_dataset ════════")

    logger.info("Chargement et filtrage...")
    raw       = load_and_filter(DATA_PATH)
    exploded  = explode_image_urls(raw)
    dataset   = prepare_dataset(exploded)

    logger.info("%d images à traiter", len(dataset))

    create_folder_structure()

    logger.info("Téléchargement des images...")
    failed = download_images(dataset)

    logger.info("Sauvegarde des métadonnées...")
    save_metadata(dataset)
    save_yaml(dataset)

    if failed:
        dest = BASE_DIR / "failed_downloads.csv"
        pd.DataFrame(failed).to_csv(dest, index=False)
        logger.warning("%d échec(s) enregistré(s) dans : %s", len(failed), dest)

    print_report(dataset, failed)
    logger.info("════════ END build_dataset ════════")


if __name__ == "__main__":
    build_dataset()
