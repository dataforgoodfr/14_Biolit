import polars as pl
import structlog
from polars import col
from pathlib import Path
import requests
import zipfile
import shutil

from biolit import DATADIR, TAXREFURL

TAXREF_HIERARCHY = ["regne", "phylum", "classe", "ordre", "famille", "sous_famille"]
LOGGER = structlog.get_logger()


def format_taxref():
    fn = DATADIR / "TAXREF_v18_2025" / "TAXREFv18.txt"

    if not _check_file_existence(fn):
        _download_taxref(fn)

    taxref = (
        pl.read_csv(fn, separator="\t")
        .rename(str.lower)
        .with_columns(
            col("lb_nom").str.to_lowercase(),
            (
                col("sous_famille").is_not_null()
                + col("famille").is_not_null() * 10
                + col("ordre").is_not_null() * 100
                + col("classe").is_not_null() * 1000
            ).alias("priority"),
        )
        .select(
            [
                col("cd_nom").alias("species_id"),
                col("lb_nom").alias("species_name"),
                "priority",
            ]
            + TAXREF_HIERARCHY
        )
    )
    _check_duplicates(taxref)
    taxref = (
        taxref.sort(["species_name", "priority"], descending=[False, True])
        .unique("species_name")
        .drop("priority")
    )
    taxref.write_parquet(DATADIR / "taxref.parquet")


def _check_duplicates(frame: pl.DataFrame):
    frame = frame.sort("species_name").filter(col("species_name").is_duplicated())
    if frame.is_empty():
        return
    frame.write_csv(DATADIR / "taxref_duplicate_species.csv")
    LOGGER.warning(
        "taxref_duplicate_species",
        n_species=len(frame),
        n_names=frame["species_name"].n_unique(),
    )


def _check_file_existence(file: Path):
    if not file.exists():
        return False

    if not file.is_file():
        LOGGER.fatal(
            "The following path has been created, but it is not a standard file",
            path=file
        )


def _download_taxref(targetpath: Path):
    # Download and save the TaxRef zip file to a temp file
    r = requests.get(TAXREFURL)

    tmpfile = Path("tmp_taxref.zip")
    with open(tmpfile, 'wb') as f:
        for chunk in r:
            if chunk:
                f.write(chunk)

    # Extract the taxonomy itself to proper path
    with zipfile.ZipFile(tmpfile) as z:
        with z.open('TAXREFv18.txt') as zf, open(targetpath, 'wb') as f:
            shutil.copyfileobj(zf, f)

    # Cleanup the zip archive and check all went well
    tmpfile.unlink()

    if not targetpath.is_file():
        LOGGER.fatal("Didn't manage to properly download the TaxRef")