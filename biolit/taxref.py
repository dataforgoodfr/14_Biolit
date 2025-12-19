import polars as pl
import structlog
from polars import col

from biolit import DATADIR

TAXREF_HIERARCHY = ["regne", "phylum", "classe", "ordre", "famille", "sous_famille"]
LOGGER = structlog.get_logger()


def format_taxref():
    fn = DATADIR / "TAXREF_v18_2025" / "TAXREFv18.txt"
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
        .select(["cd_nom", "lb_nom", "priority"] + TAXREF_HIERARCHY)
    )
    _check_duplicates(taxref)
    taxref = (
        taxref.sort(["lb_nom", "priority"], descending=[False, True])
        .unique("lb_nom")
        .drop("priority")
    )
    taxref.write_parquet(DATADIR / "taxref.parquet")


def _check_duplicates(frame: pl.DataFrame):
    frame = frame.sort("lb_nom").filter(col("lb_nom").is_duplicated())
    if frame.is_empty():
        return
    frame.write_csv(DATADIR / "taxref_duplicate_species.csv")
    LOGGER.warning(
        "taxref_duplicate_species",
        n_species=len(frame),
        n_names=frame["lb_nom"].n_unique(),
    )
