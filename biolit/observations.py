import polars as pl
import structlog
from polars import col

from biolit import DATADIR
from biolit.taxref import TAXREF_HIERARCHY

LOGGER = structlog.get_logger()


def format_observations():
    fn = DATADIR / "export_biolit.csv"
    taxref = pl.read_parquet(DATADIR / "taxref.parquet")
    biolit = (
        pl.read_csv(fn)
        .rename(lambda c: c.replace(" - observation", "").lower().replace(" ", "_"))
        .with_columns(
            col("nom_scientifique").str.to_lowercase(),
            col("espece_identifiable_?").fill_null("Identifiable"),
        )
        .filter(
            col(
                "validee"
            )  # & ~col("espece_identifiable_?").is_in(["non-identifiable"])
        )
        .join(taxref, how="left", left_on="nom_scientifique", right_on="lb_nom")
        .pipe(full_upper_hierarchy)
        .pipe(_observation_quality)
    )

    LOGGER.info(
        "valid_observations",
        size=len(biolit),
        species=biolit["nom_scientifique"].n_unique(),
    )
    biolit.write_parquet(DATADIR / "biolit_valid_observations.parquet")


def full_upper_hierarchy(frame: pl.DataFrame) -> pl.DataFrame:
    """
    Fill all levels of hierachies with the complete name of the upper levels.

    horse -> animal | vertebrate | horse
    """
    for i, name in enumerate(TAXREF_HIERARCHY):
        prefix = (
            pl.lit("")
            if not i
            else (col(TAXREF_HIERARCHY[i - 1]).fill_null("NA") + pl.lit(" | "))
        )
        return frame.with_columns((prefix + col(name).fill_null("NA")).alias(name))


def _observation_quality(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.pipe(_check_missing_nom)
        .pipe(_check_missing_taxref)
        .pipe(_check_validated_non_identifiable)
    )


def _check_missing_taxref(frame: pl.DataFrame) -> pl.DataFrame:
    missing_taxref_filter = (
        col("cd_nom").is_null() & col("nom_scientifique").is_not_null()
    )
    missing_taxref = (
        frame.filter()
        .group_by("nom_scientifique")
        .agg(col("id").count().alias("n_observations"))
        .sort("n_observations", descending=True)
    )
    missing_taxref.write_csv(DATADIR / "observations_missing_taxref.csv")

    LOGGER.warning(
        "observation_quality_missing_taxref",
        missing_taxref_species=len(missing_taxref),
        missing_taxref_observations=missing_taxref["n_observations"].sum(),
    )
    with pl.Config(tbl_rows=100):
        print(missing_taxref)
    return frame.filter(~missing_taxref_filter)


def _check_missing_nom(frame: pl.DataFrame) -> pl.DataFrame:
    missing_filter = col("nom_scientifique").is_null()
    missing_nom = frame.filter(missing_filter).select(
        "validee", "espece_identifiable_?", "lien"
    )

    missing_nom.write_csv(DATADIR / "biolit_observation_missing_nom.csv")
    LOGGER.warning(
        "observation_quality_missing_nom",
        missing_nom=len(missing_nom),
    )
    with pl.Config(fmt_str_lengths=500):
        print(missing_nom)
    return frame.filter(~missing_filter)


def _check_validated_non_identifiable(frame: pl.DataFrame) -> pl.DataFrame:
    filt = col("espece_identifiable_?") != "Identifiable"
    errors = frame.filter(filt).select(
        "lien", "espece_identifiable_?", "nom_scientifique"
    )
    errors.write_csv(DATADIR / "biolit_observation_validated_non_identifiable.csv")
    LOGGER.warning("observation_quality_validated_non_identifiable", n_obs=len(errors))
    with pl.Config(fmt_str_lengths=50):
        print(errors)
    return frame.filter(~filt)
