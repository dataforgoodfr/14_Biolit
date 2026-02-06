import itertools

import polars as pl
import structlog
from polars import col

from biolit import DATADIR, EXPORTDIR
from biolit.taxref import TAXREF_HIERARCHY, format_taxref
from biolit.visualisation.species_distribution import plot_species_distribution

LOGGER = structlog.get_logger()


def export_observations():
    format_taxref()
    format_observations()
    biolit_df = pl.read_parquet(DATADIR / "biolit_valid_observations.parquet")
    plot_species_distribution(biolit_df, fn=EXPORTDIR / "distribution_images.html")


def format_observations():
    fn = DATADIR / "export_biolit.csv"
    taxref = pl.read_parquet(DATADIR / "taxref.parquet")
    biolit = (
        pl.read_csv(fn, separator=";", truncate_ragged_lines=True)
        .rename(lambda c: c.replace(" - observation", "").lower().replace(" ", "_"))
        .with_columns(
            col("nom_scientifique").str.to_lowercase(),
            col("espece_identifiable_?").fill_null("Identifiable"),
        )
        .filter(
            col("validee")
            & ~col("espece_identifiable_?").is_in(["non-identifiable", "Ne sais pas"])
        )
        .join(taxref, how="left", left_on="nom_scientifique", right_on="species_name")
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
        frame = frame.with_columns((prefix + col(name).fill_null("NA")).alias(name))
    return frame


def _observation_quality(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.pipe(_check_validated_non_identifiable)
        .pipe(_check_missing_nom)
        .pipe(_check_missing_taxref)
    )


def _check_missing_taxref(frame: pl.DataFrame) -> pl.DataFrame:
    missing_taxref_filter = (
        col("species_id").is_null() & col("nom_scientifique").is_not_null()
    )
    missing_taxref = (
        frame.filter(missing_taxref_filter)
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
    if errors.is_empty():
        return frame

    errors.write_csv(DATADIR / "biolit_observation_validated_non_identifiable.csv")
    LOGGER.warning("observation_quality_validated_non_identifiable", n_obs=len(errors))
    with pl.Config(fmt_str_lengths=50):
        print(errors)
    return frame.filter(~filt)


def learnable_taxonomy(
    frame: pl.DataFrame, current_taxon: str, levels: list[str], n_learnable: int
) -> dict:
    """
    Liste les niveau taxonomiques les plus bas predictibles.
    """
    next_level = levels[0] if levels else "nom_scientifique"
    level_agg = frame.group_by(next_level).agg(col("n_obs").sum())
    learnables = level_agg.filter(col("n_obs") >= n_learnable)[next_level].to_list()

    unlearnable = level_agg.filter(col("n_obs") < n_learnable)
    remaining_taxon = []
    if not unlearnable.is_empty():
        autre_keyword = (
            "AUTRE -- " if unlearnable["n_obs"].sum() >= n_learnable else "NO_STATS -- "
        )
        remaining_taxon.append(autre_keyword + current_taxon)

    if not levels:
        return learnables + remaining_taxon

    next_frame = frame.group_by(levels + ["nom_scientifique"]).agg(col("n_obs").sum())

    learnable_sublevels = [
        learnable_taxonomy(
            next_frame.filter(col(next_level) == taxon),
            taxon,
            levels[1:],
            n_learnable=n_learnable,
        )
        or [taxon]
        for taxon in learnables
    ] + [remaining_taxon]
    return sorted(set(itertools.chain(*learnable_sublevels)))
