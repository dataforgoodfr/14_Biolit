import polars as pl
from polars import col

from biolit.observations import learnable_taxonomy


class TestLearnableTaxonomy:
    def test_valid_deepest_taxon(self):
        inp = pl.DataFrame(
            {
                "nom_scientifique": ["herbe", "herbe"],
                "genre": ["plante", "plante"],
                "classe": ["chlorophyle", "chlorophyle"],
                "n_obs": 1,
            }
        )

        out = learnable_taxonomy(inp, "vivant", ["genre", "classe"], 2)
        exp = ["herbe"]
        assert out == exp

    def test_autre_taxons(self):
        inp = pl.DataFrame(
            {
                "genre": "plante",
                "classe": "chlorophyle",
                "nom_scientifique": ["herbe", "mousse", "fleur"],
                "n_obs": [10, 5, 5],
            }
        )

        out = learnable_taxonomy(inp, "vivant", ["genre", "classe"], 10)
        exp = ["AUTRE -- chlorophyle", "herbe"]
        assert out == exp

    def test_not_enough_autre_taxons(self):
        inp = pl.DataFrame(
            {
                "genre": "plante",
                "classe": "chlorophyle",
                "nom_scientifique": ["herbe", "mousse", "fleur"],
                "n_obs": [10, 1, 1],
            }
        )

        out = learnable_taxonomy(inp, "vivant", ["genre", "classe"], 10)
        exp = ["NO_STATS -- chlorophyle", "herbe"]
        assert out == exp
