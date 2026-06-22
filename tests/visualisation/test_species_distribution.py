from pathlib import Path
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal

from biolit.visualisation.species_distribution import create_species_graph_properties


@patch(
    "biolit.visualisation.species_distribution.TAXREF_HIERARCHY",
    ["regne", "classe"],
)
@patch(
    "biolit.visualisation.species_distribution.LIMIT_LEARNABLE_NODES",
    2,
)
@patch(
    "biolit.visualisation.species_distribution.DATADIR",
    Path("/tmp"),
)
class TestCreateSpeciesGraphProperties:
    def test_create_species_graph_properties(self):
        frame = pl.DataFrame(
            {
                "nom_scientifique": ["herbe", "herbe"],
                "regne": ["plante", "plante"],
                "classe": ["chlorophyle", "chlorophyle"],
                "species_id": 1,
            }
        )
        edges, nodes = create_species_graph_properties(frame)

        exp_edges = pl.DataFrame(
            [
                pl.Series("value", [2, 2], dtype=pl.UInt32),
                pl.Series(
                    "color", ["rgb(31, 119, 180)", "rgb(31, 119, 180)"], dtype=pl.Utf8
                ),
                pl.Series("source", [0, 1], dtype=pl.UInt32),
                pl.Series("target", [2, 0], dtype=pl.UInt32),
            ]
        )
        assert_frame_equal(edges, exp_edges)
        exp_nodes = pl.DataFrame(
            {
                "id": [0, 1, 2],
                "node_name": ["chlorophyle", "herbe", "plante"],
                "has_label": True,
            }
        ).cast({"id": pl.UInt32})
        assert_frame_equal(nodes.select(exp_nodes.columns), exp_nodes)
