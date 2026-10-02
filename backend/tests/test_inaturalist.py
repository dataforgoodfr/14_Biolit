import json
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from biolit.inaturalist import parse_inaturalist_api


class TestParseInaturalist:
    def test_parse_inaturalist(self):
        with open(
            Path(__file__).parent / "fixtures" / "inaturalist_observation_api.json"
        ) as f:
            content = json.load(f)
        out = parse_inaturalist_api(content)
        exp = pl.DataFrame(
            {
                "inat_obs_id": [333150395, 333062785],
                "created_at": [
                    "2026-01-02T14:06:26.311",
                    "2026-01-01T22:20:45.612",
                ],
                "inat_taxon_id": [120138, 120138],
                "taxon_name": "asterias rubens",
                "inat_taxon_rank": "species",
                "inat_image_url": [
                    "https://inaturalist-open-data.s3.amazonaws.com/photos/604784329/large.jpg",
                    "https://inaturalist-open-data.s3.amazonaws.com/photos/604591769/large.jpg",
                ],
                "inat_image_id": [604784329, 604591769],
            }
        ).cast({"created_at": pl.Datetime})
        assert_frame_equal(out, exp)
