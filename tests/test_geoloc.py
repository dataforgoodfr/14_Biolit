import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from biolit.geoloc import get_info_nearest_commune, get_info_distance_to_coast

class TestDistanceToCommunes:
    def test_get_info_nearest_commune(self):
        inp = pd.DataFrame(
            {
                "id_-_n1": [1, 2, 3],
                "nom_du_lieu_-_n1": ["paris", "pornic", "ocean atlantique"],
                "longitude_-_n1": [2.33333, -2.108881, -9.864458],
                "latitude_-_n1": [48.866669, 47.111202, 47.934747],
            }
        )
        out = get_info_nearest_commune(inp)
        exp = pd.DataFrame(
            {
                "id_-_n1": [1, 2, 3],
                "nom_du_lieu_-_n1": ["paris", "pornic", "ocean atlantique"],
                "longitude_-_n1": [2.33333, -2.108881, -9.864458],
                "latitude_-_n1": [48.866669, 47.111202, 47.934747],
                "distance_commune_m": [0, 0, np.nan],
                "nearest_commune": ["Paris", "Pornic", None],
                "code_insee": ["75056", "44131", None],
                "code_postal": [75000, 44210, np.nan],
                "reg_nom": ["Île-de-France", "Pays de la Loire", None],
                "dep_nom": ["Paris", "Loire-Atlantique", None]
            }
        )
        assert_frame_equal(out, exp, check_dtype=False)

    def test_get_info_distance_to_coast(self):
        inp = pd.DataFrame(
            {
                "id_-_n1": [1, 2, 3],
                "nom_du_lieu_-_n1": ["paris", "pornic", "ocean atlantique"],
                "longitude_-_n1": [2.33333, -2.108881, -9.864458],
                "latitude_-_n1": [48.866669, 47.111202, 47.934747],
            }
        )
        out = get_info_distance_to_coast(inp)
        exp = pd.DataFrame(
            {
                "id_-_n1": [1, 2, 3],
                "nom_du_lieu_-_n1": ["paris", "pornic", "ocean atlantique"],
                "longitude_-_n1": [2.33333, -2.108881, -9.864458],
                "latitude_-_n1": [48.866669, 47.111202, 47.934747],
                "distance_to_coast": [np.nan, 0.314727, np.nan],
                "is_coastal": [False, True, False],
            }
        )
        assert_frame_equal(out, exp, check_dtype=False)
