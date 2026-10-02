import polars as pl
import unittest
from unittest.mock import patch

from biolit.flow_gatekeeper import filter_observations_for_crop


class TestFilterObservationsForCrop(unittest.TestCase):

    def setUp(self):
        """DataFrame de base réutilisé dans chaque test"""
        self.df = pl.DataFrame({"id_observation": [1111, 2345, 3223, 4456]})

    # ── Cas nominal-certaine observations ont été traitée  ──────────────────────────────────────────────────────────
    @patch("biolit.flow_gatekeeper.get_already_cropped_observations")
    def test_exclut_observations_deja_traitees(self, mock_get_already):
        """
        Les ids déjà présents en base doivent être exclus du résultat.
        """
        mock_get_already.return_value = pl.DataFrame({"id_observation": [2345, 4456]})

        result = filter_observations_for_crop(self.df, engine=None)

        self.assertEqual(result["id_observation"].to_list(), [1111, 3223])
        mock_get_already.assert_called_once_with(None)  # vérifie que le moteur est bien passé
    # ── Base vide - aucnne observation n'a été traitée ────────────────────────────────────────────────────────────
    @patch("biolit.flow_gatekeeper.get_already_cropped_observations")
    def test_base_vide_retourne_tout(self, mock_get_already):
        """
        Aucune observation en base → tout le DataFrame doit passer.
        """
        mock_get_already.return_value = pl.DataFrame({"id_observation": pl.Series([], dtype=pl.Int64)})

        result = filter_observations_for_crop(self.df, engine=None)

        self.assertEqual(result["id_observation"].to_list(), [1111, 2345, 3223, 4456])

    # ── Toutes les observations sont déjà traitées ─────────────────────────────────────────────────
    @patch("biolit.flow_gatekeeper.get_already_cropped_observations")
    def test_toutes_deja_traitees_retourne_vide(self, mock_get_already):
        """Si toutes les observations sont déjà traitées, le résultat doit être vide."""
        mock_get_already.return_value = pl.DataFrame({"id_observation": [1111, 2345, 3223, 4456]})

        result = filter_observations_for_crop(self.df, engine=None)

        self.assertTrue(result.is_empty())

    # ── FORCE_REPROCESS( changement de modèle Ml) ──────────────────────────────────────────────────────

    @patch("biolit.flow_gatekeeper.FORCE_REPROCESS", True)
    @patch("biolit.flow_gatekeeper.get_already_cropped_observations")
    def test_force_reprocess_bypasse_filtrage(self, mock_get_already):
        """
        Avec FORCE_REPROCESS=True, aucun filtrage et aucun appel DB.
        """
        result = filter_observations_for_crop(self.df, engine=None)

        mock_get_already.assert_not_called()
        self.assertEqual(result["id_observation"].to_list(), [1111, 2345, 3223, 4456])


if __name__ == "__main__":
    unittest.main()