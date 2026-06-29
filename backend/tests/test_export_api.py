from unittest.mock import Mock

import pytest

from biolit.export_api import fetch_observations, normalize_column_name, normalize_observations

PAYLOAD = [
    {
        "id": "42",
        "date": "2026-06-01",
        "photos": "https://images.example/42.jpg",
        "espece": "Asterias rubens",
        "Champ Étendu": "conservé",
    }
]


def test_normalize_column_name():
    assert normalize_column_name(" Espèce - observée ? ") == "espece_observee"


def test_normalize_observations_keeps_raw_payload():
    result = normalize_observations(PAYLOAD)

    assert result[0]["id_observation"] == "42"
    assert result[0]["nom_scientifique"] == "Asterias rubens"
    assert result[0]["champ_etendu"] == "conservé"
    assert result[0]["raw_payload"] == PAYLOAD[0]


def test_fetch_observations_uses_injected_session():
    response = Mock()
    response.json.return_value = PAYLOAD
    session = Mock()
    session.get.return_value = response

    result = fetch_observations("https://api.example", timeout=5, session=session)

    assert result[0]["id_observation"] == "42"
    session.get.assert_called_once_with("https://api.example", timeout=5)
    response.raise_for_status.assert_called_once_with()


def test_fetch_observations_rejects_non_list_payload():
    response = Mock()
    response.json.return_value = {"id": 42}
    session = Mock()
    session.get.return_value = response

    with pytest.raises(TypeError):
        fetch_observations("https://api.example", session=session)
