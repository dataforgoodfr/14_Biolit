"""Lecture et normalisation de l'API BioLit."""

import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import requests
import structlog
from dotenv import load_dotenv

load_dotenv()
LOGGER = structlog.get_logger()

COLUMN_MAPPING = {
    "id": "id_observation",
    "date": "date_observation",
    "link": "lien_observation",
    "author": "observateur",
    "_url_sortie": "url_sortie",
    "espece-identifiee": "espece_identifiee",
    "heure-debut": "heure_debut",
    "heure-fin": "heure_fin",
    "latitude": "latitude",
    "longitude": "longitude",
    "photos": "photos",
    "relais": "relais",
    "espece_id": "id_espece",
    "espece": "nom_scientifique",
    "common": "nom_commun",
    "categorie-programme": "categorie_programme",
    "programme": "programme",
    "validee": "validee",
}


def normalize_column_name(column: str) -> str:
    """Convertit un nom de champ en ``snake_case`` ASCII."""

    normalized = unicodedata.normalize("NFKD", column)
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def normalize_observations(
    data: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise les clés sans perdre la réponse API brute."""

    return [
        {
            COLUMN_MAPPING.get(key, normalize_column_name(key)): value
            for key, value in observation.items()
        }
        | {"raw_payload": dict(observation)}
        for observation in data
    ]


def fetch_observations(
    url: str | None = None,
    *,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Récupère puis normalise les observations de l'API BioLit."""

    api_url = url or os.getenv("BIOLIT_API_URL")
    if not api_url:
        raise ValueError("BIOLIT_API_URL est absente de l'environnement.")

    response = (session or requests).get(api_url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise TypeError("L'API BioLit doit retourner une liste.")

    observations = normalize_observations(payload)
    LOGGER.info("api_observations_fetched", count=len(observations))
    return observations


# Alias court conservé pour les éventuels appels externes.
fetch_biolit_from_api = fetch_observations
