"""Schéma PostgreSQL et file d'états du backend BioLit."""

import json
import os
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()
LOGGER = structlog.get_logger()

STATUSES = {
    "pending_crop",
    "cropping",
    "awaiting_manual_crop",
    "pending_classification",
    "classifying",
    "awaiting_validation",
    "validated",
    "rejected",
    "error",
}

OBSERVATIONS_DDL = """
CREATE TABLE IF NOT EXISTS observations (
    id_observation BIGINT PRIMARY KEY,
    date_observation TIMESTAMP,
    lien_observation TEXT,
    observateur TEXT,
    url_sortie TEXT,
    espece_identifiee TEXT,
    heure_debut TIME,
    heure_fin TIME,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    photos TEXT,
    relais BIGINT,
    id_espece BIGINT,
    nom_scientifique TEXT,
    nom_commun TEXT,
    categorie_programme BIGINT,
    programme TEXT,
    validee TEXT,
    raw_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

PROCESSING_DDL = """
CREATE TABLE IF NOT EXISTS processing_images (
    id_image TEXT PRIMARY KEY,
    id_observation BIGINT NOT NULL REFERENCES observations(id_observation),
    image_position INTEGER NOT NULL DEFAULT 0,
    source_url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_crop',
    original_s3_uri TEXT,
    crop_s3_uri TEXT,
    crop_source TEXT,
    crop_label TEXT,
    crop_confidence DOUBLE PRECISION,
    predicted_name TEXT,
    predicted_rank TEXT,
    prediction_score DOUBLE PRECISION,
    prediction_margin DOUBLE PRECISION,
    prediction_details TEXT,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT processing_images_status_check CHECK (
        status IN (
            'pending_crop', 'cropping', 'awaiting_manual_crop',
            'pending_classification', 'classifying', 'awaiting_validation',
            'validated', 'rejected', 'error'
        )
    ),
    UNIQUE (id_observation, image_position)
)
"""

VALIDATED_DDL = """
CREATE TABLE IF NOT EXISTS validated_species (
    id_image TEXT PRIMARY KEY REFERENCES processing_images(id_image),
    id_observation BIGINT NOT NULL REFERENCES observations(id_observation),
    scientific_name TEXT,
    taxon_rank TEXT,
    identifiable BOOLEAN NOT NULL,
    annotator TEXT,
    validation_source TEXT NOT NULL,
    validated_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

METABASE_VIEW_DDL = """
CREATE OR REPLACE VIEW metabase_observations AS
SELECT
    o.*,
    p.id_image,
    p.image_position,
    p.status AS processing_status,
    p.crop_source,
    p.crop_label,
    p.crop_confidence,
    p.predicted_name,
    p.predicted_rank,
    p.prediction_score,
    v.scientific_name AS validated_scientific_name,
    v.taxon_rank AS validated_taxon_rank,
    v.identifiable,
    v.annotator,
    v.validation_source,
    v.validated_at
FROM observations o
JOIN processing_images p USING (id_observation)
LEFT JOIN validated_species v USING (id_image)
"""

OBSERVATION_COLUMNS = (
    "id_observation",
    "date_observation",
    "lien_observation",
    "observateur",
    "url_sortie",
    "espece_identifiee",
    "heure_debut",
    "heure_fin",
    "latitude",
    "longitude",
    "photos",
    "relais",
    "id_espece",
    "nom_scientifique",
    "nom_commun",
    "categorie_programme",
    "programme",
    "validee",
    "raw_payload",
)

UPDATABLE_PROCESSING_COLUMNS = {
    "status",
    "original_s3_uri",
    "crop_s3_uri",
    "crop_source",
    "crop_label",
    "crop_confidence",
    "predicted_name",
    "predicted_rank",
    "prediction_score",
    "prediction_margin",
    "prediction_details",
    "last_error",
}


@dataclass(frozen=True)
class IngestSummary:
    observations: int
    images_discovered: int


def _integer(value: Any) -> int | None:
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _time(value: Any) -> time | None:
    if isinstance(value, time):
        return value
    if not value:
        return None
    try:
        return time.fromisoformat(str(value))
    except ValueError:
        return None


def extract_photo_urls(value: Any) -> list[str]:
    """Extrait les URL d'un champ ``photos`` sous forme texte, liste ou JSON."""

    if value is None:
        return []
    if isinstance(value, dict):
        candidate = value.get("url") or value.get("src") or value.get("large_url")
        return [candidate] if isinstance(candidate, str) and candidate.startswith("http") else []
    if isinstance(value, (list, tuple)):
        urls = []
        for item in value:
            urls.extend(extract_photo_urls(item))
        return urls
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return extract_photo_urls(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return [stripped] if stripped.startswith("http") else []
    return []


def image_rows_from_observations(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Construit les identifiants stables des images découvertes dans l'API."""

    rows = []
    for observation in observations:
        if str(observation.get("validee", "false")).strip().lower() in {"true", "1", "oui"}:
            continue
        observation_id = _integer(observation.get("id_observation"))
        if observation_id is None:
            continue
        urls = extract_photo_urls(observation.get("photos"))
        for position, url in enumerate(urls):
            image_id = str(observation_id) if len(urls) == 1 else f"{observation_id}:{position}"
            rows.append(
                {
                    "id_image": image_id,
                    "id_observation": observation_id,
                    "image_position": position,
                    "source_url": url,
                }
            )
    return rows


class Repository:
    """Façade unique sur PostgreSQL pour le worker de production."""

    def __init__(self, engine: Engine):
        self.engine = engine

    @classmethod
    def from_environment(cls) -> "Repository":
        url = os.getenv("POSTGRES_URL")
        if not url:
            raise ValueError("POSTGRES_URL est absente de l'environnement.")
        return cls(create_engine(url, pool_pre_ping=True))

    def create_schema(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text(OBSERVATIONS_DDL))
            connection.execute(
                text("ALTER TABLE observations ADD COLUMN IF NOT EXISTS raw_payload JSONB")
            )
            connection.execute(
                text(
                    "ALTER TABLE observations ADD COLUMN IF NOT EXISTS "
                    "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                )
            )
            connection.execute(
                text(
                    "ALTER TABLE observations ADD COLUMN IF NOT EXISTS "
                    "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                )
            )
            connection.execute(text(PROCESSING_DDL))
            connection.execute(text(VALIDATED_DDL))
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS processing_images_status_idx "
                    "ON processing_images(status)"
                )
            )
            connection.execute(text(METABASE_VIEW_DDL))

    def ingest(self, observations: list[dict[str, Any]]) -> IngestSummary:
        prepared = [self._prepare_observation(row) for row in observations]
        prepared = [row for row in prepared if row["id_observation"] is not None]
        images = image_rows_from_observations(observations)

        observation_query = text(
            """
            INSERT INTO observations (
                id_observation, date_observation, lien_observation, observateur,
                url_sortie, espece_identifiee, heure_debut, heure_fin, latitude,
                longitude, photos, relais, id_espece, nom_scientifique, nom_commun,
                categorie_programme, programme, validee, raw_payload
            ) VALUES (
                :id_observation, :date_observation, :lien_observation, :observateur,
                :url_sortie, :espece_identifiee, :heure_debut, :heure_fin, :latitude,
                :longitude, :photos, :relais, :id_espece, :nom_scientifique,
                :nom_commun, :categorie_programme, :programme, :validee,
                CAST(:raw_payload AS JSONB)
            )
            ON CONFLICT (id_observation) DO UPDATE SET
                date_observation = EXCLUDED.date_observation,
                lien_observation = EXCLUDED.lien_observation,
                observateur = EXCLUDED.observateur,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                photos = EXCLUDED.photos,
                raw_payload = EXCLUDED.raw_payload,
                updated_at = NOW()
            """
        )
        image_query = text(
            """
            INSERT INTO processing_images (
                id_image, id_observation, image_position, source_url
            ) VALUES (
                :id_image, :id_observation, :image_position, :source_url
            )
            ON CONFLICT (id_image) DO UPDATE SET
                source_url = EXCLUDED.source_url,
                updated_at = NOW()
            """
        )
        with self.engine.begin() as connection:
            if prepared:
                connection.execute(observation_query, prepared)
            if images:
                connection.execute(image_query, images)

        return IngestSummary(len(prepared), len(images))

    def claim(self, current_status: str, claimed_status: str, limit: int) -> list[dict]:
        self._validate_status(current_status)
        self._validate_status(claimed_status)
        query = text(
            """
            WITH selected AS (
                SELECT id_image
                FROM processing_images
                WHERE status = :current_status
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT :limit
            )
            UPDATE processing_images AS image
            SET status = :claimed_status, updated_at = NOW(), last_error = NULL
            FROM selected
            WHERE image.id_image = selected.id_image
            RETURNING image.*
            """
        )
        with self.engine.begin() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    query,
                    {
                        "current_status": current_status,
                        "claimed_status": claimed_status,
                        "limit": limit,
                    },
                ).mappings()
            ]

    def get_image(self, image_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    text("SELECT * FROM processing_images WHERE id_image = :id_image"),
                    {"id_image": image_id},
                )
                .mappings()
                .one_or_none()
            )
        return dict(row) if row else None

    def transition(self, image_id: str, **fields: Any) -> None:
        unknown = set(fields) - UPDATABLE_PROCESSING_COLUMNS
        if unknown:
            raise ValueError(f"Colonnes de traitement inconnues : {sorted(unknown)}")
        if "status" in fields:
            self._validate_status(fields["status"])
        if not fields:
            return

        assignments = ", ".join(f"{column} = :{column}" for column in fields)
        query = text(
            f"UPDATE processing_images SET {assignments}, updated_at = NOW() "
            "WHERE id_image = :id_image"
        )
        with self.engine.begin() as connection:
            connection.execute(query, fields | {"id_image": image_id})

    def finalize(
        self,
        image_id: str,
        *,
        scientific_name: str | None,
        taxon_rank: str | None,
        identifiable: bool,
        annotator: str | None,
        source: str,
        validated_at: datetime,
    ) -> None:
        image = self.get_image(image_id)
        if image is None:
            raise KeyError(f"Image inconnue : {image_id}")

        query = text(
            """
            INSERT INTO validated_species (
                id_image, id_observation, scientific_name, taxon_rank,
                identifiable, annotator, validation_source, validated_at
            ) VALUES (
                :id_image, :id_observation, :scientific_name, :taxon_rank,
                :identifiable, :annotator, :validation_source, :validated_at
            )
            ON CONFLICT (id_image) DO UPDATE SET
                scientific_name = EXCLUDED.scientific_name,
                taxon_rank = EXCLUDED.taxon_rank,
                identifiable = EXCLUDED.identifiable,
                annotator = EXCLUDED.annotator,
                validation_source = EXCLUDED.validation_source,
                validated_at = EXCLUDED.validated_at
            """
        )
        with self.engine.begin() as connection:
            connection.execute(
                query,
                {
                    "id_image": image_id,
                    "id_observation": image["id_observation"],
                    "scientific_name": scientific_name,
                    "taxon_rank": taxon_rank,
                    "identifiable": identifiable,
                    "annotator": annotator,
                    "validation_source": source,
                    "validated_at": validated_at,
                },
            )
            connection.execute(
                text(
                    "UPDATE processing_images SET status = :status, updated_at = NOW() "
                    "WHERE id_image = :id_image"
                ),
                {"status": "validated" if identifiable else "rejected", "id_image": image_id},
            )

    def finalized_assets(self, limit: int = 100) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT id_image, original_s3_uri, crop_s3_uri
            FROM processing_images
            WHERE status IN ('validated', 'rejected')
              AND (original_s3_uri IS NOT NULL OR crop_s3_uri IS NOT NULL)
            LIMIT :limit
            """
        )
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(query, {"limit": limit}).mappings()]

    def clear_assets(self, image_id: str) -> None:
        self.transition(image_id, original_s3_uri=None, crop_s3_uri=None)

    def recover_stale_claims(self, stale_minutes: int) -> int:
        """Replace les traitements interrompus dans leur file d'origine."""

        query = text(
            """
            UPDATE processing_images
            SET status = CASE
                    WHEN status = 'cropping' THEN 'pending_crop'
                    WHEN status = 'classifying' THEN 'pending_classification'
                END,
                updated_at = NOW(),
                last_error = 'Traitement interrompu puis remis en file'
            WHERE status IN ('cropping', 'classifying')
              AND updated_at < NOW() - (:minutes * INTERVAL '1 minute')
            """
        )
        with self.engine.begin() as connection:
            return connection.execute(query, {"minutes": stale_minutes}).rowcount

    def fail(self, image_id: str, error: Exception | str) -> None:
        self.transition(image_id, status="error", last_error=str(error)[:2_000])
        LOGGER.error("image_processing_failed", image_id=image_id, error=str(error))

    @staticmethod
    def _prepare_observation(row: dict[str, Any]) -> dict[str, Any]:
        prepared = {column: row.get(column) for column in OBSERVATION_COLUMNS}
        prepared.update(
            {
                "id_observation": _integer(row.get("id_observation")),
                "date_observation": _datetime(row.get("date_observation")),
                "heure_debut": _time(row.get("heure_debut")),
                "heure_fin": _time(row.get("heure_fin")),
                "latitude": _float(row.get("latitude")),
                "longitude": _float(row.get("longitude")),
                "relais": _integer(row.get("relais")),
                "id_espece": _integer(row.get("id_espece")),
                "categorie_programme": _integer(row.get("categorie_programme")),
                "photos": json.dumps(row.get("photos"), ensure_ascii=False),
                "raw_payload": json.dumps(row.get("raw_payload", row), ensure_ascii=False),
            }
        )
        return prepared

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"Statut inconnu : {status}")
