"""Passerelle REST légère vers les deux projets Label Studio."""

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from biolit.settings import CROP_PROJECT, LABEL_CONFIG_DIR, VALIDATION_PROJECT

load_dotenv()


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    original_width: int
    original_height: int


@dataclass(frozen=True)
class ManualCropAnnotation:
    task_id: int
    image_id: str
    bounding_box: BoundingBox | None
    no_species: bool
    annotator: str | None
    annotated_at: datetime


@dataclass(frozen=True)
class ValidationAnnotation:
    task_id: int
    image_id: str
    decision: str
    corrected_name: str | None
    annotator: str | None
    annotated_at: datetime


def _datetime(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = datetime.now(UTC)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _annotator(annotation: dict[str, Any]) -> str | None:
    username = annotation.get("created_username")
    if username:
        return str(username).split(",")[0].strip()
    completed_by = annotation.get("completed_by")
    return str(completed_by) if completed_by is not None else None


class LabelStudioGateway:
    """Crée les projets, publie les tâches et lit les annotations par HTTP."""

    def __init__(self, session: requests.Session, base_url: str):
        self.session = session
        self.base_url = base_url.rstrip("/")
        self._project_ids: dict[str, int] = {}

    @classmethod
    def from_environment(cls) -> "LabelStudioGateway":
        url = os.getenv("LABEL_STUDIO_URL")
        api_key = os.getenv("LABEL_STUDIO_API_KEY")
        if not url or not api_key:
            raise ValueError("LABEL_STUDIO_URL ou LABEL_STUDIO_API_KEY est absente.")
        session = requests.Session()
        session.headers.update({"Authorization": f"Token {api_key}"})
        return cls(session, url)

    def ensure_projects(self) -> None:
        self._ensure_project(CROP_PROJECT, LABEL_CONFIG_DIR / "crop.xml")
        self._ensure_project(VALIDATION_PROJECT, LABEL_CONFIG_DIR / "validation.xml")

    def push_manual_crop(self, image_id: str, image_url: str) -> None:
        self._import_tasks(
            CROP_PROJECT,
            [{"data": {"image": image_url, "id_image": image_id}}],
        )

    def push_validation(
        self,
        *,
        image_id: str,
        image_url: str,
        predicted_name: str | None,
        predicted_rank: str | None,
        score: float | None,
    ) -> None:
        self._import_tasks(
            VALIDATION_PROJECT,
            [
                {
                    "data": {
                        "image": image_url,
                        "id_image": image_id,
                        "predicted_name": predicted_name or "Aucune prédiction",
                        "predicted_rank": predicted_rank or "inconnu",
                        "score": round(score or 0.0, 4),
                    }
                }
            ],
        )

    def manual_crop_annotations(self) -> list[ManualCropAnnotation]:
        annotations = []
        for task in self._tasks(CROP_PROJECT):
            annotation = task.get("annotations", [None])[0] if task.get("annotations") else None
            if not isinstance(annotation, dict):
                continue

            bounding_box = None
            no_species = False
            for result in annotation.get("result", []):
                value = result.get("value", {})
                if result.get("from_name") == "presence":
                    no_species = "Pas d'espèce" in value.get("choices", [])
                elif result.get("from_name") == "crop" and result.get("type") == "rectanglelabels":
                    bounding_box = BoundingBox(
                        x=float(value["x"]),
                        y=float(value["y"]),
                        width=float(value["width"]),
                        height=float(value["height"]),
                        original_width=int(result["original_width"]),
                        original_height=int(result["original_height"]),
                    )
            if bounding_box is None and not no_species:
                continue
            annotations.append(
                ManualCropAnnotation(
                    task_id=int(task["id"]),
                    image_id=str(task["data"]["id_image"]),
                    bounding_box=bounding_box,
                    no_species=no_species,
                    annotator=_annotator(annotation),
                    annotated_at=_datetime(annotation.get("created_at")),
                )
            )
        return annotations

    def validation_annotations(self) -> list[ValidationAnnotation]:
        annotations = []
        for task in self._tasks(VALIDATION_PROJECT):
            annotation = task.get("annotations", [None])[0] if task.get("annotations") else None
            if not isinstance(annotation, dict):
                continue

            decision = None
            corrected_name = None
            for result in annotation.get("result", []):
                value = result.get("value", {})
                if result.get("from_name") == "decision":
                    decision = next(iter(value.get("choices", [])), None)
                elif result.get("from_name") == "corrected_species":
                    corrected_name = next(iter(value.get("text", [])), None)
            if decision is None:
                continue
            annotations.append(
                ValidationAnnotation(
                    task_id=int(task["id"]),
                    image_id=str(task["data"]["id_image"]),
                    decision=decision,
                    corrected_name=corrected_name.strip() if corrected_name else None,
                    annotator=_annotator(annotation),
                    annotated_at=_datetime(annotation.get("created_at")),
                )
            )
        return annotations

    def delete_task(self, task_id: int) -> None:
        self._request("DELETE", f"/api/tasks/{task_id}")

    def _ensure_project(self, title: str, config_path: Path) -> int:
        if title in self._project_ids:
            return self._project_ids[title]
        for project in self._list("/api/projects"):
            if project.get("title") == title:
                self._project_ids[title] = int(project["id"])
                return self._project_ids[title]
        project = self._request(
            "POST",
            "/api/projects",
            json={"title": title, "label_config": config_path.read_text()},
        ).json()
        self._project_ids[title] = int(project["id"])
        return self._project_ids[title]

    def _tasks(self, title: str) -> list[dict[str, Any]]:
        return self._list(
            "/api/tasks",
            params={"project": self._project_id(title), "page_size": 100},
        )

    def _import_tasks(self, title: str, tasks: list[dict[str, Any]]) -> None:
        self._request(
            "POST",
            f"/api/projects/{self._project_id(title)}/import",
            params={"return_task_ids": "true"},
            json=tasks,
        )

    def _project_id(self, title: str) -> int:
        if title not in self._project_ids:
            self.ensure_projects()
        return self._project_ids[title]

    def _list(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        items = []
        url = f"{self.base_url}{path}"
        while url:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                items.extend(payload)
                break
            items.extend(payload.get("results") or payload.get("tasks") or [])
            next_url = payload.get("next")
            if next_url and next_url.startswith("http"):
                url = next_url
            elif next_url:
                url = f"{self.base_url}/{next_url.lstrip('/')}"
            else:
                url = None
            params = None
        return items

    def _request(self, method: str, path: str, **kwargs):
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            timeout=30,
            **kwargs,
        )
        response.raise_for_status()
        return response
