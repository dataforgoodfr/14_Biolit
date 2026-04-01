from dotenv import load_dotenv
import os

import structlog
from datetime import datetime

from label_studio_sdk import LabelStudio
from label_studio_sdk.data_manager import Filters, Column, Operator, Type

LOGGER = structlog.get_logger()

class LabelStudioAccessor:
    def __init__(self) -> None:
        load_dotenv('.env')

        if any([
            var not in os.environ
            for var in ['LABEL_STUDIO_URL', 'LABEL_STUDIO_API_KEY']
        ]):
            LOGGER.fatal('Missing some connexion information for LabelStudio')

        self._studio_url = os.environ.get('LABEL_STUDIO_URL')
        self._studio_api_key = os.environ.get('LABEL_STUDIO_API_KEY')

        self._client = LabelStudio(
            base_url=self._studio_url,
            api_key=self._studio_api_key
        )


    def get_project_id_from_title(
        self, project_title = ""
    ) -> int:
        candidates_id = [
            p.id for p in self._client.projects.list()
            if p.title == project_title
        ]

        if len(candidates_id) == 0:
            LOGGER.fatal(
                "Did not find a matching project for this title on this studio",
                studio = self._studio_url,
                project_title = project_title
            )
        if len(candidates_id) > 1:
            LOGGER.error(
                "Found several projects with same title on this studio.",
                studio = self._studio_url,
                project_title = project_title
            )

        return candidates_id[0]


    def create_project_export_for_date_range(
        self,
        project_id: int,
        date_min: datetime, date_max: datetime
    ) -> None:
        filters = Filters.create(Filters.OR, [
            Filters.item(
                Column.completed_at, Operator.IN, Type.Datetime,
                Filters.value(date_min, date_max)
            )
        ])

        view = self._client.views.create(
            data=filters, project=project_id
        )
        export_job = self._client.projects.exports.create(
            id = project_id,
            task_filter_options = {"view": view.id}
        )
        export = self._client.projects.exports.get(
            id=project_id, export_pk=export_job.id
        )
        return export.id


    def store_project_export_as_json(
        self, path: str,
        project_id: int, export_id: int
    ) -> None:
        with open(path, "wb") as f:
            for chunk in self._client.projects.exports.download(
                id=project_id,
                export_pk=export_id,
                export_type="JSON",
                request_options={"chunk_size": 1024},
            ):
                f.write(chunk)