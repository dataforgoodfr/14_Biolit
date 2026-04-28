import os
import polars as pl
from dotenv import load_dotenv
import structlog
from label_studio_sdk import LabelStudio

LOGGER = structlog.get_logger()
load_dotenv()

def push_tasks_label_studio_crops(project_title: str, df: pl.DataFrame):
    api_key = os.getenv("LABEL_STUDIO_API_KEY_DATAFORGOOD")
    url = os.getenv("LABEL_STUDIO_URL")

    client = LabelStudio(base_url=url, api_key=api_key)
    projects = client.projects.list()

    project_id = None
    for project in projects:
        if project.title == project_title:
            project_id = project.id
            LOGGER.info(f"Projet ID={project.id}, Nom={project.title} exists")
            break

    if project_id is None:
        LOGGER.info(f"The project {project_title} does not exist.")
        return

    tasks = []
    for row in df.to_dicts():
        tasks.append({
            "data": {
                "image": row["path_s3"],
                "id_crops": row["id_crops"],
                "id_observation": row.get("id_observation", ""),
                "regne_yolo": row.get("regne_yolo", ""),
                "confiance_yolo": row.get("confiance_yolo", ""),
                "best_label": row.get("best_label", ""),
                "best_level": row.get("best_level", ""),
                "best_score": row.get("best_score", ""),
                "famille": row.get("famille", ""),
                "species_name": row.get("species_name", ""),
            }
        })

    client.projects.import_tasks(
        id=project_id,
        request=tasks,
        return_task_ids=True,
    )
    LOGGER.info("Crops tasks imported to Label Studio", value=len(df))

def push_tasks_label_studio_no_crops(project_title: str, df: pl.DataFrame):
    api_key = os.getenv("LABEL_STUDIO_API_KEY_DATAFORGOOD")
    url = os.getenv("LABEL_STUDIO_URL")

    client = LabelStudio(base_url=url, api_key=api_key)
    projects = client.projects.list()

    project_id = None
    for project in projects:
        if project.title == project_title:
            project_id = project.id
            LOGGER.info(f"Projet ID={project.id}, Nom={project.title} exists")
            break

    if project_id is None:
        LOGGER.info(f"The project {project_title} does not exist.")
        return

    tasks = []

    for row in df.to_dicts():

        tasks.append({
            "data": {
                "image": row["path_s3"],
                "id_observation": row["id_observation"],
                "site": row["relais"] or "",
                "region": row["reg_nom"] or "",
                "commune": row["nearest_commune"] or "",
                "geo_map_html":f'<a href="https://www.openstreetmap.org/?mlat={row["latitude"]}&mlon={row["longitude"]}" target="_blank">Voir la carte</a>' or "",
                "latitude": row["latitude"] or "",
                "longitude": row["longitude"] or "",
                "departement": row["dep_nom"] or "",
            }
        })

    # Import tasks
    client.projects.import_tasks(
        id=project_id,
        request=tasks,
        return_task_ids=True,
    )
    LOGGER.info("The tasks have been successfully imported ; number of tasks :", value=len(df))


def delete_tasks_label_studio(project_title: str):
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    url = "http://label-studio:8080"

    client = LabelStudio(base_url=url, api_key=api_key)
    projects = client.projects.list()
    project_id = None

    for project in projects:
        if project.title == project_title:
            project_id = project.id
            LOGGER.info(f"Projet ID={project.id}, Nom={project.title} exists")
            break

    if project_id is None:
        LOGGER.info(f"The project {project_title} does not exist.")
        return

    tasks = client.tasks.list(project=project.id)
    if not tasks:
        LOGGER.info("No tasks to delete.")
        return

    task_ids = [t.id for t in tasks]
    for task_id in task_ids:
        client.tasks.delete(task_id)
    LOGGER.info(f"{len(task_ids)} tasks deleted from project {project.id}")

def extract_no_crop_data_from_label_studio(project_title: str) -> pl.DataFrame:
    """
    Extraction brute des tâches Label Studio (aucun filtrage métier).
    """

    api_key = os.getenv("LABEL_STUDIO_API_KEY_DATAFORGOOD")
    url = os.getenv("LABEL_STUDIO_URL")

    client = LabelStudio(base_url=url, api_key=api_key)

    # -------------------------
    # Récupération projet
    # -------------------------
    projets = client.projects.list()

    project_id = next(
        (p.id for p in projets if p.title == project_title),
        None
    )

    if project_id is None:
        LOGGER.warning(f"Projet {project_title} introuvable")
        return pl.DataFrame()

    # -------------------------
    # Récupération des tasks
    # -------------------------
    tasks = client.tasks.list(project=project_id)

    rows = []

    for task in tasks:

        annotation = task.annotations[0] if task.annotations else None

        rows.append({
            # -------------------------
            # Identifiants
            # -------------------------
            "task_id": task.id,
            "id_observation": task.data.get("id_observation"),
            "image": task.data.get("image"),

            # -------------------------
            # Etats Label Studio
            # -------------------------
            "completed": getattr(task, "is_labeled", None),
            "cancelled": bool(getattr(task, "cancelled", False)),
            "has_annotations": bool(task.annotations),

            # -------------------------
            # Annotation (si existe)
            # -------------------------
            "annotated_by": getattr(annotation, "completed_by", None) if annotation else None,
            "annotated_at": getattr(annotation, "created_at", None) if annotation else None,

            # -------------------------
            # Predictions (ML)
            # -------------------------
            "predictions": getattr(task, "predictions", None),
        })

    df = pl.DataFrame(rows)

    LOGGER.info(
        "Extraction brute Label Studio terminée",
        nb_tasks=len(df)
    )

    return df