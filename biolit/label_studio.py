import os
import polars as pl
from dotenv import load_dotenv
import json
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

def extract_crop_data_from_label_studio(project_title: str) -> pl.DataFrame:
    """
    Extraction brute des tâches Label Studio (aucun filtrage métier) Projet Crops.
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
        return pl.DataFrame()

    # -------------------------
    # Récupération des tasks
    # -------------------------
    tasks = client.tasks.list(project=project_id)

    rows = []

    for task in tasks:
        annotation = task.annotations[0] if task.annotations else None

        annotator = None
        annotated_at = None
        decision = None
        correction_mode = None
        espece_corrigee = None
        commentaire = None

        if annotation and isinstance(annotation, dict):

            # -------------------------
            # Labels
            # -------------------------
            results = annotation.get("result", [])
            for r in results:
                from_name = r.get("from_name")
                value = r.get("value", {})

                if r.get("type") == "choices":
                    choice = value.get("choices", [None])[0]

                    if from_name == "decision":
                        decision = choice

                    elif from_name == "correction_mode":
                        correction_mode = choice

                    elif from_name in ["alternative_rapide", "species"]:
                        if espece_corrigee is None: 
                            espece_corrigee = choice

                elif r.get("type") == "textarea":
                    text = value.get("text", [None])[0]

                    if from_name == "commentaire":
                        commentaire = text
            # -------------------------
            # Annotateur
            # -------------------------
            created_username = annotation.get("created_username")

            if created_username:
                annotator = created_username.split(",")[0].strip()
            else:
                annotator = annotation.get("completed_by")

            # -------------------------
            # Metadata annotation
            # -------------------------
            annotated_at = annotation.get("created_at")
            cancelled = annotation.get("was_cancelled", False)
            
    

        rows.append({
            # -------------------------
            # Identifiants
            # -------------------------
            "task_id": task.id,
            "id_observation": task.data.get("id_observation"),
            "image": task.data.get("image"),
            "espece_pred": task.data.get('espece_pred'),

            # -------------------------
            # Etats tâche
            # -------------------------
            "task_created_date": getattr(task, "created_at", None),
            # -------------------------
            # Annotation
            # -------------------------
            "annotator": annotator,
            "annotated_at": annotated_at,
            "decision": decision,
            "commentaire":commentaire,
            "correction_mode":correction_mode,
            "espece_corigée" :espece_corrigee,
        })
    df=pl.DataFrame(rows)

    return df


import os
import polars as pl
from label_studio_sdk import LabelStudio

def extract_no_crops_data_from_label_studio(project_title: str) -> pl.DataFrame:
    """
    Extraction brute des tâches Label Studio (aucun filtrage métier) Projet No Crops.
    """
    api_key = os.getenv("LABEL_STUDIO_API_KEY_DATAFORGOOD")
    url = os.getenv("LABEL_STUDIO_URL")

    client = LabelStudio(base_url=url, api_key=api_key)

    # -------------------------
    # Récupération projet
    # -------------------------
    projects = client.projects.list()
    project_id = next((p.id for p in projects if p.title == project_title), None)

    if project_id is None:
        return pl.DataFrame()

    # -------------------------
    # Récupération des tasks
    # -------------------------
    tasks = client.tasks.list(project=project_id)

    rows = []

    for task in tasks:
        annotation = task.annotations[0] if task.annotations else None

        if not annotation:
            continue

        results = annotation.get("result", [])

        annotator = annotation.get("created_username")
        if annotator:
            annotator = annotator.split(",")[0].strip()
        else:
            annotator = annotation.get("completed_by")

        annotated_at = annotation.get("created_at")

        commentaire = None
        crops = []

        # -------------------------
        # Parsing résultats
        # -------------------------
        for r in results:
            from_name = r.get("from_name")
            value = r.get("value", {})

            # commentaire
            if from_name == "commentaire" and r.get("type") == "textarea":
                commentaire = value.get("text", [None])[0]

            # crops (rectangle)
            if r.get("type") == "rectanglelabels":
                label = value.get("rectanglelabels", [None])[0]

                crops.append({
                    "x": value.get("x"),
                    "y": value.get("y"),
                    "width": value.get("width"),
                    "height": value.get("height"),
                    "label": label,
                    "original_width": r.get("original_width"),
                    "original_height": r.get("original_height"),
                })

        # -------------------------
        # 1 ligne = 1 crop
        # -------------------------
        for idx, crop in enumerate(crops):
            rows.append({
                "task_id": task.id,
                "id_observation": task.data.get("id_observation"),
                "image": task.data.get("image"),
                 # -------------------------
                # Etats tâche
                # -------------------------
                "task_created_date": getattr(task, "created_at", None),
                # -------------------------

                # crop
                "crop_index": idx,
                "x": crop["x"],
                "y": crop["y"],
                "width": crop["width"],
                "height": crop["height"],
                "label": crop["label"],
                "original_width": crop["original_width"],
                "original_height": crop["original_height"],

                # annotation
                "annotator": annotator,
                "annotated_at": annotated_at,
                "commentaire": commentaire,
            })

    df = pl.DataFrame(rows)
    return df
    
    
    