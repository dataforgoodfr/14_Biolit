import os
import polars as pl
from datetime import datetime
from dotenv import load_dotenv
import structlog
from label_studio_sdk import LabelStudio

LOGGER = structlog.get_logger()
load_dotenv()

def get_label_studio_client() -> LabelStudio:
    api_key = os.getenv("LABEL_STUDIO_API_KEY_DATAFORGOOD")
    url = os.getenv("LABEL_STUDIO_URL")
    return LabelStudio(base_url=url, api_key=api_key)

def recuperation_project_id(project_title: str) -> int:
    client = get_label_studio_client()
    projects = client.projects.list()
    project_id = None
    for project in projects:
        if project.title == project_title:
            project_id = project.id
            break
    if project_id is None:
        LOGGER.info(f"The project {project_title} does not exist.")
        return

    return project_id

def push_tasks_label_studio_crops(project_title: str, df: pl.DataFrame):
    client = get_label_studio_client()
    project_id = recuperation_project_id(project_title)
    if project_id is not None:
        tasks = []
        for row in df.to_dicts():
            tasks.append({
                "data": {
                    "image": row["path_s3"],
                    "id_crops": row["id_crops"],
                    "id_observation": row.get("id_observation") or "",
                    "regne_yolo": row.get("regne_yolo") or "",
                    "confiance_yolo": row.get("confiance_yolo") or "",
                    "best_label": row.get("best_label") or "",
                    "best_level": row.get("best_level") or "",
                    "best_score": row.get("best_score") or "",
                    "regne": row.get("regne") or "",
                    "phylum": row.get("phylum") or "",
                    "classe": row.get("classe") or "",
                    "ordre": row.get("ordre") or "",
                    "famille": row.get("famille") or "",
                    "species_name": row.get("species_name") or "Pas d'espèce identifiée par la ML",
                    "region": row.get("reg_nom") or "",
                    "commune": row.get("nearest_commune") or "",
                    "latitude": row.get("latitude") or "",
                    "longitude": row.get("longitude") or "",
                    "departement": row.get("dep_nom") or "",
                    "geo_map_html": (
                        f'<a href="https://www.openstreetmap.org/?mlat={row["latitude"]}&mlon={row["longitude"]}&zoom=12" target="_blank">Voir la carte</a>'
                        if row["latitude"] and row["longitude"]
                        else "<em>Localisation non disponible</em>"
                    ),
                    "lien_doris": row["lien_doris"] or "",
                    "lien_doris_html": (
                        f'<a href="{row["lien_doris"]}" target="_blank">Voir sur DORIS</a>'
                        if row["lien_doris"]
                        else "<em>Aucun lien disponible</em>"
                    ),
                }
            })
        client.projects.import_tasks(
            id=project_id,
            request=tasks,
            return_task_ids=True,
        )
        LOGGER.info("Crops tasks imported to Label Studio", value=len(df))

def push_tasks_label_studio_no_crops(project_title: str, df: pl.DataFrame):
    client = get_label_studio_client()
    project_id = recuperation_project_id(project_title)

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
    LOGGER.info("No crops tasks have been successfully imported ; number of tasks :", value=len(df))

def delete_tasks_label_studio(project_title: str):
    client = get_label_studio_client()
    project_id = recuperation_project_id(project_title)

    tasks = client.tasks.list(project=project_id)
    if not tasks:
        LOGGER.info("No tasks to delete.")
        return

    task_ids = [t.id for t in tasks]
    for task_id in task_ids:
        client.tasks.delete(task_id)
    LOGGER.info(f"{len(task_ids)} tasks deleted from project {project_title}")

def extract_crops_data_from_label_studio(project_title: str, start_datetime: datetime, end_datetime: datetime) -> pl.DataFrame:
    """
    Extraction brute des tâches Label Studio Projet Crops.
    """
    client = get_label_studio_client()
    project_id = recuperation_project_id(project_title)

    # -------------------------
    # Récupération des tasks
    # -------------------------
    tasks = client.tasks.list(project=project_id)
    rows = []

    for task in tasks:
        annotation = task.annotations[0]  if task.annotations else None
        annotateur = None
        annotated_at = None
        source = None
        commentaire = None
        nom_scientifique = None
        espece_identifiee = None

        if annotation and isinstance(annotation, dict):
            # -------------------------
            # Labels
            # -------------------------
            results = annotation.get("result", [])
            for r in results:
                input_type = r.get("from_name")
                value = r.get("value", {})

                # Choix utilisateur
                if input_type == "decision":
                    decision = value.get("choices", [None])[0]
                    if decision == "Corriger l'espèce":
                        source = "ml_taxonomie_humaine"
                        espece_identifiee = "True"
                    elif decision == "Prédiction correcte":
                        source = "ml_taxonomie_ml"
                        espece_identifiee = "True"
                        nom_scientifique = task.data.get('species_name')
                    elif decision == "Non identifiable":
                        source = "ml_taxonomie_NI"
                        espece_identifiee = "False"
                # Si decision == corriger l'espece, il faut recupérer son nom dans un second output
                elif input_type == "espece_corrigee":
                    nom_scientifique = value.get("text", [None])[0]
                # Commentaire
                elif input_type == "commentaire":
                    commentaire = value.get("text", [None])[0]

            # Annotateur
            # -------------------------
            created_username = annotation.get("created_username")
            annotated_at = annotation.get("created_at")

            if created_username:
                annotateur = created_username.split(",")[0].strip()
            else:
                annotateur = annotation.get("completed_by")

        rows.append({
            "task_id": task.id,
            "task_created_date": getattr(task, "created_at", None),
            "id_crops": f"{task.data.get('id_observation')}_0",
            "id_observation": task.data.get("id_observation"),
            "nom_scientifique": nom_scientifique,
            "species_name": task.data.get("species_name"),
            "annotateur": annotateur,
            "annotated_at": annotated_at,
            "source": source,
            "commentaire": commentaire,
            "espece_identifiee": espece_identifiee,
            "validee": "True"
        })

    df = pl.DataFrame(rows)
    df = df.filter(
        pl.col("annotated_at").is_between(
            start_datetime,
            end_datetime
        )
    )
    return df

def extract_no_crops_data_from_label_studio(project_title: str, start_datetime: datetime, end_datetime: datetime) -> pl.DataFrame:
    """
    Extraction brute des tâches Label Studio Projet No Crops.
    """
    client = get_label_studio_client()
    project_id = recuperation_project_id(project_title)

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
        annotateur = annotation.get("created_username")
        if annotateur:
            annotateur = annotateur.split(",")[0].strip()
        else:
            annotateur = annotation.get("completed_by")

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

            # Cas pas d'espece
            if from_name == "presence":
                rows.append({
                    "task_id": task.id,
                    "task_created_date": getattr(task, "created_at", None),
                    "id_crops": f"{task.data.get('id_observation')}_0",
                    "id_observation": task.data.get("id_observation"),
                    "crop_index": 0,
                    "x": None,
                    "y": None,
                    "width": None,
                    "height": None,
                    "original_width": None,
                    "original_height": None,
                    "nom_scientifique": None,
                    "annotateur": annotateur,
                    "annotated_at": annotated_at,
                    "commentaire": commentaire,
                    "source": "ml_crops",
                    "espece_identifiee": "False",
                    "validee": "True",
                })

            # espece & image
            if from_name == "nom_espece" and r.get("type") == "textarea":
                crops.append({
                    "nom_scientifique": value.get("text", [None])[0],
                    "x": value.get("x"),
                    "y": value.get("y"),
                    "width": value.get("width"),
                    "height": value.get("height"),
                    "original_width": r.get("original_width"),
                    "original_height": r.get("original_height"),
                    "source": "ml_crops_humaine",
                    "espece_identifiee": "True",
                    "validee": "True"
                })

        # -------------------------
        # 1 ligne par crop
        # -------------------------
        for idx, crop in enumerate(crops):
            rows.append({
                "task_id": task.id,
                "task_created_date": getattr(task, "created_at", None),
                "id_crops": f"{task.data.get('id_observation')}_{idx}",
                "id_observation": task.data.get("id_observation"),
                "crop_index": idx,
                "x": crop["x"],
                "y": crop["y"],
                "width": crop["width"],
                "height": crop["height"],
                "original_width": crop["original_width"],
                "original_height": crop["original_height"],
                "nom_scientifique": crop["nom_scientifique"],
                "annotateur": annotateur,
                "annotated_at": annotated_at,
                "commentaire": commentaire,
                "source": crop["source"],
                "espece_identifiee": crop["espece_identifiee"],
                "validee": crop["validee"],
            })

    df = pl.DataFrame(rows)
    return df
