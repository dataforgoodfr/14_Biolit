import os
from dotenv import load_dotenv
import structlog
from label_studio_sdk import LabelStudio

LOGGER = structlog.get_logger()
load_dotenv()

def push_tasks_to_labelstudio(df):
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    PROJECT_ID = 9

    url = "http://label-studio:8080"

    client = LabelStudio(base_url=url, api_key=api_key)

    tasks = []

    for row in df.to_dicts():
        url = row["photos"]
        filename = url.split("/")[-1]
        tasks.append({
            "data": {
                "image": f"s3://crops-data/{row['id_observation']}/{filename}",
                "latitude": row["latitude"],
                "longitude": row["longitude"]
            }
        })

    # Import tasks
    resp = client.projects.import_tasks(
        id=PROJECT_ID,
        request=tasks,
        return_task_ids=True,
    )
    LOGGER.info(resp)

