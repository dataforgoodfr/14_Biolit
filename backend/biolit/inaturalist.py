import polars as pl
import requests


def fetch_inaturalist_observations(specie: str):
    options = {
        "has[]": "photos",
        "per_page": 10,
        "taxon_name": specie,
        "quality_grade": "research",
    }
    url = "https://www.inaturalist.org/observations.json?" + "&".join(
        [f"{k}={v}" for k, v in options.items()]
    )
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Failed to fetch observations: {r.status_code} {r.text}")
    return r.json()


def parse_inaturalist_api(content: list) -> pl.DataFrame:
    parsed_observations = [
        {
            "inat_obs_id": obs["id"],
            "created_at": obs["created_at"],
            "inat_taxon_id": obs["taxon"]["id"],
            "taxon_name": obs["taxon"]["name"].lower(),
            "inat_taxon_rank": obs["taxon"]["rank"],
            "inat_image_url": obs["photos"][0]["large_url"],
            "inat_image_id": obs["photos"][0]["id"],
        }
        for obs in content
    ]
    return pl.DataFrame(parsed_observations).cast({"created_at": pl.Datetime})
