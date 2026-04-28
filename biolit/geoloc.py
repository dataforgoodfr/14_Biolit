import geopandas as gpd
from shapely.geometry import Point
from typing import Tuple, Optional
import pandas as pd
import requests
import structlog
import io
from io import BytesIO
from pathlib import Path
import tempfile
import zipfile

from biolit import DATA_GOUV_INFO_COMMUNES_URL, DATA_GOUV_CONTOUR_COMMUNES_URL, WORLD_COAST_LINES_URL
from biolit.create_table import load_observations_from_db
from biolit.s3 import (
    create_s3_client,
    _check_file_existence_s3,
    _read_file_s3
)

LOGGER = structlog.get_logger()


def geoloc_enrichie_data_biolit_db(engine):
    """
    Pipeline :
    DB → enrichissement → dataframe
    """

    # 1. Load depuis PostgreSQL
    df_biolit = get_biolit_df_from_db(engine)

    # 2. Enrichissement commune
    df = get_info_nearest_commune(df_biolit)

    # 3. Enrichissement littoral
    df_coastal = get_info_distance_to_coast(df, 8000)

    LOGGER.info("Geoloc enrichment done", count=len(df_coastal))

    return df_coastal

def get_biolit_df_from_db(engine) -> pd.DataFrame:
    df = load_observations_from_db(engine)

    LOGGER.info("biolit df loaded from DB", count=len(df))

    return df.to_pandas()

def get_geometry_communes() -> gpd.GeoDataFrame:
    client = create_s3_client()
    key = "geoloc/data_gouv/geometry_communes.parquet"
    bucket_name = "biolit-uploads"
    url = DATA_GOUV_CONTOUR_COMMUNES_URL

    if not _check_file_existence_s3(client, bucket_name, key):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            LOGGER.info("download_start", url=url)

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                file_path = tmpdir / "geometry_communes.json"
                with open(file_path, "wb") as f:
                    for chunk in r:
                        if chunk:
                            f.write(chunk)
            geometry_communes = (
                gpd.read_file(file_path, layer="a_com2022")
                .rename(columns={"codgeo": "code_insee", "libgeo": "nom_communes"})
            )

        # Enregistrement sur le S3
        buffer = BytesIO()
        geometry_communes.to_parquet(buffer)
        buffer.seek(0)
        client.put_object(
            Body=buffer,
            Bucket=bucket_name,
            Key=key,
            ContentLength=buffer.getbuffer().nbytes,
        )
        LOGGER.info("Parquet uploaded", path=f"s3://{bucket_name}/{key}")

    data = _read_file_s3(client, bucket_name, key)
    gdf = gpd.read_parquet(io.BytesIO(data))

    LOGGER.info("geometry_communes_loaded", count=len(gdf))
    return gdf

def get_info_communes() -> pd.DataFrame:
    client = create_s3_client()
    key = "geoloc/data_gouv/info_communes.parquet"
    bucket_name = "biolit-uploads"
    url = DATA_GOUV_INFO_COMMUNES_URL

    if not _check_file_existence_s3(client, bucket_name, key):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            LOGGER.info("download_start", url=url)

            file_path = tmpdir / "info_communes.csv"

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            info_communes = pd.read_csv(
                file_path,
                dtype={
                    "code_insee": str,
                    "code_postal": str,
                },
                low_memory=False
            )

        buffer = BytesIO()
        info_communes.to_parquet(buffer)
        buffer.seek(0)

        client.put_object(
            Body=buffer,
            Bucket=bucket_name,
            Key=key,
            ContentLength=buffer.getbuffer().nbytes,
        )

        LOGGER.info("Parquet uploaded", path=f"s3://{bucket_name}/{key}")

    data = _read_file_s3(client, bucket_name, key)
    df = pd.read_parquet(io.BytesIO(data))

    df = df[["code_insee", "code_postal", "reg_nom", "dep_nom"]]

    LOGGER.info("info_communes_loaded", count=len(df))
    return df

def get_trace_littoral() -> gpd.GeoDataFrame:
    client = create_s3_client()
    bucket_name = "biolit-uploads"
    key = "geoloc/osm/coastlines.parquet"
    url = WORLD_COAST_LINES_URL

    if not _check_file_existence_s3(client, bucket_name, key):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            zip_path = tmpdir / "coastlines.zip"

            LOGGER.info("download_start", url=url)

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)

            shp = list(tmpdir.rglob("*.shp"))[0]
            gdf = gpd.read_file(shp).to_crs(epsg=2154)

        buffer = BytesIO()
        gdf.to_parquet(buffer)
        buffer.seek(0)

        client.put_object(
            Body=buffer,
            Bucket=bucket_name,
            Key=key,
            ContentLength=buffer.getbuffer().nbytes,
        )

        LOGGER.info("Parquet uploaded", path=f"s3://{bucket_name}/{key}")

    data = _read_file_s3(client, bucket_name, key)

    return gpd.read_parquet(io.BytesIO(data))

def distance_to_communes(point: Point, communes_gdf: gpd.GeoDataFrame, sindex, search_radius: float = 20000) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Fonction permettant de déterminer le polygon le plus proche du point
    """
    candidate_idx = list(
        sindex.intersection(point.buffer(search_radius).bounds)
    )

    if not candidate_idx:
        return None, None, None

    candidates = communes_gdf.iloc[candidate_idx]
    distances = candidates.distance(point)
    min_idx = distances.idxmin()
    return (
        distances.min(),
        communes_gdf.loc[min_idx, "nom_communes"],
        communes_gdf.loc[min_idx, "code_insee"]
    )

def get_info_nearest_commune(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction permettant d'attribuer à un point Biolit la commune la plus proche + info departement / region
    """
    # Points DB Biolit
    biolit_df = frame
    gdf = gpd.GeoDataFrame(
        biolit_df,
        geometry=gpd.points_from_xy(biolit_df["longitude"], biolit_df["latitude"]),
        crs="EPSG:4326"
    ).to_crs(epsg=2154)

    # Information Géometrie Communes
    communes = get_geometry_communes()
    sindex = communes.sindex

    # Recherche de la commune la plus proche
    results = gdf.geometry.apply(
        lambda p: distance_to_communes(p, communes, sindex, search_radius=20000)
    )
    gdf["distance_commune_m"] = results.apply(lambda x: x[0])
    gdf["nearest_commune"] = results.apply(lambda x: x[1])
    gdf["code_insee"] = results.apply(lambda x: x[2])

    df_export = gdf.drop(columns="geometry")

    # Informations sur la commune la plus proche
    info_communes = get_info_communes()

    df_export = df_export.merge(
        info_communes,
        on = "code_insee",
        how="left"
    )

    LOGGER.info("Nearest Municipality enriched with dep_name & region_name", count=len(df_export))
    return df_export

def distance_to_coast(point: Point, coast_gdf: gpd.GeoDataFrame, sindex, search_radius: float = 20000) -> Optional[float]:
    """ Fonction de Calcul de distance entre le point et la ligne de côte """
    candidate_idx = list(
        sindex.intersection(point.buffer(search_radius).bounds)
    )

    if not candidate_idx:
        return

    candidates = coast_gdf.iloc[candidate_idx]
    return candidates.distance(point).min()

def get_info_distance_to_coast(frame: pd.DataFrame, distance_max: float = 8000) -> pd.DataFrame:
    # Récupération Tracé Littoral
    coast_gdf = get_trace_littoral()
    coast_sindex = coast_gdf.sindex

    # Points Biolit
    biolit_df = frame
    gdf = gpd.GeoDataFrame(biolit_df, geometry=gpd.points_from_xy(biolit_df["longitude"], biolit_df["latitude"]), crs="EPSG:4326").to_crs(epsg=2154)
    distances = []

    for p in gdf.geometry:
        d = distance_to_coast(p, coast_gdf, coast_sindex, search_radius=20000)
        distances.append(d)

    gdf["distance_to_coast"] = distances

    gdf["is_coastal"] = (
        gdf["distance_to_coast"].notna()
        & (gdf["distance_to_coast"] <= distance_max)
    )

    gdf_export = gdf.drop(columns="geometry", errors="ignore")

    LOGGER.info("Biolit Data Points enriched with distance to coast", nb_not_coastal = (~gdf_export["is_coastal"]).sum(), nb_coastal = gdf_export["is_coastal"].sum())
    return gdf_export