from dotenv import load_dotenv
import geopandas as gpd
from shapely.geometry import Point
from typing import Tuple, Optional
import polars as pl
import pandas as pd
import requests
from sqlalchemy import create_engine
import structlog
from pathlib import Path
import zipfile
import os
import folium
from folium.plugins import MarkerCluster

from biolit import DATADIR, DATA_GOUV_INFO_COMMUNES_URL, DATA_GOUV_CONTOUR_COMMUNES_URL, WORLD_COAST_LINES_URL

LOGGER = structlog.get_logger()
load_dotenv()


def geoloc_enrichie_data_biolit():
    """
    Connexion à la base de données issue de l'API
    Fonction finale :
        - Enrichissement de la commune la plus proche
        - Création de la colonne "is_coastal" -> est ce que le point est proche du littoral
        - Export vers fichier parquet
    """
    df_biolit = get_biolit_df_from_db()
    df = get_info_nearest_commune(df_biolit)
    df_coastal = get_info_distance_to_coast(df, 8000)

    export_path_geoloc_enrichie = DATADIR / "data_enrichi_with_communes_and_is_coastal.parquet"
    export_path_geoloc_enrichie.parent.mkdir(parents=True, exist_ok=True)
    df_coastal.to_parquet(export_path_geoloc_enrichie)
    LOGGER.info(f"Files exported: {export_path_geoloc_enrichie}")

def get_biolit_df_from_db():
    try:
        engine = create_engine(os.environ["DATABASE_URL"])

        query = "SELECT * FROM observations_biolit_api"
        df = pd.read_sql(query, engine)

        LOGGER.info("biolit df uploaded", count=len(df))
        LOGGER.info("Colonnes df", values=list(df.columns))

        return df

    except Exception as e:
        LOGGER.error("Erreur lors de la récupération des données depuis PostgreSQL", error=str(e))
        return None

def download_geometry_communes(targetpath: Path):
    """
    Fonction permettant de récupérer les géométries des communes depuis :
        DATA_GOUV_CONTOUR_COMMUNES_URL = "https://www.data.gouv.fr/api/1/datasets/r/00c0c560-3ad1-4a62-9a29-c34c98c3701e"
    """
    targetpath.parent.mkdir(parents=True, exist_ok=True)
    _download_file_from_url(DATA_GOUV_CONTOUR_COMMUNES_URL, targetpath)
    if not targetpath.is_file():
        LOGGER.fatal("Didn't manage to properly extract the file with municipalities borders")

def get_geometry_communes(file: Path) -> gpd.GeoDataFrame:
    if _check_file_existence(file):
        geometry_communes = (
            gpd.read_file(file, layer="a_com2022")
            .rename(columns={"codgeo": "code_insee", "libgeo": "nom_communes"})
        )
        LOGGER.info("geometry_communes_loaded", count=len(geometry_communes))
    return geometry_communes

def download_info_communes(targetpath: Path):
    """
    Fonction permettant de récupérer les géométries des communes depuis :
        DATA_GOUV_INFO_COMMUNES_URL = "https://www.data.gouv.fr/api/1/datasets/r/f5df602b-3800-44d7-b2df-fa40a0350325"
    """
    targetpath.parent.mkdir(parents=True, exist_ok=True)
    _download_file_from_url(DATA_GOUV_INFO_COMMUNES_URL, targetpath)
    if not targetpath.is_file():
        LOGGER.fatal("Didn't manage to properly extract the file with municipalities information")

def get_info_communes(file: Path) -> pd.DataFrame:
    if _check_file_existence(file):
        info_communes = (
            pl.read_csv(file, ignore_errors = True, schema_overrides={"code_insee": pl.Utf8})
        )
        # Filtre sur les colonnes qui nous intéressent
        info_communes = info_communes.select(["code_insee", "code_postal", "reg_nom", "dep_nom"]).to_pandas()

        LOGGER.info("info_communes_loaded", count=len(info_communes))
    return info_communes

def download_trace_littoral(targetpath: Path):
    """
    Fonction permettant de récupérer le tracé du littoral (monde) depuis OpenStreetMap :
        WORLD_COAST_LINES_URL = "https://osmdata.openstreetmap.de/download/coastlines-split-4326.zip"
    """
    targetpath.parent.mkdir(parents=True, exist_ok=True)
    base_path = DATADIR / "geoloc" / "open_street_map"
    zip_path = base_path / "coastlines.zip"
    extract_path = base_path / "coastlines-split-4326"

    if not zip_path.exists():
        _download_file_from_url(WORLD_COAST_LINES_URL, zip_path)
    else:
        LOGGER.info("download_skipped", reason="zip already exists")

    if not zip_path.is_file():
        LOGGER.fatal("Download failed")
        return
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(base_path)
    LOGGER.info("Extraction successful")
    shp_files = list(extract_path.glob("*.shp"))
    if not shp_files:
        LOGGER.fatal("No shapefile found after extraction")
    else:
        LOGGER.info("Shapefile ready", file=str(shp_files[0]))

def simplify_trace_littoral(file: Path) -> Path:
    """ Diminuer la précision du littoral pour ne pas surcharger nos tâches """
    if not _check_file_existence(file):
        LOGGER.info(f"This file does not exist: {file}")
        return
    coast_gdf = gpd.read_file(file)
    coast_gdf['geometry'] = coast_gdf['geometry'].simplify(tolerance=50)
    export_path_trace_littoral_simplify = DATADIR / "geoloc" / "coast_file_simplified" / "coastlines.shp"
    export_path_trace_littoral_simplify.parent.mkdir(parents=True, exist_ok=True)
    coast_gdf.to_file(export_path_trace_littoral_simplify)
    LOGGER.info(f"Files exported: {export_path_trace_littoral_simplify}")
    return

def get_trace_littoral(file: Path) -> gpd.GeoDataFrame:
    if not _check_file_existence(file):
        LOGGER.info(f"This file does not exist: {file}")
        return
    coast_gdf = gpd.read_file(file)
    coast_gdf = coast_gdf.to_crs(epsg=2154)
    LOGGER.info("info_coast_loaded", count=len(coast_gdf))
    return coast_gdf

def _check_file_existence(file: Path):
    if not file.exists():
        LOGGER.info("file_missing", path=str(file))
        return False

    if not file.is_file():
        LOGGER.fatal(
            "The following path has been created, but it is not a standard file",
            path=file
        )
    return True

def _download_file_from_url(url: str, targetpath: Path):
    LOGGER.info("download_start", url=url)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(targetpath, "wb") as f:
            for chunk in r:
                if chunk:
                    f.write(chunk)

    LOGGER.info("download_success", path=str(targetpath))

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
    path_geometry_communes = DATADIR / "geoloc" / "data_gouv" / "geometry_communes.json"
    if not _check_file_existence(path_geometry_communes):
        download_geometry_communes(path_geometry_communes)

    communes = get_geometry_communes(path_geometry_communes)
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
    path_info_communes = DATADIR / "geoloc" / "data_gouv" / "info_communes.csv"
    if not _check_file_existence(path_info_communes):
        download_info_communes(path_info_communes)
    info_communes = get_info_communes(path_info_communes)

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
    path_trace_littoral = DATADIR / "geoloc" / "open_street_map" / "coastlines-split-4326" / "lines.shp"
    path_trace_littoral_simplified = DATADIR / "geoloc" / "coast_file_simplified" / "coastlines.shp"
    if not _check_file_existence(path_trace_littoral_simplified):
        if not _check_file_existence(path_trace_littoral):
            download_trace_littoral(path_trace_littoral)
        simplify_trace_littoral(path_trace_littoral)

    coast_gdf = get_trace_littoral(path_trace_littoral_simplified)
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

def carte_points_biolit_checks_geoloc(file_to_map: Path):
    """
    Si besoin de checks - possibilité de créer une carte avec en vert les points proches de la côte et en rouge les points éloignés de la côte
    """
    if not os.path.exists(file_to_map):
        LOGGER.info("The file you want to map does not exist")
        return

    df = pl.read_parquet(file_to_map)
    gdf = gpd.GeoDataFrame(df.to_pandas(), geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326").dropna(subset=["latitude", "longitude"])

    carte = folium.Map(location=[48.95, 2.29], zoom_start=6)

    marker_cluster = MarkerCluster().add_to(carte)

    for _, row in gdf.iterrows():
        color = "green" if row["is_coastal"] else "red"

        popup_text = (
            f"ID: {row['id_observation']}<br>"
            f"Côtier: {row['is_coastal']}<br>"
            f"Distance : {row['distance_to_coast']} m"
        )

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_text,
        ).add_to(marker_cluster)
    carte.save(DATADIR / 'carte_points_biolit.html')
    LOGGER.info('Map created')

