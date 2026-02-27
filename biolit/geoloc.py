import geopandas as gpd
from shapely.geometry import Point
import polars as pl
import requests
import structlog
from pathlib import Path
import zipfile
# Si besoin de checks pour tracer la carte
import os
import folium
from folium.plugins import MarkerCluster

from biolit import DATADIR, DATA_GOUV_INFO_COMMUNES_URL, DATA_GOUV_CONTOUR_COMMUNES_URL, WORLD_COAST_LINES_URL

LOGGER = structlog.get_logger()


'''
    Fonction finale :
        - Enrichissement de la commune la plus proche
        - Création de la colonne "is_coastal" -> es ce que le point est proche du littoral
        - Export vers fichier parquet
'''

def geoloc_enrichie_data_biolit():
    fn = DATADIR / "biolit_valid_observations.parquet"

    df = get_info_nearest_commune(fn)
    export_path_communes = DATADIR / "data_enrichi_with_communes.parquet"
    export_path_communes.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(export_path_communes)
    LOGGER.info(f"Files exported: {export_path_communes}")
    df = check_geoloc_distance_to_coast(export_path_communes, 8000)
    export_path_is_coastal = DATADIR / "data_enrichi_with_communes_and_is_coastal.parquet"
    df.to_parquet(export_path_is_coastal)
    LOGGER.info(f"Files exported: {export_path_is_coastal}")

'''
    Téléchargement des fichiers permettant d'enrichir les données Biolit sur les communes + Littoral Monde
    Fichiers issus du site data.gouv :
        DATA_GOUV_CONTOUR_COMMUNES_URL = "https://www.data.gouv.fr/api/1/datasets/r/00c0c560-3ad1-4a62-9a29-c34c98c3701e"
        DATA_GOUV_INFO_COMMUNES_URL = "https://www.data.gouv.fr/api/1/datasets/r/f5df602b-3800-44d7-b2df-fa40a0350325"
    Fichier issu de OpenStreetMap : tracé du littoral (monde)
        WORLD_COAST_LINES_URL = "https://osmdata.openstreetmap.de/download/coastlines-split-4326.zip"
'''

def get_geometry_communes():
    fn = DATADIR / "geoloc" / "data_gouv" / "contours_communes.json"
    if not _check_file_existence(fn):
        _download_data_contour_communes(fn)

    contours_communes = (
        gpd.read_file(fn, layer="a_com2022")
        .rename(columns={"codgeo": "code_insee", "libgeo": "nom_communes"})
    )
    LOGGER.info("contours_communes_loaded", count=len(contours_communes))
    return contours_communes

def get_info_communes():
    fn = DATADIR / "geoloc" / "data_gouv" / "info_communes.csv"
    if not _check_file_existence(fn):
        _download_data_info_communes(fn)

    info_communes = (
        pl.read_csv(fn, ignore_errors = True, schema_overrides={"code_insee": pl.Utf8})
    )
    LOGGER.info("info_communes_loaded", count=len(info_communes))
    return info_communes

def get_trace_littoral():
    fn = DATADIR / "geoloc" / "open_street_map" / "coastlines-split-4326" / "lines.shp"
    if not _check_file_existence(fn):
        _download_data_trace_littoral(fn)

    coast_gdf = gpd.read_file(fn)
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

def _download_data_contour_communes(targetpath: Path):
    targetpath.parent.mkdir(parents=True, exist_ok=True)
    _download_file_from_url(DATA_GOUV_CONTOUR_COMMUNES_URL, targetpath)
    if not targetpath.is_file():
        LOGGER.fatal("Didn't manage to properly extract the file with municipalities borders")

def _download_data_info_communes(targetpath: Path):
    targetpath.parent.mkdir(parents=True, exist_ok=True)
    _download_file_from_url(DATA_GOUV_INFO_COMMUNES_URL, targetpath)
    if not targetpath.is_file():
        LOGGER.fatal("Didn't manage to properly extract the file with municipalities information")

def _download_data_trace_littoral(targetpath: Path):
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

def _download_file_from_url(url: str, targetpath: Path):
    LOGGER.info("download_start", url=url)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(targetpath, "wb") as f:
            for chunk in r:
                if chunk:
                    f.write(chunk)

    LOGGER.info("download_success", path=str(targetpath))



'''
    Fonctions permettant d'attribuer à un point Biolit la commune la plus proche.
'''

def distance_to_communes(point: Point, communes_gdf: gpd.GeoDataFrame, sindex, search_radius: float):
    ''' Fonction permettant de déterminer le polygon le plus proche du point '''
    candidate_idx = list(
        sindex.intersection(point.buffer(search_radius).bounds)
    )

    if not candidate_idx:
        return search_radius, None, None

    candidates = communes_gdf.iloc[candidate_idx]
    distances = candidates.distance(point)
    min_idx = distances.idxmin()
    return (
        distances.min(),
        communes_gdf.loc[min_idx, "nom_communes"],
        communes_gdf.loc[min_idx, "code_insee"]
    )

def get_info_nearest_commune(file_path: Path):
    ''' On détermine la commune la plus proche '''
    communes = get_geometry_communes()
    sindex = communes.sindex

    biolit_df = (
        pl.read_parquet(file_path)
        )
    biolit_df = biolit_df.with_columns([
        pl.col("longitude_-_n1")
        .str.replace(",", ".")
        .cast(pl.Float64),
        pl.col("latitude_-_n1")
        .str.replace(",", ".")
        .cast(pl.Float64)
    ])

    # Convertir en pandas avant GeoDataFrame
    biolit_pd = biolit_df.to_pandas()

    # Créer GeoDataFrame
    gdf = gpd.GeoDataFrame(
        biolit_pd,
        geometry=gpd.points_from_xy(biolit_pd["longitude_-_n1"], biolit_pd["latitude_-_n1"]),
        crs="EPSG:4326"
    ).to_crs(epsg=2154)

    LOGGER.info("biolit df uploaded", count=len(biolit_df))

    distances = []
    nom_communes = []
    code_insee = []

    for p in gdf.geometry:
        d, n, c = distance_to_communes(p, communes, sindex, search_radius=20000)
        distances.append(d)
        nom_communes.append(n)
        code_insee.append(c)


    gdf["distance_commune_m"] = distances
    gdf["nearest_commune"] = nom_communes
    gdf["code_insee"] = code_insee

    df_export = gdf.drop(columns="geometry")
    LOGGER.info("Nearest Municipality exported", count=len(df_export))

    ''' On récupère les informations de la commune la plus proche '''
    info_communes = get_info_communes()
    info_communes = info_communes.to_pandas()

    ''' Filtre sur les colonnes qui nous intéressent '''
    info_communes = info_communes[["code_insee", "code_postal", "reg_nom", "dep_nom"]]

    df_export = df_export.merge(
        info_communes,
        on = "code_insee",
        how="left"
    )

    LOGGER.info("Nearest Municipality enriched with dep_name & region_name", count=len(df_export))
    return df_export


'''
    Fonctions permettant de calculer la distance à la côte
'''

def distance_to_coast(point: Point, coast_gdf: gpd.GeoDataFrame, sindex, search_radius: float):
    ''' Fonction de Calcul de distance entre le point et la ligne de côte '''
    candidate_idx = list(
        sindex.intersection(point.buffer(search_radius).bounds)
    )

    if not candidate_idx:
        return search_radius

    candidates = coast_gdf.iloc[candidate_idx]
    return candidates.distance(point).min()

def check_geoloc_distance_to_coast(file_path: Path, distance_max: float):
    coast_gdf = get_trace_littoral()
    coast_sindex = coast_gdf.sindex

    biolit_df = (
        pl.read_parquet(file_path)
        )
    LOGGER.info("biolit_df", count=len(biolit_df))


    gdf = gpd.GeoDataFrame(biolit_df.to_pandas(), geometry=gpd.points_from_xy(biolit_df["longitude_-_n1"], biolit_df["latitude_-_n1"]), crs="EPSG:4326").to_crs(epsg=2154)
    distances = []

    for p in gdf.geometry:
        d = round(distance_to_coast(p, coast_gdf, coast_sindex, search_radius=20000))
        distances.append(d)

    gdf["distance_to_coast"] = distances

    gdf["is_coastal"] = (
        gdf["distance_to_coast"].notna()
        & (gdf["distance_to_coast"] <= distance_max)
    )

    gdf_export = gdf.drop(columns="geometry", errors="ignore")

    LOGGER.info("Biolit Data Points enriched with distance to coast", nb_not_coastal = (~gdf_export["is_coastal"]).sum(), nb_coastal = gdf_export["is_coastal"].sum())
    return gdf_export

'''
    Si besoin de checks - possibilité de créer une carte avec en vert les points proches de la côte et en rouge les points éloignés de la côte
'''

def carte(file_to_map: Path):
    if not os.path.exists(file_to_map):
        LOGGER.info("The file you want to map does not exist")
    else:
        df = pl.read_parquet(file_to_map)
        gdf = gpd.GeoDataFrame(df.to_pandas(), geometry=gpd.points_from_xy(df["longitude_-_n1"], df["latitude_-_n1"]), crs="EPSG:4326").dropna(subset=["latitude_-_n1", "longitude_-_n1"])

        carte = folium.Map(location=[48.95, 2.29], zoom_start=6)

        marker_cluster = MarkerCluster().add_to(carte)

        for _, row in gdf.iterrows():
            color = "green" if row["is_coastal"] else "red"

            popup_text = (
                f"ID: {row['id']}<br>"
                f"Côtier: {row['is_coastal']}<br>"
                f"Distance : {row['distance_to_coast']}m (Peut être plus si 20 000m)"
            )

            folium.CircleMarker(
                location=[row["latitude_-_n1"], row["longitude_-_n1"]],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_text,
            ).add_to(marker_cluster)
        carte.save('carte.html')
        LOGGER.info('Map created')


