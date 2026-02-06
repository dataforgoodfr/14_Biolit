from shapely.geometry import LineString
import re
import geopandas as gpd
import polars as pl
import requests
import os
import tarfile
# Pour checker les résultats sur une carte
import folium
from folium.plugins import MarkerCluster

'''
Fonctions pour checker si un point GPS est situé à proximité du litoral.
Solution basée sur Geopandas - calcul de distance entre le litoral et le point GPS.

Commande : uv run python geoloc/check_geoloc.py

2 fonctions principales :
    check_geoloc_distance_to_coast(file_path, distance_max):
        -> file_path : chemin vers le fichier avec les points qu'on veut checker
        -> distance_max : si le point est à plus de distance que distance_max -> hors côte

    carte(file_to_map):
        -> file_to_map: fichier enrichi de la donnée distance à la côte à mapper
        -> permet de créer une map avec l'ensemble des points du dataset pour check

A creuser :
    - Normaliser le df (nom de colonnes, gestion des duplicates, etc.)
    - Dans un 2e temps : essayer de voir si on peut retrouver les bonnes lat/long à partir du nom de lieu. (API Nominatim ?)
    - Possibilité de checker les résultats avec : https://nominatim.openstreetmap.org/ui/reverse.html?lat=49.4&lon=-8.79&zoom=18
'''


def retrieve_data_from_url_zip(site_url, file_name, path):
    # Fonction pour récupérer le fichier permettant de dessiner le littoral
    # Depuis le site : https://www.evl.uic.edu/pape/data/WDB/
    os.makedirs(f"{path}", exist_ok=True)

    url = site_url + file_name
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        local_filename = f"{path}/{file_name}"
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extraire les fichiers qui sont au format compressés
    with tarfile.open(f"{path}/{file_name}", "r:gz") as tar:
        tar.extractall(path=f"{path}")


def parse_cil_file_to_geo_df(path):
    # Fonction pour dessiner le litoral
    segments = []

    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("segment"):
            match = re.search(r"segment\s+\d+\s+rank\s+(\d+)\s+points\s+(\d+)", line)
            if not match:
                i += 1
                continue

            rank = int(match.group(1))
            n_points = int(match.group(2))
            coords = []
            for j in range(1, n_points + 1):
                lat, lon = map(float, lines[i + j].split())
                coords.append((lon, lat))

            if len(coords) >= 2:
                geom = LineString(coords)
                segments.append({
                    "rank": rank,
                    "geometry": geom
                })

            i += n_points + 1
        else:
            i += 1
    gdf = gpd.GeoDataFrame(segments, crs="EPSG:4326")
    return gdf


def distance_to_coast(point, coast_gdf, sindex, search_radius=20000):
    # Fonction de Calcul de distance entre le point et la ligne de côte
    candidate_idx = list(
        sindex.intersection(point.buffer(search_radius).bounds)
    ) # On vérifie que si on trace un cercle autour de notre point de rayon search_radius -> on intercepte un des segments de la côte

    if not candidate_idx:
        return search_radius

    candidates = coast_gdf.iloc[candidate_idx]
    return candidates.distance(point).min()


def check_geoloc_distance_to_coast(file_to_check, distance_max):
    # Si le fichier de contour du littoral n'existe pas, le télécharger.
    file_coast = os.path.join("data", "geoloc", "WDB", "europe-cil.txt")
    if not os.path.exists(file_coast):
        path = os.path.join("data", "geoloc")
        retrieve_data_from_url_zip("https://www.evl.uic.edu/pape/data/WDB/", "europe.tar.gz", path)

    coast = parse_cil_file_to_geo_df(file_coast).to_crs(epsg=2154) # EPSG 2154 pour convertir en m (+France)
    coast_sindex = coast.sindex

    # Test sur les données du sample_data
    biolit = (
        pl.read_csv(file_to_check, separator=";", truncate_ragged_lines=True,
            schema_overrides={
                "latitude - N1": pl.Utf8,
                "longitude - N1": pl.Utf8,
                "ID - observation": pl.Int64,
            },
            null_values=["", "NA", "null"]
        )
        .with_columns([pl.col("latitude - N1").str.replace(",", ".").cast(pl.Float64),
            pl.col("longitude - N1").str.replace(",", ".").cast(pl.Float64),
        ])
        .rename(lambda c: c.replace(" - observation", "").lower().replace(" ", "_"))
    )


    gdf = gpd.GeoDataFrame(biolit.to_pandas(), geometry=gpd.points_from_xy(biolit["longitude_-_n1"], biolit["latitude_-_n1"]), crs="EPSG:4326").to_crs(epsg=2154)

    distances = []

    for p in gdf.geometry:
        d = round(distance_to_coast(p, coast, coast_sindex, search_radius=20000))
        distances.append(d)

    gdf["distance_to_coast"] = distances

    COAST_THRESHOLD = distance_max

    gdf["is_coastal"] = (
        gdf["distance_to_coast"].notna()
        & (gdf["distance_to_coast"] <= COAST_THRESHOLD)
    )

    rename_dict = {
        "id_-_n1": "id",
        "nom_du_lieu_-_n1": "nom_du_lieu",
        "latitude_-_n1": "latitude",
        "longitude_-_n1": "longitude",
        "distance_to_coast": "distance_cote",
        "is_coastal": "is_coastal"
    }

    gdf_export = gdf[[
        "id_-_n1",
        "nom_du_lieu_-_n1",
        "latitude_-_n1",
        "longitude_-_n1",
        "distance_to_coast",
        "is_coastal"
    ]].rename(columns=rename_dict)

    gdf_export = gdf_export.drop(columns="geometry", errors="ignore")
    gdf_export.to_csv("geoloc/biolit_with_coast_distance.csv")

    print("Fichier créé")
    nb_not_coastal = (~gdf_export["is_coastal"]).sum()
    nb_coastal = gdf_export["is_coastal"].sum()
    print(f"Nombre de points non côtiers : {nb_not_coastal}")
    print(f"Nombre de points côtiers : {nb_coastal}")


def carte(file_to_map):

    if not os.path.exists(file_to_map):
        print("Le fichier n'existe pas.")
    else:
        df = pl.read_csv(file_to_map)
        gdf = gpd.GeoDataFrame(df.to_pandas(), geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
        gdf = gdf.dropna(subset=["latitude", "longitude"])

        carte = folium.Map(location=[48.95, 2.29], zoom_start=6)

        marker_cluster = MarkerCluster().add_to(carte)

        for _, row in gdf.iterrows():
            color = "green" if row["is_coastal"] else "red"

            popup_text = (
                f"ID: {row['id']}<br>"
                f"Côtier: {row['is_coastal']}<br>"
                f"Distance : {row['distance_cote']}m (Peut être plus si 20 000m)"
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
        carte.save('geoloc/carte.html')


file_path = os.path.join("data", "export_biolit.csv") # Fichier d'extract de DB Biolit.
distance_max = 5_000 # Distance Max à la côte.
check_geoloc_distance_to_coast(file_path, distance_max)
carte('geoloc/biolit_with_coast_distance.csv')