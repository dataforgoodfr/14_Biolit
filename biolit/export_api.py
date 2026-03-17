import requests
import polars as pl
import structlog

LOGGER = structlog.get_logger()

# ------------------------------
# Helper pour récupérer une clé dans meta
# ------------------------------
def get_meta(meta: dict, key: str):
    """Retourne la première valeur d'une clé meta, ou None si absente"""
    if not meta:
        return None
    value = meta.get(key)
    if isinstance(value, list) and value:
        return value[0]
    return value


# ------------------------------
# FETCH API
# ------------------------------
def fetch_biolit_from_api(per_page=100, max_pages=5):
    """
    Récupère les observations depuis l'API Biolit.
    Limite par défaut à max_pages pour éviter les 150+ pages.
    """
    url_base = "https://biolit.fr/wp-json/biolitapi/v1/observations"
    all_data = []

    for page in range(1, max_pages + 1):
        url = f"{url_base}?per_page={per_page}&page={page}"
        LOGGER.info(f"Fetching page {page} from API")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            break
        all_data.extend(data)

    LOGGER.info(f"Fetched {len(all_data)} observations total")
    return all_data


# ------------------------------
# ADAPT API -> PARQUET
# ------------------------------
def adapt_api_to_parquet_schema(data: list) -> pl.DataFrame:
    """
    Transforme la structure API Biolit en DataFrame pour parquet.
    """
    rows = []

    for item in data:
        obs = item.get("observation", {})
        meta = obs.get("meta", {})
        parents = item.get("parents", {})
        especes = item.get("especes", [])

        quadra = parents.get("quadra", {})
        abb = parents.get("abb", {})
        abb_meta = abb.get("meta", {})

        # Gestion des espèces
        nom_scientifique = None
        nom_commun = None
        nombre_mollusques = None

        if especes:
            nom_scientifique = especes[0].get("nom")
            nombre_mollusques = especes[0].get("nombre_presents")

        row = {
            # Niveau N1 (Quadra)
            "protocole": get_meta(meta, "jet_tax__protocole"),
            "ID - N1": quadra.get("ID"),
            "titre - N1": quadra.get("title"),
            "lien - N1": get_meta(meta, "_url_sortie"),
            "auteur - N1": None,
            "images - N1": None,
            "date - N1": obs.get("date"),
            "heure-de-debut - N1": get_meta(meta, "heure-debut"),
            "heure-de-fin - N1": get_meta(meta, "heure-fin"),
            "latitude - N1": get_meta(meta, "latitude"),
            "longitude - N1": get_meta(meta, "longitude"),
            "relais-local - N1": get_meta(abb_meta, "relais-local"),
            "nom du lieu - N1": get_meta(abb_meta, "nom-du-lieu-abb"),

            # Observation
            "ID - observation": obs.get("ID"),
            "titre - observation": obs.get("title"),
            "lien - observation": obs.get("link"),
            "Nom scientifique - observation": nom_scientifique,
            "Nom commun - observation": nom_commun,
            "programme espèce": get_meta(meta, "jet_tax__categorie-programme"),
            "images - observation": obs.get("images"),
            "nombre de mollusques - observation": nombre_mollusques,
            "validee - observation": get_meta(meta, "validee"),
            "espece identifiable ? - observation": get_meta(meta, "espece-identifiee"),
        }

        rows.append(row)

    return pl.DataFrame(rows)


# ------------------------------
# LOAD (Fetch + Adapt)
# ------------------------------
def load_biolit_from_api(per_page=100, max_pages=5) -> pl.DataFrame:
    """
    Récupère et transforme les données Biolit depuis l'API.
    """
    raw_data = fetch_biolit_from_api(per_page=per_page, max_pages=max_pages)
    df = adapt_api_to_parquet_schema(raw_data)
    return df

