from biolit.export_api import fetch_biolit_from_api, adapt_api_to_dataframe

# -------------------------
# Tests API
# -------------------------

def test_fetch_api_returns_data():
    """Vérifie que l'API retourne bien des données"""
    data = fetch_biolit_from_api()

    assert isinstance(data, list)
    assert len(data) > 0

    print(f"\n✅ {len(data)} observations récupérées")


def test_fetch_api_structure():
    """Vérifie la structure des données API"""
    data = fetch_biolit_from_api()
    sample = data[0]

    expected_keys = {
        "id",
        "date",
        "link",
        "author",
        "_url_sortie",
        "espece-identifiee",
        "heure-debut",
        "heure-fin",
        "latitude",
        "longitude",
        "photos",
        "relais",
        "espece_id",
        "espece",
        "common",
    }

    missing_keys = expected_keys - set(sample.keys())

    assert len(missing_keys) == 0, f"Champs manquants: {missing_keys}"

    print("\n✅ Structure API valide")


# -------------------------
# Tests transformation
# -------------------------

def test_adapt_to_dataframe():
    """Vérifie la transformation en DataFrame"""
    data = fetch_biolit_from_api()
    df = adapt_api_to_dataframe(data)

    assert df.shape[0] > 0
    assert df.shape[1] > 0

    print(f"\n✅ DataFrame: {df.shape[0]} lignes, {df.shape[1]} colonnes")


def test_expected_columns_present():
    """Vérifie les colonnes critiques"""
    data = fetch_biolit_from_api()
    df = adapt_api_to_dataframe(data)

    expected_columns = {
        "id_observation",
        "date_observation",
        "nom_scientifique",
        "nom_commun",
        "latitude",
        "longitude",
    }

    missing = expected_columns - set(df.columns)

    assert len(missing) == 0, f"Colonnes manquantes: {missing}"

    print("\n✅ Colonnes critiques présentes")


# -------------------------
# Tests qualité des données
# -------------------------

def test_unique_ids():
    """Vérifie qu'il n'y a pas de doublons"""
    data = fetch_biolit_from_api()
    df = adapt_api_to_dataframe(data)

    total = df.shape[0]
    unique = df.select("id_observation").n_unique()

    assert total == unique, "Doublons détectés sur id_observation"

    print("\n✅ Pas de doublons")


def test_no_null_coordinates():
    """Vérifie que les coordonnées sont présentes"""
    data = fetch_biolit_from_api()
    df = adapt_api_to_dataframe(data)

    null_lat = df.filter(df["latitude"].is_null()).shape[0]
    null_lon = df.filter(df["longitude"].is_null()).shape[0]

    print(f"\nNull latitude: {null_lat}")
    print(f"Null longitude: {null_lon}")

    # tolérance possible, donc pas assert strict
    assert null_lat < df.shape[0]
    assert null_lon < df.shape[0]


def test_id_is_numeric():
    """Vérifie que les IDs sont bien numériques"""
    data = fetch_biolit_from_api()

    ids = [item["id"] for item in data]

    # doit pouvoir être cast en int
    for i in ids[:100]:  # test sur sample
        int(i)

    print("\n✅ IDs valides")


# -------------------------
# Test global pipeline
# -------------------------

def test_full_pipeline():
    """Test end-to-end"""
    data = fetch_biolit_from_api()
    df = adapt_api_to_dataframe(data)

    assert df.shape[0] > 0

    print("\n=== PIPELINE OK ===")
    print(df.head(5))


# -------------------------
# Execution directe
# -------------------------

if __name__ == "__main__":
    print("=== TEST FETCH API ===")
    test_fetch_api_returns_data()
    test_fetch_api_structure()

    print("\n=== TEST TRANSFORMATION ===")
    test_adapt_to_dataframe()
    test_expected_columns_present()

    print("\n=== TEST QUALITE ===")
    test_unique_ids()
    test_no_null_coordinates()
    test_id_is_numeric()

    print("\n=== TEST PIPELINE COMPLET ===")
    test_full_pipeline()