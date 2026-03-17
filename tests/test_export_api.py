
from biolit.export_api import fetch_biolit_from_api, adapt_api_to_parquet_schema

# -------------------------
# Fonctions d'inspection
# -------------------------

def inspect_api_structure(raw_data):
    """Affiche les clés du niveau supérieur dans l'API"""
    keys = set()
    for item in raw_data:
        keys.update(item.keys())
    print("TOP LEVEL KEYS:")
    for k in sorted(keys):
        print("-", k)

def inspect_meta_keys(raw_data):
    """Affiche toutes les clés présentes dans le champ 'meta' de chaque observation"""
    meta_keys = set()
    for item in raw_data:
        if "observation" not in item:
            continue
        meta = item["observation"].get("meta", {})
        meta_keys.update(meta.keys())
    print("META KEYS:")
    for k in sorted(meta_keys):
        print("-", k)

def inspect_meta_values(raw_data, field):
    """Affiche les valeurs et leur quantité pour un champ du meta"""
    values = {}
    for item in raw_data:
        if "observation" not in item:
            continue
        meta = item["observation"].get("meta", {})
        val = meta.get(field)
        if isinstance(val, list) and val:
            val = val[0]
        values[val] = values.get(val, 0) + 1
    print(f"\nVALUES FOR {field}:")
    for k, v in values.items():
        print(k, ":", v)

# -------------------------
# Tests
# -------------------------

def test_fetch_small_sample():
    """Test fetch API avec un petit nombre de pages pour debug"""
    raw_data = fetch_biolit_from_api(per_page=1000, max_pages=5)  # petit sample
    assert isinstance(raw_data, list)
    print(f"\nNombre total d'éléments récupérés: {len(raw_data)}")
    inspect_api_structure(raw_data)
    inspect_meta_keys(raw_data)
    inspect_meta_values(raw_data, "validee")

def test_load_and_adapt():
    """Test l'adaptation de l'API vers le schéma parquet"""
    raw_data = fetch_biolit_from_api(per_page=1000, max_pages=5)
    df = adapt_api_to_parquet_schema(raw_data)
    print(f"\nDataframe adapté avec {len(df)} lignes")
    print(df.head(10))  # affiche les 10 premières lignes pour vérification


if __name__ == "__main__":
    print("=== TEST FETCH SMALL SAMPLE ===")
    test_fetch_small_sample()
    print("\n=== TEST LOAD AND ADAPT ===")
    test_load_and_adapt()