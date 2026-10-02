# Templates d'annotation Label Studio

Configurations d'annotation (`label_config` XML) des deux projets Label Studio du
pipeline Biolit.

- [`crops.xml`](crops.xml) — projet **Biolit Crops** : affiche la **proposition du
  modèle IA** (espèce, score de confiance, taxonomie, détection YOLO, lien DORIS,
  carte, lieu) et demande à l'annotateur de la **valider** (*Prédiction correcte /
  Corriger l'espèce / Non identifiable*).
- [`nocrops.xml`](nocrops.xml) — projet **Biolit No Crops** : images sans crop détecté,
  l'annotateur **entoure chaque espèce** (rectangles) et la **nomme**, ou coche
  « aucune espèce identifiable ».

## Pourquoi ces fichiers existent

Le `label_config` d'un projet Label Studio vit **uniquement dans la base de données
LS**, pas dans le code. Il n'est **pas recréé automatiquement** au déploiement (comme
la config *Cloud Storage S3* et le CORS du bucket). Si les projets LS sont recréés ou
la base réinitialisée, ces templates sont **perdus** : on les garde ici pour pouvoir
les **ré-appliquer**.

## ⚠️ À ne pas casser : cohérence avec le code d'extraction

Les **noms de contrôles** et les **valeurs de choix** doivent correspondre EXACTEMENT à
ce que lit [`backend/biolit/label_studio.py`](../../backend/biolit/label_studio.py)
(`extract_crops_data_from_label_studio` / `extract_no_crops_data_from_label_studio`),
sinon le remplissage de `db_finale` casse silencieusement.

| Projet   | Contrôles attendus                          | Valeurs de choix (accentuées !)                                  |
|----------|---------------------------------------------|------------------------------------------------------------------|
| Crops    | `decision`, `espece_corrigee`, `commentaire`| `Prédiction correcte`, `Corriger l'espèce`, `Non identifiable`    |
| No Crops | `presence`, `nom_espece`, `commentaire`     | `Aucune espece identifiable`                                      |

Les accents des valeurs `decision` sont significatifs (comparaison de chaînes dans le
code) — garder l'encodage **UTF-8**.

## Comment (ré)appliquer

Via l'API (projets Crops = id 1, No Crops = id 2, à adapter). Nécessite l'URL LS et un
token API (cf. `LABEL_STUDIO_URL` / `LABEL_STUDIO_API_KEY` de l'app d'ingestion) :

```bash
export LS_URL="https://<label-studio>"        # sans slash final
export LS_TOKEN="<token API Label Studio>"
```

```python
import json, os, urllib.request
LS, TOK = os.environ["LS_URL"], os.environ["LS_TOKEN"]
H = {"Authorization": "Token " + TOK, "Content-Type": "application/json"}
for pid, path in ((1, "crops.xml"), (2, "nocrops.xml")):
    cfg = open(path, encoding="utf-8").read()
    r = urllib.request.Request(f"{LS}/api/projects/{pid}",
        data=json.dumps({"label_config": cfg}).encode("utf-8"), headers=H, method="PATCH")
    urllib.request.urlopen(r, timeout=40)
    print(f"projet {pid}: config appliquee")
```

Alternative : copier-coller le XML dans *Project → Settings → Labeling Interface → Code*.

## Champs de données disponibles dans les tâches

Poussés par `push_tasks_label_studio_crops` / `push_tasks_label_studio_no_crops` :

- **Crops** : `image`, `id_observation`, `id_crops`, `species_name`, `best_label`,
  `best_level`, `best_score`, `regne_yolo`, `confiance_yolo`, `regne`, `phylum`,
  `classe`, `ordre`, `famille`, `region`, `commune`, `departement`, `latitude`,
  `longitude`, `geo_map_html`, `lien_doris`, `lien_doris_html`.
- **No Crops** : `image`, `id_observation`, `site`, `region`, `commune`, `departement`,
  `latitude`, `longitude`, `geo_map_html`.

## Pistes d'évolution

- **Pré-annotation** : afficher la prédiction n'est pas la pré-cocher. Pour que
  « Prédiction correcte » soit pré-sélectionnée, le pipeline doit pousser un objet
  `predictions` (et pas seulement `data`) à l'import des tâches.
