# ML — Détection et crop d'espèces marines (BioLit)

## C'est quoi l'objectif ?

On veut automatiser la détection des espèces marines (animaux et végétaux) dans les photos BioLit.
Concrètement : le programme regarde une photo, trouve où se trouve l'espèce, et **découpe cette zone** (= le crop).

Ce crop servira ensuite à :
- classifier automatiquement l'espèce si la détection est fiable
- envoyer l'image dans Label Studio pour annotation manuelle si la détection est douteuse
- ignorer l'image si aucune espèce n'est trouvée

---

## Ce que fait chaque fichier

### `build_dataset.py` — Construire la base d'images

Ce script part du fichier `export_biolit.csv` (l'export brut de la plateforme BioLit) et prépare toutes les images pour le travail suivant.

Il fait 4 choses dans l'ordre :
1. Lit le CSV et garde seulement les observations **validées**
2. Télécharge les photos depuis internet
3. Les range dans des dossiers selon qu'elles sont **identifiables ou non**
4. Sauvegarde un fichier de métadonnées (`metadata.csv`) et un fichier de config pour YOLO (`data.yaml`)

Structure créée automatiquement :

```
dataset_biolit/
├── images/
│   ├── identifiable/        ← photos d'espèces qu'on peut identifier
│   └── non_identifiable/    ← photos floues ou ambiguës (à valider)
├── labels/
│   ├── identifiable/        ← réservé aux annotations YOLO
│   └── non_identifiable/
├── metadata.csv             ← tableau avec toutes les infos par image
└── data.yaml                ← config pour entraîner un modèle YOLO
```

> ⚠️ Ce script doit être exécuté **en premier**, avant tout le reste.

---

### `crop_pipeline.py` — Premier test de détection

Ce script est le **prototype** : il teste Grounding DINO sur une seule image pour vérifier que le modèle fonctionne et que les crops sont corrects.

Il utilise le prompt `"animal, plant"` sur une image de test et sauvegarde le crop découpé.

C'est le point de départ qui a servi de base pour construire le benchmark.

---

### `benchmark_gdino_simplifie.py` — Trouver le meilleur prompt ← **nouveau**

C'est le script principal de cette étape.

#### Pourquoi un benchmark de prompts ?

Grounding DINO est un modèle **zéro-shot** : on lui dit en texte ce qu'on cherche, et il détecte.
Par exemple : `"crab, starfish, sea urchin"`.

Le problème : on ne sait pas quel texte donne les **meilleures détections** sur nos photos BioLit.
Ce benchmark teste automatiquement **19 variantes de prompts** sur 100 images et compare les résultats.

#### Ce qu'il fait, étape par étape

1. **Charge 100 images** au hasard depuis `dataset_biolit/images/identifiable/`
2. **Charge le modèle** Grounding DINO (téléchargé automatiquement la première fois, ~700 MB)
3. **Pour chaque prompt**, applique la détection sur toutes les images et enregistre :
   - combien de détections ont été trouvées
   - le score de confiance de chaque détection
   - les coordonnées de chaque boîte (bbox)
   - les crops découpés (sauvegardés en `.webp`)
4. **Calcule des statistiques** par prompt : nombre de détections, confiance moyenne, % d'images avec confiance ≥ 60%
5. **Génère 5 graphiques** pour comparer visuellement les prompts
6. **Sauvegarde tout** : CSV brut, CSV de stats, et les graphiques

#### Les 19 prompts testés

| Nom du prompt | Texte envoyé au modèle |
|---|---|
| `simple` | `animal, plant` |
| `marin` | `marine animal, marine plant` |
| `sous_marin` | `underwater animal, underwater plant, marine organism` |
| `animaux_biolit` | `crab, starfish, sea urchin, anemone, shellfish...` |
| `plantes_biolit` | `seaweed, algae, brown algae, red algae...` |
| `equilibre` | `crab, starfish, sea urchin, anemone, seaweed...` |
| `morphologique` | `animal with shell, animal with tentacles...` |
| `ultra_detaille` | liste complète des espèces BioLit connues |
| `generique` | `animal, plant, marine organism, shellfish...` |
| `taxonomique` | `crustacean, mollusk, echinoderm, cnidarian...` |
| `crustaces` | `crab, lobster, shrimp, barnacle, mussel...` |
| `echinodermes` | `starfish, sea star, sea urchin, brittle star...` |
| `cnidaires` | `sea anemone, anemone, jellyfish...` |
| `algues_couleur` | `brown seaweed, red seaweed, green seaweed...` |
| `intertidal` | `intertidal animal, intertidal plant, tide pool creature...` |
| `visuel` | `spiny animal, shelled animal, tentacled creature...` |
| `top_combo` | `crab, starfish, sea urchin, mussel, barnacle, brown algae...` |
| `sessiles` | `barnacle, mussel, oyster, sea anemone...` |
| `mobiles` | `crab, starfish, sea urchin, snail, whelk...` |

#### Les graphiques produits

| Fichier | Ce qu'il montre |
|---|---|
| `01_detections.png` | Nombre total de détections par prompt |
| `02_confiance.png` | Score de confiance moyen par prompt |
| `03_haute_confiance.png` | % d'images avec confiance ≥ 60% |
| `04_heatmap.png` | Vue d'ensemble des 3 métriques normalisées |
| `05_violon_confiance.png` | Distribution des scores pour les 5 meilleurs prompts |

#### Utilisation

```bash
# Test rapide (10 images)
python benchmark_gdino_prompts.py --n-images 10 --output results/gdino_benchmark

# Benchmark complet (100 images)
python benchmark_gdino_prompts.py --n-images 100 --output results/gdino_benchmark
```

#### Sorties

```
results/gdino_benchmark/
├── crops/
│   ├── simple/              ← crops découpés par prompt
│   ├── marin/
│   └── ...
├── visualisations/
│   ├── 01_detections.png
│   ├── 02_confiance.png
│   ├── 03_haute_confiance.png
│   ├── 04_heatmap.png
│   └── 05_violon_confiance.png
├── resultats_bruts_YYYYMMDD.csv    ← une ligne par détection
└── stats_par_prompt_YYYYMMDD.csv   ← une ligne par prompt
```

---

## Schéma du pipeline complet

```
export_biolit.csv
       │
       ▼
build_dataset.py        → télécharge et range les images
       │
       ▼
dataset_biolit/images/identifiable/
       │
       ▼
benchmark_gdino_prompts.py   → teste 19 prompts, trouve le meilleur
       │
       ▼
   résultats + graphiques
       │
       ├── confiance forte  →  Classification automatique
       ├── confiance faible →  Label Studio (annotation manuelle)
       └── aucune détection →  stop
```

---

## Dépendances Python

```bash
pip install torch transformers pandas matplotlib Pillow tqdm
```

> Le modèle Grounding DINO est téléchargé automatiquement par Hugging Face lors du premier lancement (~700 MB).
