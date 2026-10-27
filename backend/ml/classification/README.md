# Classification taxonomique

Module appelé par `pipelines/run.py` pour classifier les crops produits par
`ml/crop_inference` : `flow_ml_classification(crops_images, df_crops)`.

## Méthode (v3 « fusion hiérarchique »)

```text
crop ──BioCLIP-2 gelé (ONNX)──▶ vecteur 768d
     ──têtes (ONNX)──▶ P(espèce) = w × régression logistique + (1 − w) × prototypes
                       P(famille), P(ordre)… = somme des P(espèce) du taxon
     ──décision──▶ niveau le plus fin dont la probabilité atteint son seuil
```

- **Prototypes** : un vecteur par espèce (≥ 3 photos) = moyenne visuelle + texte BioCLIP-2.
- **Régression logistique** : multinomiale sur les vecteurs 768d standardisés.
- **Décision** : espèce si P ≥ seuil espèce, sinon famille si P ≥ seuil famille, etc.
  Les colonnes renvoyées sont cohérentes : niveaux supérieurs issus de TAXREF, niveaux plus
  fins que `best_level` à `None`.

Seuils, poids de fusion et températures sont choisis à l'entraînement sur un jeu de
validation, sous contrainte de ne jamais faire moins bien que la méthode v2 ; ils sont stockés
dans `classifier_meta.json`. L'inférence n'utilise que **onnxruntime + numpy + PIL**.

## Fichiers

```text
classifier_bioclip.py      # prétraitement numpy + extracteur BioCLIP-2 (ONNX)
classifier_infer.py        # chargement du modèle, têtes ONNX, décision hiérarchique
config.py                  # chemins, fichiers du modèle, hyper-paramètres
pipeline_classification.py # orchestration appelée par le pipeline principal
tests/                     # tests unitaires et d'intégration
```

## Fichiers du modèle (Hugging Face `DataForGood/biolit-species-identification`, dossier `classification/`)

| Fichier | Contenu |
|---|---|
| `bioclip2_visual.onnx` | encodeur visuel BioCLIP-2 FP32 (image → 768d), 1,2 Go |
| `classifier_heads.onnx` | prototypes + régression logistique + fusion + agrégation par niveau |
| `classifier_meta.json` | classes par niveau, taxonomie, seuils, paramètres de fusion |

`CLASSIF_MODEL_SOURCE=local` charge ces fichiers depuis `results/` au lieu de Hugging Face.

## Ré-entraînement

Le code d'entraînement (reconstruction du dataset, export ONNX, entraînement, évaluation,
référence v2) n'est pas inclus ici pour alléger le module d'inférence. Il est conservé sur
la branche `feat/ml-classification-training`.

## Tests

```bash
uv run pytest ml/classification/tests                           # rapides (CI)
RUN_ONNX_REGRESSION=1 uv run pytest ml/classification/tests -vv  # + intégration (modèle requis)
```
