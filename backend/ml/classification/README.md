# Classification taxonomique

Module utilise par `pipelines/run.py` pour classifier les crops produits par
`ml/crop_inference`.

## Fichiers

```text
classifier_bioclip.py      # extraction de features BioCLIP
classifier_infer_v2.py     # inference et fusion des predictions
classifier_mlp.py          # definition du MLP taxonomique
classifier_s3.py           # lecture/ecriture des assets via S3/Cellar
config.py                  # seuils, device, noms de fichiers modele
db.py                      # persistance des resultats en PostgreSQL
pipeline_classification.py # orchestration appelee par le pipeline principal
```

Les anciens scripts d'entrainement, samples et resultats experimentaux ne sont
pas inclus dans cette branche de production. Les poids/modeles necessaires au
runtime doivent etre recuperes depuis Hugging Face, S3/Cellar ou une source
externe configuree.

## Entree et sortie

Entree:

- liste de chemins/images de crops;
- dataframe des crops issu de `flow_ml_crops`.

Sortie:

- dataframe de predictions taxonomiques;
- donnees pretes a etre poussees vers Label Studio ou PostgreSQL.

## Lancement via Python

```python
from ml.classification.pipeline_classification import flow_ml_classification

df_taxonomy = flow_ml_classification(crops_images, df_crops)
```

Le module est concu pour tourner sur CPU dans Clever Cloud.
