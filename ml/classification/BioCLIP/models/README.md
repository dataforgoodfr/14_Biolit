# Modèles pré-entraînés

Ce dossier contient les deux modèles nécessaires à l'inférence du système hybride v4.

---

## Fichiers

### `best_model_top50.pth`
Classifier MLP entraîné sur les **50 espèces les plus représentées** dans BioLit (>50 images chacune).

- **Architecture** : LayerNorm → Linear(512→512) → GELU → Dropout(0.5) → Linear(512→256) → GELU → Dropout(0.25) → Linear(256→50)
- **Entrée** : features BioCLIP 512d (normalisées L2)
- **Sortie** : logits sur 50 classes
- **Accuracy** : 89.7% sur les espèces communes
- **Entraîné sur** : GPU T4, dataset BioLit top100 (split 70/15/15)

---

### `prototypes_v4.pt`
Prototypes Proto-CLIP pour les **100 espèces** (communes + rares), avec transformation whitening ..

- **Prototypes visuels** : moyennes pondérées des features BioCLIP par espèce, 3 itérations de raffinement (arXiv:2110.11553)
- **Fusion Proto-CLIP** : α=0.8 visuel + 0.2 textuel (arXiv:2307.03073)
- **Espace** : features whitened 256d après PCA 
- **Température** : T*=23.8, apprise par gradient (arXiv:2108.00340)
- **Accuracy rares** : 51.1%

**Clés contenues dans le `.pt` :**

| Clé | Type | Description |
|---|---|---|
| `prototypes` | Tensor [100, 256] | Prototypes Proto-CLIP fusionnés |
| `temperature` | float | Température T* apprise |
| `idx_to_species` | dict | Index → nom espèce |
| `species_to_idx` | dict | Nom espèce → index |
| `whitening_mu` | numpy [1, 512] | Moyenne de centrage (whitening) |
| `whitening_W` | numpy [512, 256] | Matrice de projection PCA |
| `pca_dim` | int | Dimension cible (256) |
| `alpha` | float | Coefficient fusion visuel/textuel (0.8) |
| `seuil_top50` | float | Seuil confiance classifier (0.80) |
| `seuil_proto` | float | Seuil confiance prototypes (0.40) |

> **Important** : `whitening_mu` et `whitening_W` sont indispensables à l'inférence. Toute nouvelle image doit être projetée dans le même espace 256d avant comparaison aux prototypes. Le script `infer_local_v4.py` gère cela automatiquement.

---

## Chargement manuel

```python
import torch

# Classifier
ckpt = torch.load("best_model_top50.pth", map_location="cpu")
# Clés : classifier_state_dict, idx_to_species, species_to_idx, num_species, feature_dim

# Prototypes
pt = torch.load("prototypes_v4.pt", map_location="cpu")
prototypes  = pt["prototypes"]       # [100, 256]
mu          = pt["whitening_mu"]     # numpy [1, 512]
W           = pt["whitening_W"]      # numpy [512, 256]
T           = pt["temperature"]      # float
```
