# Résultats

Ce dossier contient les fichiers de sortie du système hybride v4 sur les données de test.

---

## Fichiers

### `predictions_sample_test.csv`
Prédictions du système hybride v4 sur **40 images terrain** (.webp), collectées sur le littoral français en conditions réelles après crop par Dino.

**Colonnes :**

| Colonne | Description |
|---|---|
| `image` | Nom du fichier image |
| `espece_pred` | Espèce prédite (`?` si rejetée) |
| `confiance` | Score de confiance de la prédiction retenue [0–1] |
| `methode` | Composante ayant produit la prédiction finale |
| `top1_common` | 1ère prédiction du classifier top50 |
| `conf_common` | Confiance associée |
| `top2_common` | 2ème alternative classifier |
| `top3_common` | 3ème alternative classifier |
| `top1_proto` | 1ère prédiction des prototypes |
| `conf_proto` | Confiance associée |
| `top2_proto` | 2ème alternative prototypes |
| `top3_proto` | 3ème alternative prototypes |

**Valeurs possibles de `methode` :**

| Valeur | Signification |
|---|---|
| `classifier_top50` | Classifier confiant (conf ≥ 0.80) |
| `classifier_top50_low` | Classifier accepté à confiance modérée (conf ≥ 0.40) |
| `prototypical_rare` | Prototype confiant sur une espèce rare (conf ≥ 0.40) |
| `rejected` | Les deux composantes sous le seuil — image non classifiée |

**Résultats sur ces 40 images :**

| Méthode | N | % | Conf. moy. |
|---|---|---|---|
| `classifier_top50` | 10 | 25.0% | 0.92 |
| `classifier_top50_low` | 12 | 30.0% | 0.62 |
| `prototypical_rare` | 7 | 17.5% | 0.86 |
| `rejected` | 11 | 27.5% | — |

- **Taux acceptées** : 72.5%
- **Confiance moyenne (acceptées)** : 0.78

> L'écart de couverture avec le benchmark (88%) est attendu : les images terrain présentent des conditions de prise de vue très variables (éclairage, angle, fond, espèces partiellement visibles) non représentées dans les données d'entraînement.

---

### `export_biolit_enriched.csv`
Dataset BioLit enrichi via l'API GBIF avec la hiérarchie taxonomique complète de chaque espèce.

**Colonnes ajoutées par enrichissement GBIF :**

| Colonne | Description |
|---|---|
| `regne` | Règne (ex: Animalia, Plantae) |
| `embranchement` | Embranchement (ex: Arthropoda) |
| `classe` | Classe (ex: Malacostraca) |
| `ordre` | Ordre |
| `famille` | Famille |
| `genre` | Genre |

Ce fichier a servi à construire les descriptions textuelles utilisées dans les prototypes Proto-CLIP et à sélectionner les 100 espèces les plus représentées pour l'entraînement.
