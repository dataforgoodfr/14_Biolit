"""
Tests unitaires de la décision hiérarchique (rapides, sans modèle).
"""

import numpy as np
import pytest

from ml.classification.classifier_infer import decide

TAX = {
    "patella vulgata":  {"regne": "Animalia", "phylum": "Mollusca", "classe": "Gastropoda",
                         "ordre": "Patellida", "famille": "Patellidae"},
    "patella depressa": {"regne": "Animalia", "phylum": "Mollusca", "classe": "Gastropoda",
                         "ordre": "Patellida", "famille": "Patellidae"},
    "carcinus maenas":  {"regne": "Animalia", "phylum": "Arthropoda", "classe": "Malacostraca",
                         "ordre": "Decapoda", "famille": "Carcinidae"},
}
META = {
    "classes": {
        "species_name": ["carcinus maenas", "patella depressa", "patella vulgata"],
        "famille": ["Carcinidae", "Patellidae"],
        "ordre": ["Decapoda", "Patellida"],
        "classe": ["Gastropoda", "Malacostraca"],
        "phylum": ["Arthropoda", "Mollusca"],
        "regne": ["Animalia"],
    },
    "taxonomy": TAX,
    "taxon_parents": {
        "regne": {"Animalia": {}},
        "phylum": {"Mollusca": {"regne": "Animalia"}, "Arthropoda": {"regne": "Animalia"}},
        "classe": {"Gastropoda": {"regne": "Animalia", "phylum": "Mollusca"},
                   "Malacostraca": {"regne": "Animalia", "phylum": "Arthropoda"}},
        "ordre": {"Patellida": {"regne": "Animalia", "phylum": "Mollusca", "classe": "Gastropoda"},
                  "Decapoda": {"regne": "Animalia", "phylum": "Arthropoda", "classe": "Malacostraca"}},
        "famille": {"Patellidae": {"regne": "Animalia", "phylum": "Mollusca",
                                   "classe": "Gastropoda", "ordre": "Patellida"},
                    "Carcinidae": {"regne": "Animalia", "phylum": "Arthropoda",
                                   "classe": "Malacostraca", "ordre": "Decapoda"}},
    },
}
TH = {"species_name": 0.5, "famille": 0.6, "ordre": 0.6, "classe": 0.6, "phylum": 0.6, "regne": 0.95}


def probs(sp):
    """Probabilités d'espèces → probabilités agrégées par niveau (comme les têtes ONNX)."""
    sp = np.array(sp, dtype=np.float32)
    out = {"species_name": sp}
    for lvl, classes in META["classes"].items():
        if lvl == "species_name":
            continue
        out[lvl] = np.array([sum(p for s, p in zip(META["classes"]["species_name"], sp)
                                 if TAX[s][lvl] == c) for c in classes], dtype=np.float32)
    return out


def test_species_when_confident():
    r = decide(probs([0.1, 0.2, 0.7]), META, TH)
    assert (r["best_level"], r["best_label"], r["path"]) == ("species_name", "patella vulgata",
                                                             "hier_species_name")
    assert r["famille"] == "Patellidae" and r["regne"] == "Animalia"
    assert r["best_score"] == pytest.approx(0.7)


def test_family_when_species_split_between_sisters():
    # deux patelles à 0,45 : aucune espèce ≥ 0,5, mais la famille Patellidae est à 0,9
    r = decide(probs([0.1, 0.45, 0.45]), META, TH)
    assert (r["best_level"], r["best_label"]) == ("famille", "Patellidae")
    assert (r["regne"], r["phylum"], r["ordre"]) == ("Animalia", "Mollusca", "Patellida")
    assert r["species_name"] is None


def test_rejection_and_empty_taxonomy():
    th = {**TH, "famille": 0.99, "ordre": 0.99, "classe": 0.99, "phylum": 0.99, "regne": 1.01}
    r = decide(probs([0.3, 0.35, 0.35]), META, th)
    assert r["best_level"] is None and r["path"] == "hier_rejet"
    assert all(r[k] is None for k in ["regne", "phylum", "classe", "ordre", "famille", "species_name"])


def test_output_contract():
    r = decide(probs([0.1, 0.2, 0.7]), META, TH)
    for key in ["best_level", "best_label", "best_score", "margin", "path", "all_scores",
                "regne", "phylum", "classe", "ordre", "famille", "species_name"]:
        assert key in r
    assert len(r["all_scores"]["species_name"]) == 3
