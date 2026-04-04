"""
Benchmark de prompts textuels pour Grounding DINO

Ce script teste différents textes de recherche (prompts) sur un lot d'images
pour trouver lequel détecte le mieux les organismes marins du projet BioLit.

Utilisation :
    python benchmark_gdino_prompts.py --n-images 100 --output results/gdino_benchmark
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


# ── 1. CONFIGURATION GÉNÉRALE ─────────────────────────────────────────────────
# Chemins vers les données
IMAGES_DIR = Path("dataset_biolit/images/identifiable")

# GPU si disponible, sinon CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modèle à utiliser (tiny = plus rapide, base = plus précis)
MODEL_ID = "IDEA-Research/grounding-dino-base"

# Seuil de confiance : on ignore les détections en-dessous de 30%
THRESHOLD = 0.3


# ── 2. PROMPTS À TESTER ───────────────────────────────────────────────────────
# Chaque entrée : nom_du_prompt → texte envoyé au modèle
PROMPTS = {
    "simple":            "animal, plant",

}


# ── 3. CHARGEMENT DU MODÈLE ───────────────────────────────────────────────────
def charger_modele():
    """Charge le modèle Grounding DINO et son processeur d'images."""
    print(f"Chargement du modèle sur {DEVICE}...")
    processeur = AutoProcessor.from_pretrained(MODEL_ID)
    modele = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(MODEL_ID)
        .eval()           # mode évaluation : désactive le dropout
        .to(DEVICE)       # envoie le modèle sur GPU ou CPU
    )
    print("Modèle prêt !\n")
    return modele, processeur


# ── 4. DÉTECTION SUR UNE IMAGE ────────────────────────────────────────────────
@torch.inference_mode()   # désactive le calcul du gradient → plus rapide
def detecter(image, prompt, modele, processeur):
    """
    Applique Grounding DINO sur une image avec un prompt textuel.
    Retourne un dict avec les clés : 'scores', 'labels', 'boxes'.
    """
    # Grounding DINO attend les labels séparés par des points
    # Ex : "crab, starfish" → "crab. starfish."
    labels = [lb.strip() for lb in prompt.split(",") if lb.strip()]
    texte_formate = ". ".join(labels) + "."

    # Préparer les entrées du modèle
    entrees = processeur(
        images=image,
        text=texte_formate,
        return_tensors="pt"
    ).to(DEVICE)

    # Inférence
    sorties = modele(**entrees)

    # Convertir les sorties en boîtes de détection lisibles
    resultats = processeur.post_process_grounded_object_detection(
        sorties,
        input_ids=entrees["input_ids"],
        target_sizes=[(image.height, image.width)],
        threshold=THRESHOLD,
        text_threshold=THRESHOLD,
    )[0]

    return resultats


# ── 5. SAUVEGARDER LES CROPS ──────────────────────────────────────────────────
def sauvegarder_crops(image, resultats, nom_image, dossier_sortie):
    """
    Découpe et sauvegarde chaque zone détectée (crop) dans un fichier .webp.
    Nom du fichier : nomimage_crop0_label_score.webp
    """
    for i, (score, label, boite) in enumerate(
        zip(resultats["scores"], resultats["labels"], resultats["boxes"])
    ):
        x1, y1, x2, y2 = map(int, boite.tolist())
        crop = image.crop((x1, y1, x2, y2))
        nom_fichier = f"{nom_image}_crop{i}_{label}_{score:.2f}.webp"
        crop.save(dossier_sortie / nom_fichier, "WEBP", quality=90)


# ── 6. TESTER UN PROMPT SUR TOUTES LES IMAGES ─────────────────────────────────
def tester_prompt(nom_prompt, texte_prompt, images, modele, processeur, dossier_crops):
    """
    Applique un prompt sur toutes les images.
    Retourne une liste de dicts (une entrée par détection trouvée).
    """
    resultats_prompt = []

    # Créer un sous-dossier pour les crops de ce prompt
    dossier_prompt = dossier_crops / nom_prompt
    dossier_prompt.mkdir(exist_ok=True)

    for chemin_image, image in tqdm(images, desc=f"  {nom_prompt}", leave=False):
        try:
            resultats = detecter(image, texte_prompt, modele, processeur)

            # Sauvegarder les crops si on a trouvé quelque chose
            if len(resultats["scores"]) > 0:
                sauvegarder_crops(image, resultats, chemin_image.stem, dossier_prompt)

            # Enregistrer chaque détection dans notre tableau de résultats
            for score, label, boite in zip(
                resultats["scores"], resultats["labels"], resultats["boxes"]
            ):
                resultats_prompt.append({
                    "prompt":       nom_prompt,
                    "texte_prompt": texte_prompt,
                    "image":        chemin_image.name,
                    "label":        label,
                    "confiance":    float(score),
                    "x1": float(boite[0]), "y1": float(boite[1]),
                    "x2": float(boite[2]), "y2": float(boite[3]),
                })

        except Exception as e:
            print(f"  ⚠ Erreur sur {chemin_image.name} : {e}")

    # Afficher un résumé pour ce prompt
    n_images_avec_detection = len({r["image"] for r in resultats_prompt})
    print(f"  → {len(resultats_prompt)} détections sur {n_images_avec_detection}/{len(images)} images")

    return resultats_prompt


# ── 7. GÉNÉRER LES GRAPHIQUES ─────────────────────────────────────────────────
def generer_violon(df_detections, stats, dossier_viz, timestamp):
    """
    Violin plot de la distribution des scores de confiance pour les top 5 prompts.
    Le violon montre la forme de la distribution : large = beaucoup de valeurs a cette hauteur.
    La ligne centrale = mediane, le point = moyenne.
    """
    print("  -> Generation du violin plot...")

    try:
        # Selectionner les 5 prompts avec le plus de detections
        top5 = stats.sort_values("n_detections", ascending=False).head(5)["prompt"].tolist()

        # Garder seulement les prompts avec au moins 2 detections
        # (matplotlib ne peut pas dessiner un violon avec un seul point)
        donnees = []
        labels_valides = []
        for p in top5:
            valeurs = df_detections[df_detections["prompt"] == p]["confiance"].values
            if len(valeurs) >= 2:
                donnees.append(valeurs)
                labels_valides.append(p)
            else:
                print(f"    Prompt {p!r} ignore : seulement {len(valeurs)} detection(s), minimum 2 requis")

        if len(donnees) == 0:
            print("  Violin plot ignore : aucun prompt n a assez de detections (min 2)")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        parts = ax.violinplot(
            donnees,
            positions=range(len(labels_valides)),
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )

        # Couleurs distinctes pour chaque violon
        couleurs = plt.cm.Set3([i / max(len(labels_valides), 1) for i in range(len(labels_valides))])
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(couleurs[i])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(labels_valides)))
        ax.set_xticklabels(labels_valides, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Score de confiance", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_title("Distribution des scores de confiance - Top 5 des prompts",
                     fontsize=14, fontweight="bold", pad=20)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        chemin = dossier_viz / f"05_violon_confiance_{timestamp}.png"
        fig.savefig(chemin, dpi=150)
        plt.close()
        print(f"  OK Violin plot -> {chemin.name}")

    except Exception as e:
        print(f"  ERREUR violin plot : {e}")
        plt.close()


def generer_graphiques(stats, dossier_viz, timestamp, df_detections=None):
    """Crée 5 graphiques de comparaison et les sauvegarde en PNG."""

    # -- Graphique 1 : nombre total de détections par prompt
    fig, ax = plt.subplots(figsize=(14, 8))
    stats_tri = stats.sort_values("n_detections")
    ax.barh(stats_tri["prompt"], stats_tri["n_detections"], color="steelblue")
    ax.set_xlabel("Nombre de détections")
    ax.set_title("Détections totales par prompt")
    plt.tight_layout()
    fig.savefig(dossier_viz / f"01_detections_{timestamp}.png", dpi=150)
    plt.close()

    # -- Graphique 2 : confiance moyenne par prompt
    fig, ax = plt.subplots(figsize=(14, 8))
    stats_tri = stats.sort_values("confiance_moyenne")
    ax.barh(stats_tri["prompt"], stats_tri["confiance_moyenne"], color="darkorange")
    ax.set_xlabel("Confiance moyenne")
    ax.set_xlim(0, 1)
    ax.set_title("Confiance moyenne par prompt")
    plt.tight_layout()
    fig.savefig(dossier_viz / f"02_confiance_{timestamp}.png", dpi=150)
    plt.close()

    # -- Graphique 3 : images avec confiance ≥ 60%
    fig, ax = plt.subplots(figsize=(14, 8))
    stats_tri = stats.sort_values("pct_conf_60")
    ax.barh(stats_tri["prompt"], stats_tri["pct_conf_60"], color="seagreen")
    ax.set_xlabel("% d'images avec confiance ≥ 60%")
    ax.set_xlim(0, 100)
    ax.set_title("Images avec haute confiance (≥60%) par prompt")
    plt.tight_layout()
    fig.savefig(dossier_viz / f"03_haute_confiance_{timestamp}.png", dpi=150)
    plt.close()

    # -- Graphique 4 : heatmap des 3 métriques normalisées
    fig, ax = plt.subplots(figsize=(10, 9))
    stats_tri = stats.sort_values("n_detections", ascending=False)

    # Normaliser chaque colonne entre 0 et 1
    matrice = stats_tri[["n_detections", "confiance_moyenne", "pct_conf_60"]].copy()
    matrice["n_detections"]    = matrice["n_detections"] / matrice["n_detections"].max()
    matrice["pct_conf_60"]     = matrice["pct_conf_60"] / 100

    im = ax.imshow(matrice.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Détections", "Confiance moy.", "Conf. ≥60%"], fontweight="bold")
    ax.set_yticks(range(len(stats_tri)))
    ax.set_yticklabels(stats_tri["prompt"].values, fontsize=9)
    for i in range(len(stats_tri)):
        for j in range(3):
            ax.text(j, i, f"{matrice.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Score normalisé (0–1)")
    ax.set_title("Heatmap de performance des prompts", fontweight="bold")
    plt.tight_layout()
    fig.savefig(dossier_viz / f"04_heatmap_{timestamp}.png", dpi=150)
    plt.close()

    # -- Graphique 5 : violin plot top 5 prompts
    if df_detections is not None and not df_detections.empty:
        generer_violon(df_detections, stats, dossier_viz, timestamp)

    print(f"✓ Graphiques sauvegardés dans : {dossier_viz}")


# ── 8. PROGRAMME PRINCIPAL ────────────────────────────────────────────────────
def main():
    # Lire les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Benchmark prompts Grounding DINO")
    parser.add_argument("--n-images", type=int,  default=100,                      help="Nombre d'images à tester")
    parser.add_argument("--output",   type=str,  default="results/gdino_benchmark", help="Dossier de sortie")
    args = parser.parse_args()

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dossier     = Path(args.output)
    dossier_crops = dossier / "crops"
    dossier_viz   = dossier / "visualisations"
    dossier_crops.mkdir(parents=True, exist_ok=True)
    dossier_viz.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE} | Modèle : {MODEL_ID}")
    print(f"Images : {args.n_images} | Seuil : {THRESHOLD}\n")

    # ── Étape A : charger les images ──────────────────────────────────────────
    fichiers = (
        list(IMAGES_DIR.glob("*.webp")) +
        list(IMAGES_DIR.glob("*.jpg"))  +
        list(IMAGES_DIR.glob("*.jpeg")) +
        list(IMAGES_DIR.glob("*.png"))
    )
    random.seed(42)  # seed fixe → même sélection à chaque run
    fichiers = random.sample(fichiers, min(args.n_images, len(fichiers)))

    images = []
    for chemin in fichiers:
        try:
            images.append((chemin, Image.open(chemin).convert("RGB")))
        except Exception as e:
            print(f"⚠ Impossible de charger {chemin.name} : {e}")
    print(f"{len(images)} images chargées.\n")

    # ── Étape B : charger le modèle ───────────────────────────────────────────
    modele, processeur = charger_modele()

    # ── Étape C : tester tous les prompts ────────────────────────────────────
    toutes_detections = []

    for nom_prompt, texte_prompt in PROMPTS.items():
        print(f"▶ Prompt : {nom_prompt}")
        detections = tester_prompt(
            nom_prompt, texte_prompt, images, modele, processeur, dossier_crops
        )
        toutes_detections.extend(detections)

    # ── Étape D : sauvegarder les résultats bruts ─────────────────────────────
    df = pd.DataFrame(toutes_detections)
    chemin_csv = dossier / f"resultats_bruts_{timestamp}.csv"
    df.to_csv(chemin_csv, index=False)
    print(f"\n✓ Résultats bruts → {chemin_csv}")

    # ── Étape E : calculer les statistiques par prompt ────────────────────────
    if df.empty:
        print("Aucune détection. Arrêt.")
        return

    stats = df.groupby("prompt").agg(
        n_detections     = ("image", "count"),
        n_images_detectees = ("image", "nunique"),
        confiance_moyenne  = ("confiance", "mean"),
        confiance_mediane  = ("confiance", "median"),
    ).reset_index()

    # Ajouter : % d'images avec au moins une détection ≥ 60%
    pct_list = []
    for prompt in stats["prompt"]:
        df_p = df[df["prompt"] == prompt]
        images_haute_conf = df_p[df_p["confiance"] >= 0.6]["image"].nunique()
        total             = df_p["image"].nunique()
        pct_list.append(images_haute_conf / total * 100 if total > 0 else 0)
    stats["pct_conf_60"] = pct_list

    stats = stats.sort_values("n_detections", ascending=False)

    chemin_stats = dossier / f"stats_par_prompt_{timestamp}.csv"
    stats.to_csv(chemin_stats, index=False)
    print(f"✓ Statistiques → {chemin_stats}")

    # Afficher le podium dans le terminal
    print("\n🏆 TOP 3 DES PROMPTS :")
    for i, (_, row) in enumerate(stats.head(3).iterrows(), 1):
        print(f"  {i}. {row['prompt']}")
        print(f"     {int(row['n_detections'])} détections | "
              f"confiance moy. {row['confiance_moyenne']:.3f} | "
              f"{row['pct_conf_60']:.1f}% images conf≥60%")

    # ── Étape F : générer les graphiques ─────────────────────────────────────
    generer_graphiques(stats, dossier_viz, timestamp, df_detections=df)

    print(f"\n✅ Terminé ! Tout est dans : {dossier}")


if __name__ == "__main__":
    main()

