import argparse
from collections import Counter
from pathlib import Path

import cv2
import yaml

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MIN_SIZE = 100  # pixels — en dessous c'est inutilisable


def inspect(img_dir: str):
    path = Path(img_dir)
    images = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMG_EXT]

    print(f"\n── Inspection : {path} ({len(images)} images) ──\n")

    corrupted = []
    too_small = []
    widths, heights = [], []
    species = Counter()

    for f in images:
        img = cv2.imread(str(f))
        if img is None:
            corrupted.append(f.name)
            continue

        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

        if w < MIN_SIZE or h < MIN_SIZE:
            too_small.append(f.name)

        # Le format de nom est "{id}_{espece}_{idx}.ext" (cf. build_dataset.py)
        parts = f.stem.split("_")
        if len(parts) >= 2:
            species[parts[1]] += 1

    ok = len(images) - len(corrupted)
    print(f"OK : {ok}")
    print(f"Corrompues : {len(corrupted)}")
    print(f"Trop petites : {len(too_small)}  (< {MIN_SIZE}px)")

    if widths:
        print(f"\nRésolution min : {min(widths)} x {min(heights)}")
        print(f"Résolution max : {max(widths)} x {max(heights)}")
        med_w = sorted(widths)[len(widths) // 2]
        med_h = sorted(heights)[len(heights) // 2]
        print(f"Résolution médiane : {med_w} x {med_h}")

    if species:
        print(f"\nEspèces ({len(species)}) :")
        for name, count in species.most_common():
            print(f"{name:<30} {count} image(s)")

    if corrupted:
        print("\nFichiers corrompus :")
        for name in corrupted:
            print(f" - {name}")

    if too_small:
        print("\nFichiers trop petits :")
        for name in too_small:
            print(f" - {name}")

    print()

    if corrupted or too_small:
        print(f"{len(corrupted)} corrompue(s), {len(too_small)} trop petite(s)")
        print("Lance build_dataset.py pour nettoyer, ou supprime-les manuellement.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/autodistill_boostrap.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    inspect(cfg["img_path"])


if __name__ == "__main__":
    main()
