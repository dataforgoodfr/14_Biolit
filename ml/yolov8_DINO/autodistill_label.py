"""
Pseudo-labeling GroundingDINO.
Lance après build_dataset.py + inspect_data.py.

Usage:
    python autodistill_label.py
    python autodistill_label.py --config configs/autodistill_boostrap.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# compat —> ajustement version
torch.use_deterministic_algorithms(False)
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **kw, weights_only=False)

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO

sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import get_logger

log = get_logger("autodistill_label")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/autodistill_boostrap.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log.info("Labeling GroundingDINO → %s", cfg["labeled_output"])

    model = GroundingDINO(ontology=CaptionOntology(cfg["ontology"]))
    model.label(
        input_folder=cfg["img_path"],
        output_folder=cfg["labeled_output"],
        extension="",
    )

    log.info("Labels générés → %s", cfg["labeled_output"])


if __name__ == "__main__":
    main()
