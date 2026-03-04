from pathlib import Path

import matplotlib as mpl

ROOTDIR = Path(__file__).parent / ".."
DATADIR = ROOTDIR / "data"
RAWDIR = DATADIR / "raw"
EXPORTDIR = DATADIR / "exports"

# /!\ URL emporaire de l'archive le temps que le museum corrige sa cybersec...
TAXREFURL = "https://assets.patrinat.fr/files/referentiel/TAXREF_v18_2025.zip"


mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
