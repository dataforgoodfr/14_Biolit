from pathlib import Path

import matplotlib as mpl

ROOTDIR = Path(__file__).parent / ".."
DATADIR = ROOTDIR / "data"
RAWDIR = DATADIR / "raw"
EXPORTDIR = DATADIR / "exports"
CONFIGDIR = Path(__file__) / "config"


mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
