from pathlib import Path

import matplotlib as mpl

ROOTDIR = Path(__file__).parent / ".."
DATADIR = ROOTDIR / "data"


mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
