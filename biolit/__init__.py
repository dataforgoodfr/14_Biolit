from pathlib import Path

import matplotlib as mpl

ROOTDIR = Path(__file__).parent / ".."
DATADIR = ROOTDIR / "data"
RAWDIR = DATADIR / "raw"
EXPORTDIR = DATADIR / "exports"

DATA_GOUV_CONTOUR_COMMUNES_URL = "https://www.data.gouv.fr/api/1/datasets/r/00c0c560-3ad1-4a62-9a29-c34c98c3701e"
DATA_GOUV_INFO_COMMUNES_URL = "https://www.data.gouv.fr/api/1/datasets/r/f5df602b-3800-44d7-b2df-fa40a0350325"
WORLD_COAST_LINES_URL = "https://osmdata.openstreetmap.de/download/coastlines-split-4326.zip"

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
