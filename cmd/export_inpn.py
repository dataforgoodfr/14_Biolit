import sys
from pathlib import Path

_base_dir = str(Path(__file__).parent.parent)
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

if True:
    from biolit.taxref import format_taxref


def main():
    format_taxref()


if __name__ == "__main__":
    main()
