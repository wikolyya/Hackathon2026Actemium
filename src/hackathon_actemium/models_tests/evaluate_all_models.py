from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    # Allows direct execution from this folder without installing the package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from hackathon_actemium.models_tests.main_compare import main
else:
    from .main_compare import main


if __name__ == "__main__":
    main()
