import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def save_params(params: dict, path: Path = BASE_DIR / "artefacts" / "best_params.json"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # crée le dossier si besoin

    with open(path, "w") as f:
        json.dump(params, f, indent=4)


def load_params(path: Path = BASE_DIR / "artefacts" / "best_params.json") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with open(path, "r") as f:
        return json.load(f)


def save_metrics(metrics: dict, path: Path = BASE_DIR / "artefacts" / "metrics.json"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)