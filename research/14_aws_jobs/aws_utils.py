import json
from datetime import datetime
from pathlib import Path

# Helper functions
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_text(path: Path, content: str) -> None:
    path.write_text(content)

def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))
