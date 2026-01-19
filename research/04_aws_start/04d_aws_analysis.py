from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# Minimal, hard-coded configuration. Change these for each run.
DATE_TAG = "20260119-095902"  # e.g., folder name under data/
FILENAME = "measurement_counts.json"


def main():
    base_dir = Path(__file__).resolve().parent
    in_path = base_dir / "data" / DATE_TAG / FILENAME

    # Load JSON payload (assume a dict); minimal branching
    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Accept either direct mapping or nested under "measurement_counts"
    counts = payload.get("measurement_counts", payload)

    # Infer qubit count and construct ordered basis
    n = max(len(k) for k in counts.keys())
    order = [format(i, f"0{n}b") for i in range(2**n)]

    total = sum(int(counts.get(bs, 0)) for bs in order)
    probs = np.array([int(counts.get(bs, 0)) / total for bs in order], dtype=float)

    # Print the normalized vector (probabilities sum to 1)
    np.set_printoptions(precision=6, suppress=True)
    print("state vector (normalized counts):")
    print(probs)


if __name__ == "__main__":
    main()
