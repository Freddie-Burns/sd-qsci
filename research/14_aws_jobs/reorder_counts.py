from __future__ import annotations

import json
from pathlib import Path
from qiskit import qpy

def reorder_counts(counts: dict[str, int], run_dir: Path) -> dict[str, int]:
    """Reorder counts based on the transpiled circuit's layout."""
    tqc_path = run_dir / "qiskit_circuit_transpiled.qpy"
    if not tqc_path.exists():
        print(f"[warning] No transpiled circuit found at {tqc_path}. Skipping reordering.")
        return counts

    with open(tqc_path, 'rb') as f:
        qc = qpy.load(f)[0]
        layout = qc.layout

    if layout is None:
        print("[warning] No layout found in the transpiled circuit. Skipping reordering.")
        return counts

    n = max(len(k) for k in counts.keys())
    
    # Map physical qubits to virtual qubits
    index_map = []
    try:
        for i in range(n):
            index_map.append(layout.initial_layout._p2v[i]._index)
    except Exception as e:
        print(f"[warning] Error while mapping layout: {e}. Skipping reordering.")
        return counts

    reordered_counts = {}
    for bitstring, count in counts.items():
        # Qiskit bitstrings are often little-endian (qubit 0 at the right)
        # 04d script reverses it before mapping, then reverses back.
        bitstring_rev = bitstring[::-1] 
        bitstring_index_pairs = zip(bitstring_rev, index_map)
        sorted_pairs = sorted(bitstring_index_pairs, key=lambda x: x[1])
        new_bitstring = ''.join([b for b, i in sorted_pairs])
        new_bitstring = new_bitstring[::-1] 
        reordered_counts[new_bitstring] = count

    return reordered_counts

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reorder counts in run directories.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Path(s) to run directory/directories.")
    parser.add_argument("--combined", action="store_true", help="Use combined_counts.json instead of measurement_counts.json")
    args = parser.parse_args()

    counts_filename = "combined_counts.json" if args.combined else "measurement_counts.json"

    for run_dir in args.run_dirs:
        counts_path = run_dir / counts_filename
        if not counts_path.exists():
            print(f"[skip] No {counts_filename} found in {run_dir}")
            continue

        print(f"[process] Reordering {counts_path}...")
        with open(counts_path, "r", encoding="utf-8") as f:
            counts = json.load(f)

        reordered = reorder_counts(counts, run_dir)

        # Save reordered counts
        reordered_filename = f"reordered_{counts_filename}"
        with open(run_dir / reordered_filename, "w", encoding="utf-8") as f:
            json.dump(reordered, f, indent=2, sort_keys=True)
        print(f"[done] Saved to {run_dir / reordered_filename}")

if __name__ == "__main__":
    main()
