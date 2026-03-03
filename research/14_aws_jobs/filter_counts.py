from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Dict, Any

def filter_particle_number_counts(run_dir: Path, counts: dict[str, int], meta: dict) -> dict[str, int]:
    """
    Filter counts to only include bitstrings with the correct Hamming weight.
    
    Determines the target Hamming weight from metadata (H4 -> 4, H6 -> 6).
    Saves filtered counts to filtered_counts.json.
    """
    mol_name = meta.get("molecule") or meta.get("geometry_name")
    
    if not mol_name:
        print(f"[warning] Could not determine molecule name for {run_dir}. Skipping filtering.")
        return counts

    target_weight = None
    if "H4" in mol_name.upper():
        target_weight = 2
    elif "H6" in mol_name.upper():
        target_weight = 3
    else:
        print(f"[warning] Unknown molecule {mol_name} for {run_dir}. Skipping filtering.")
        return counts

    # Collect particle number counts
    particle_counts = {}
    for bs, count in counts.items():
        weight = sum(int(bit) for bit in bs)
        particle_counts[weight] = particle_counts.get(weight, 0) + count

    # Save particle number counts
    stats_path = run_dir / "particle_number_counts.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        # Convert keys to strings for JSON
        json.dump({str(k): v for k, v in sorted(particle_counts.items())}, f, indent=2)

    # Filter only correct alpha and beta number bitstrings
    filtered_counts = {
        bs: count for bs, count in counts.items() 
        if sum(int(bit) for bit in bs[:len(bs)//2]) == target_weight
        and sum(int(bit) for bit in bs[len(bs)//2:]) == target_weight
    }

    output_path = run_dir / "filtered_counts.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_counts, f, indent=2, sort_keys=True)
    
    # Save top 10 filtered counts
    top10_output_path = run_dir / "filtered_counts_top10.json"
    # Sort by value descending and take top 10
    top10_filtered_counts = dict(sorted(filtered_counts.items(), key=lambda item: item[1], reverse=True)[:10])
    with open(top10_output_path, "w", encoding="utf-8") as f:
        json.dump(top10_filtered_counts, f, indent=2, sort_keys=True)
    
    print(f"[process] Saved top 10 filtered counts to {top10_output_path}")

    # Also keep saving to correct_particle_number_counts.json for backward compatibility if needed, 
    # but the requirement is filtered_counts.json
    legacy_output_path = run_dir / "correct_particle_number_counts.json"
    with open(legacy_output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_counts, f, indent=2, sort_keys=True)

    print(f"[process] Saved particle counts to {stats_path}")
    print(f"[process] Saved filtered counts to {output_path}")
    return filtered_counts

def get_metadata_from_path(path: Path):
    """Helper to get metadata from a path, checking for various possible filenames."""
    for filename in ["metadata.json", "job_metadata.json", "combined_metadata.json"]:
        metadata_file = path / filename
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)
    return None

def main():
    data_dir = Path(__file__).resolve().parent / "data"
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found.")
        return

    print(f"Scanning {data_dir} for measurement and combined counts...")
    
    count_files = ["measurement_counts.json", "combined_counts.json"]
    
    for root, dirs, files in os.walk(data_dir):
        root_path = Path(root)
        for count_file in count_files:
            if count_file in files:
                counts_path = root_path / count_file
                print(f"\nProcessing {counts_path}...")
                
                with open(counts_path, "r") as f:
                    try:
                        counts = json.load(f)
                    except json.JSONDecodeError:
                        print(f"[error] Failed to decode JSON from {counts_path}")
                        continue
                
                meta = get_metadata_from_path(root_path)
                if not meta:
                    # Try parent directory for metadata if not found in current (common in combined folders or batch folders)
                    meta = get_metadata_from_path(root_path.parent)
                if not meta:
                    # Try grandparent (some batch jobs are deeply nested)
                    meta = get_metadata_from_path(root_path.parent.parent)
                
                if meta:
                    filter_particle_number_counts(root_path, counts, meta)
                else:
                    print(f"[warning] No metadata found for {root_path}. Skipping.")

if __name__ == "__main__":
    main()
