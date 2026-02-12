import os
import json
import csv
import re
from pathlib import Path

def is_timestamp(name):
    return re.match(r'^\d{8}-\d{6}$', name) is not None

def get_top_bitstring(folder_path):
    # Try combined_counts_top10.json first, then measurement_counts_top10.json
    paths = [
        folder_path / "combined_counts_top10.json",
        folder_path / "measurement_counts_top10.json",
        folder_path / "reordered_measurement_counts.json"
    ]
    for p in paths:
        if p.exists():
            try:
                with open(p, 'r') as f:
                    data = json.load(f)
                    if data:
                        # Top bitstring is the first key
                        return list(data.keys())[0]
            except Exception:
                pass
    
    # If no top-level file, check if it's a batch folder and look in the first batch
    batches = sorted([sub for sub in folder_path.iterdir() if sub.is_dir() and is_timestamp(sub.name)])
    if batches:
        return get_top_bitstring(batches[0])
        
    return None

def get_bond_length(geometry_str):
    if not geometry_str:
        return ""
    # Geometry format: "H 0 0 0.0\n    H 0 0 1.5\n    ..."
    lines = [line.strip() for line in geometry_str.strip().split('\n') if line.strip()]
    if len(lines) < 2:
        return ""
    
    def get_z(line):
        parts = line.split()
        if len(parts) >= 4:
            return float(parts[3])
        return None

    z1 = get_z(lines[0])
    z2 = get_z(lines[1])
    
    if z1 is not None and z2 is not None:
        return abs(z2 - z1)
    return ""

def process_data(data_dir):
    results = []
    data_path = Path(data_dir)
    
    # Iterate over molecule/device folders (e.g., 14a_h4_ankaa)
    for entry in data_path.iterdir():
        if entry.is_dir() and not entry.name.startswith("combined_"):
            # Iterate over timestamped folders
            for ts_folder in entry.iterdir():
                if ts_folder.is_dir() and is_timestamp(ts_folder.name):
                    # Check for batches
                    batches = []
                    for sub in ts_folder.iterdir():
                        if sub.is_dir() and is_timestamp(sub.name):
                            batches.append(sub.name)
                    
                    # Read metadata
                    metadata_path = ts_folder / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                meta = json.load(f)
                        except Exception:
                            meta = {}
                    else:
                        meta = {}
                    
                    top_bitstring = get_top_bitstring(ts_folder)
                    
                    results.append({
                        "timestamp": ts_folder.name,
                        "device": meta.get("device", ""),
                        "molecule": meta.get("molecule", ""),
                        "bond length": get_bond_length(meta.get("geometry", "")),
                        "uhf seed": meta.get("uhf_spin_seed", ""),
                        "number of shots": meta.get("SHOTS", ""),
                        "batches": len(batches) if batches else 1,
                        "top measured bitstring": top_bitstring or ""
                    })
    return results

def main():
    base_dir = Path("research/14_aws_jobs/data")
    output_file = "research/14_aws_jobs/jobs_summary.csv"
    
    print(f"Processing data in {base_dir}...")
    results = process_data(base_dir)
    
    fieldnames = ["timestamp", "device", "molecule", "bond length", "uhf seed", "number of shots", "batches", "top measured bitstring"]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"Successfully created {output_file} with {len(results)} rows.")

if __name__ == "__main__":
    main()
