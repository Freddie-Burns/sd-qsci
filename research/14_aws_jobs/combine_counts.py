import json
import uuid
from pathlib import Path
from datetime import datetime
from collections import Counter

# Target directory containing subfolders with measurement_counts.json
TARGET_DIR = Path(__file__).resolve().parent / "data" / "14b_h4_forte" / "20260130-212325"

def get_counts_from_path(path):
    """Helper to get counts from a path, checking for various possible filenames."""
    for filename in ["measurement_counts.json", "combined_counts.json"]:
        counts_file = path / filename
        if counts_file.exists():
            with open(counts_file, "r") as f:
                return json.load(f)
    return None

def get_metadata_from_path(path):
    """Helper to get metadata from a path, checking for various possible filenames."""
    for filename in ["metadata.json", "job_metadata.json", "combined_metadata.json"]:
        metadata_file = path / filename
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)
    return None

def uhf_soln_combine(paths):
    paths = [Path(p) for p in paths]
    if not paths:
        print("No paths provided for UHF combination.")
        return

    # Use the parent of the first path as the base directory for the new folder
    base_dir = paths[0].parent
    
    # New naming convention: combined_last6-last6
    suffixes = [p.name[-6:] for p in paths]
    dir_name = "combined_" + "-".join(suffixes)
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_counter = Counter()
    metadata = {
        "origins": [],
        "combined_at": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "type": "uhf_combined"
    }

    # Extract common metadata if possible
    common_metadata = {}

    for path in paths:
        counts = get_counts_from_path(path)
        if counts:
            combined_counter.update(counts)
            
        origin_metadata = get_metadata_from_path(path)
        if origin_metadata:
            # Take geometry from the first one that has it
            if "geometry" not in common_metadata and "geometry" in origin_metadata:
                common_metadata["geometry"] = origin_metadata["geometry"]
            if "device" not in common_metadata and "device" in origin_metadata:
                common_metadata["device"] = origin_metadata["device"]
            if "molecule" not in common_metadata and "molecule" in origin_metadata:
                common_metadata["molecule"] = origin_metadata["molecule"]

        origin_info = {
            "path": str(path.resolve()),
            "timestamp": path.name,
            "metadata": origin_metadata
        }
        metadata["origins"].append(origin_info)

    metadata.update(common_metadata)

    # Save combined counts
    output_counts_file = output_dir / "combined_counts.json"
    with open(output_counts_file, "w") as f:
        json.dump(dict(combined_counter), f, indent=2)

    # Save top 10 counts
    output_top10_file = output_dir / "combined_counts_top10.json"
    top_10_counts = dict(combined_counter.most_common(10))
    with open(output_top10_file, "w") as f:
        json.dump(top_10_counts, f, indent=2)

    # Save metadata
    output_metadata_file = output_dir / "metadata.json"
    with open(output_metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"UHF combined results saved to: {output_dir}")
    return output_dir

def combine_counts(target_dir):
    target_path = Path(target_dir)
    # Find all subdirectories that contain measurement_counts.json
    subdirs = [d for d in target_path.iterdir() if d.is_dir() and (d / "measurement_counts.json").exists()]
    
    if not subdirs:
        print(f"No subdirectories with measurement_counts.json found in {target_path}")
        return

    combined_counter = Counter()
    metadata = {
        "origins": [],
        "combined_at": datetime.now().strftime("%Y%m%d-%H%M%S")
    }

    for path in sorted(subdirs):
        counts = get_counts_from_path(path)
        if counts:
            combined_counter.update(counts)
            
        # Collect metadata from the origin
        origin_metadata = {
            "path": str(path.resolve()),
            "original_metadata": get_metadata_from_path(path)
        }
        
        metadata["origins"].append(origin_metadata)

    # Save combined counts and metadata in the target directory
    output_counts_file = target_path / "combined_counts.json"
    output_top10_file = target_path / "combined_counts_top10.json"
    output_metadata_file = target_path / "combined_metadata.json"

    with open(output_counts_file, "w") as f:
        json.dump(dict(combined_counter), f, indent=2)

    # Save top 10 counts
    top_10_counts = dict(combined_counter.most_common(10))
    with open(output_top10_file, "w") as f:
        json.dump(top_10_counts, f, indent=2)

    with open(output_metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Combined counts saved to: {output_counts_file}")
    print(f"Top 10 counts saved to: {output_top10_file}")
    print(f"Combined metadata saved to: {output_metadata_file}")

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    
    # 1. Combine batches within each directory
    targets = [
        data_dir / "14f_h6_forte" / "20260130-220739",
        data_dir / "14f_h6_forte" / "20260130-220814"
    ]

    for target in targets:
        if target.exists():
            print(f"Combining counts for: {target}")
            combine_counts(target)
        else:
            print(f"Target directory not found: {target}")

    # 2. Combine the combined results into a single folder
    uhf_soln_combine(targets[::-1])
