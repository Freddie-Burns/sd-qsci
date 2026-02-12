from pathlib import Path
import sys
import os

# Add the research/14_aws_jobs directory to the path so we can import combine_counts
script_dir = Path(__file__).resolve().parent / "research" / "14_aws_jobs"
sys.path.append(str(script_dir))

import combine_counts

def test_uhf_soln_combine():
    target_dir = Path("research/14_aws_jobs/data/14b_h4_forte/20260130-212325")
    path1 = target_dir / "20260130-212326"
    path2 = target_dir / "20260130-212330"
    
    print(f"Testing uhf_soln_combine with:\n  {path1}\n  {path2}")
    
    output_dir = combine_counts.uhf_soln_combine([path1, path2])
    
    if output_dir and output_dir.exists():
        print(f"Success: Output directory {output_dir} created.")
        files = list(output_dir.iterdir())
        print(f"Files in output directory: {[f.name for f in files]}")
        
        # Verify files exist
        expected_files = ["combined_counts.json", "combined_counts_top10.json", "metadata.json"]
        for ef in expected_files:
            if (output_dir / ef).exists():
                print(f"  - {ef} exists.")
            else:
                print(f"  - {ef} MISSING!")
    else:
        print("Failure: Output directory not created.")

if __name__ == "__main__":
    test_uhf_soln_combine()
