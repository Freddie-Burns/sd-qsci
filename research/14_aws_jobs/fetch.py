import json
from pathlib import Path
from qiskit_braket_provider import BraketProvider
from aws_utils import write_json, write_text

def fetch_job(run_dir: Path):
    job_id_path = run_dir / "job_id.txt"
    metadata_path = run_dir / "job_metadata.json"
    
    if not job_id_path.exists():
        print(f"[warning] No job_id.txt found in {run_dir}")
        return

    job_id = job_id_path.read_text().strip()
    
    # Try to get device name from metadata
    device_name = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            device_name = metadata.get("device") or metadata.get("backend")
        except Exception as e:
            print(f"[warning] Could not read metadata in {run_dir}: {e}")
    
    if not device_name:
        print(f"[warning] Could not determine device for job {job_id} in {run_dir}. Skipping.")
        return

    # Check for specific region in metadata if possible
    # Garnet/Emerald are in eu-north-1
    # Ankaa-3 is in us-west-1
    # Forte-1 is in us-east-1
    # Aria-1/2 are in us-east-1
    # This might be needed if the default region is not where the device is

    print(f"[info] Retrieving job {job_id} on {device_name}...")
    provider = BraketProvider()
    backend = provider.get_backend(device_name)
    
    try:
        job = backend.retrieve_job(job_id)
    except Exception as e:
        print(f"[error] Failed to retrieve job {job_id}: {e}")
        return

    status = str(job.status())
    write_text(run_dir / "job_status.txt", status + "\n")
    print(f"[info] Current job status: {status}")

    # Check if job is completed
    status_upper = status.upper()
    # Qiskit JobStatus: INITIALIZING, QUEUED, VALIDATING, RUNNING, CANCELLED, DONE, ERROR
    if "DONE" not in status_upper and "COMPLETED" not in status_upper:
        print(f"[info] Job {job_id} is not completed yet ({status}).")
        return

    print(f"[info] Job {job_id} completed. Fetching results...")
    try:
        result = job.result()
        counts = result.get_counts()
        
        if counts:
            write_json(run_dir / "measurement_counts.json", dict(counts))
            print(f"[success] Saved measurement_counts.json to {run_dir}")
            
            # Save top 10 for convenience
            sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
            write_json(run_dir / "measurement_counts_top10.json", {k: v for k, v in sorted_items})
        else:
            print(f"[warning] No counts returned for job {job_id}")
            write_text(run_dir / "measurement_counts.txt", "no counts returned\n")
            
    except Exception as e:
        print(f"[error] Failed to fetch or process results for job {job_id}: {e}")

def fetch_all():
    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / "data"
    
    if not data_root.exists():
        print(f"[info] Data directory not found: {data_root}")
        return

    print(f"[info] Scanning {data_root} for jobs to fetch...")
    
    # Walk through the data directory recursively
    # Look for any directory that contains job_id.txt but not measurement_counts.json
    for job_id_path in data_root.rglob("job_id.txt"):
        run_dir = job_id_path.parent
        counts_path = run_dir / "measurement_counts.json"
        
        if not counts_path.exists():
            print(f"\n[process] Found job in {run_dir}")
            fetch_job(run_dir)

if __name__ == "__main__":
    fetch_all()
