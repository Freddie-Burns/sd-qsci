"""
I am confused by the bitstrings I am getting back from Forte.
This is a script to try and test whether the qubit ordering is
as I expect.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from time import sleep

from qiskit import transpile, QuantumCircuit
from qiskit import qpy
from qiskit_braket_provider import BraketProvider


DEVICE_NAME = "Forte 1"
SHOTS = 100  # per requirement: run each circuit with 100 shots


def main():
    # submit_one()
    submit_all()
    # fetch_all()


def fetch_layout():
    """
    Try to decipher qubit reordering from the saved transpiled circuit file.
    """



def submit_one():
    """
    Submit a single circuit with a single X gate on qubit 6 (Qiskit indexing 0).
    """
    # Prepare output directory
    base_dir = Path(__file__).resolve().parent
    run_root = base_dir / "data" / now_tag()
    run_root.mkdir(parents=True, exist_ok=True)

    # Connect to Forte backend via qiskit-braket provider
    provider = BraketProvider()
    backend = provider.get_backend(DEVICE_NAME)

    # 8 qubits 0-7, X gate on qubit 6
    qc = QuantumCircuit(8)
    qc.x(6)
    qc.measure_all()

    # Save human-readable circuit
    circ_dir = run_root
    circ_dir.mkdir(parents=True, exist_ok=True)
    write_text(circ_dir / "qiskit_circuit_measured.txt", str(qc))

    # Save QPY for exact reconstruction
    with open(circ_dir / "qiskit_circuit_measured.qpy", "wb") as f:
        qpy.dump(qc, f)

    # Transpile wihtout optimisation!
    tqc = transpile(qc, backend=backend, optimization_level=0)
    write_text(circ_dir / "qiskit_circuit_transpiled.txt", str(tqc))
    with open(circ_dir / "qiskit_circuit_transpiled.qpy", "wb") as f:
        qpy.dump(tqc, f)

    # Submit job
    job = backend.run(tqc, shots=SHOTS)
    job_id = job.job_id()
    print(f"[info] Submitted circuit (X on qubit 6) -> job_id={job_id}")

    # Save job metadata for later retrieval
    write_text(circ_dir / "job_id.txt", str(job_id))
    write_json(
        circ_dir / "job_metadata.json",
        {
            "backend": getattr(
                backend,
                "name",
                getattr(backend, "__str__", lambda: "?")()
            ),
            "device": DEVICE_NAME,
            "shots": SHOTS,
            "job_id": job_id,
            "x_qubit": 6,
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

def submit_all():
    """
    Create 8 simple circuits (8 qubits) with a single X gate on qubits 1..8
    (Qiskit indexing 0..7), measure all, and submit each to Forte with 100 shots.
    Saves per-circuit artifacts and job ids under a timestamped data folder.
    """
    # Prepare output directory
    base_dir = Path(__file__).resolve().parent
    run_root = base_dir / "data" / now_tag()
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] Preparing 8 X-gate circuits and submitting to {DEVICE_NAME} with {SHOTS} shots each...")

    # Connect to Forte backend via qiskit-braket provider
    provider = BraketProvider()
    backend = provider.get_backend(DEVICE_NAME)

    n_qubits_total = 8
    job_ids = []

    for k in range(4):
        sleep(2) # Ensure there is a gap between submitting jobs.
        # Circuit k+1: X on qubit k (so circuits correspond to qubits 1..8 for the user)
        qc = QuantumCircuit(n_qubits_total)
        qc.x(k)
        qc.measure_all()

        # Save human-readable circuit
        circ_dir = run_root / f"circuit_{k+1}"
        circ_dir.mkdir(parents=True, exist_ok=True)
        write_text(circ_dir / "qiskit_circuit_measured.txt", str(qc))

        # Save QPY for exact reconstruction
        with open(circ_dir / "qiskit_circuit_measured.qpy", "wb") as f:
            qpy.dump(qc, f)

        # Transpile and save transpiled representation
        tqc = transpile(qc, backend=backend, optimization_level=0)
        write_text(circ_dir / "qiskit_circuit_transpiled.txt", str(tqc))
        with open(circ_dir / "qiskit_circuit_transpiled.qpy", "wb") as f:
            qpy.dump(tqc, f)

        # Submit job
        job = backend.run(tqc, shots=SHOTS)
        job_id = job.job_id()
        job_ids.append(job_id)
        print(f"[info] Submitted circuit {k+1} (X on qubit {k+1}) -> job_id={job_id}")

        # Save job metadata for later retrieval
        write_text(circ_dir / "job_id.txt", str(job_id))
        write_json(
            circ_dir / "job_metadata.json",
            {
                "backend": getattr(backend, "name", getattr(backend, "__str__", lambda: "?")()),
                "device": DEVICE_NAME,
                "shots": SHOTS,
                "job_id": job_id,
                "circuit_index": k + 1,
                "x_qubit": k + 1,  # 1-based for readability
                "submitted_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

    # Save a summary at the root
    write_json(
        run_root / "submission_summary.json",
        {"device": DEVICE_NAME, "shots": SHOTS, "job_ids": job_ids},
    )

    print(f"[info] Submitted 8 jobs. Artifacts under: {run_root}")


def fetch(time_tag: str):
    base_dir = Path(__file__).resolve().parent
    run_dir_input = base_dir / "data" / time_tag

    run_dir = Path(run_dir_input)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    # Load metadata
    job_meta_path = run_dir / "job_metadata.json"
    job_id_path = run_dir / "job_id.txt"
    if not job_meta_path.exists() or not job_id_path.exists():
        raise SystemExit(
            "job_metadata.json or job_id.txt not found in run directory")

    job_meta = json.loads(job_meta_path.read_text())
    job_id = job_id_path.read_text().strip()
    device_meta = job_meta.get("device", DEVICE_NAME)

    # Reconnect to backend and retrieve job
    print(f"[info] Retrieving job {job_id} on backend {device_meta}...")
    provider = BraketProvider()
    backend = provider.get_backend(device_meta)

    try:
        retrieve_job = getattr(backend, "retrieve_job")
    except AttributeError:
        raise SystemExit("Backend does not support retrieve_job(job_id)")

    job = retrieve_job(job_id)
    # Save current status
    status = str(job.status())
    status = "unknown"
    write_text(run_dir / "job_status.txt", status + "\n")
    print(f"[info] Current job status: {status}")

    # If not done, exit without blocking
    status_upper = status.upper()
    if not any(s in status_upper for s in ["DONE", "COMPLETED", "SUCCESS"]):
        print("[info] Job not completed yet. Run fetch again later.")
        return

    print("[info] Job completed. Fetching results...")
    result = job.result()

    # Fetch counts from Qiskit Result
    counts = result.get_counts()

    print("\nMeasurement counts (top 10):")
    if counts:
        write_json(run_dir / "measurement_counts.json", dict(counts))
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1],
                              reverse=True)[:10]
        for bitstr, cnt in sorted_items:
            print(f"  {bitstr}: {cnt}")
        write_json(
            run_dir / "measurement_counts_top10.json",
            {k: v for k, v in sorted_items},
        )
    else:
        print("  (no counts returned)")
        write_text(run_dir / "measurement_counts.txt", "no counts returned\n")

    print(f"[info] Result artifacts saved under: {run_dir}")


def fetch_all() -> None:
    """
    Walk through research/04_aws_start/data/* and fetch results for any run
    directory produced by submit(), which now creates per-circuit subfolders
    (circuit_1..circuit_8). For each circuit folder that has submission
    artifacts (job_metadata.json & job_id.txt) but is missing
    measurement_counts.json, retrieve results and save them within that
    circuit folder.

    Backward compatibility: if a run directory contains job files at its root
    (older layout), fetch results there as well.
    """
    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / "data"
    if not data_root.exists():
        print(f"[info] Nothing to do. Data directory not found: {data_root}")
        return

    # Consider immediate subdirectories under data/ as run directories
    run_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    if not run_dirs:
        print(f"[info] No run directories found under: {data_root}")
        return

    provider = BraketProvider()

    def process_job_dir(job_dir: Path) -> None:
        meta_path = job_dir / "job_metadata.json"
        id_path = job_dir / "job_id.txt"
        if not (meta_path.exists() and id_path.exists()):
            return

        counts_path = job_dir / "measurement_counts.json"
        if counts_path.exists():
            return  # already fetched

        job_meta = json.loads(meta_path.read_text())
        job_id = id_path.read_text().strip()
        device_name = job_meta.get("device", DEVICE_NAME)

        print(f"[info] Retrieving job {job_id} on backend {device_name} for {job_dir.name}...")
        backend = provider.get_backend(device_name)

        retrieve_job = getattr(backend, "retrieve_job", None)
        if retrieve_job is None:
            print("[warn] Backend does not support retrieve_job(job_id); skipping.")
            return

        job = retrieve_job(job_id)
        status = str(job.status())
        write_text(job_dir / "job_status.txt", status + "\n")
        print(f"[info] Current job status: {status}")

        status_upper = status.upper()
        if not any(s in status_upper for s in ["DONE", "COMPLETED", "SUCCESS"]):
            print("[info] Job not completed yet. Skipping for now.")
            return

        print("[info] Job completed. Fetching results...")
        result = job.result()
        counts = result.get_counts()

        if counts:
            write_json(counts_path, dict(counts))
            # Save top-10 as a convenience
            sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
            write_json(job_dir / "measurement_counts_top10.json", {k: v for k, v in sorted_items})
            print(f"[info] Saved counts for {job_dir}")
        else:
            write_text(job_dir / "measurement_counts.txt", "no counts returned\n")
            print(f"[info] No counts returned for {job_dir}")

    # Walk each run directory. Prefer per-circuit subdirectories; otherwise, try flat layout
    total_dirs = 0
    fetched_dirs = 0
    for rd in sorted(run_dirs):
        circuit_dirs = [d for d in rd.iterdir() if d.is_dir() and d.name.startswith("circuit_")]
        if circuit_dirs:
            print(f"\n[info] Processing run: {rd.name} ({len(circuit_dirs)} circuit folders)")
            for cdir in sorted(circuit_dirs, key=lambda p: p.name):
                total_dirs += 1
                before = (cdir / "measurement_counts.json").exists()
                process_job_dir(cdir)
                after = (cdir / "measurement_counts.json").exists()
                if after and not before:
                    fetched_dirs += 1
        else:
            # Legacy flat layout
            print(f"\n[info] Processing legacy/flat run: {rd.name}")
            total_dirs += 1
            before = (rd / "measurement_counts.json").exists()
            process_job_dir(rd)
            after = (rd / "measurement_counts.json").exists()
            if after and not before:
                fetched_dirs += 1

    if total_dirs == 0:
        print("[info] No circuit or run directories to process.")
    else:
        print(f"\n[info] Fetch complete. Updated {fetched_dirs} of {total_dirs} directories.")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.write_text(content)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
