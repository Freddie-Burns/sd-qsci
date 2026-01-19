"""
Create a UHF orbital-rotation quantum circuit for H2 at bond length 2.0 Å and
run it on Amazon Braket (SV1) similarly to 04a and 04b examples.

Workflow:
- Build H2 molecule in PySCF (sto-3g, R=2.0 Å), run RHF and derive UHF.
- Construct the RHF→UHF orbital-rotation circuit (Qiskit) using project utils.
- Transpile and export to OpenQASM 3, import into a Braket Circuit.
- Measure all qubits and run on the managed simulator SV1.

Notes:
- The circuit is created with the existing Qiskit+ffsim tooling in this repo
  and then translated to OpenQASM3. If translation of a custom gate fails on
  your environment, the script prints guidance. Ensure dependencies from
  pyproject.toml are installed (Qiskit 1.x, ffsim, PySCF, Braket SDK).
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Tuple

import numpy as np
from pyscf import gto, scf

from sd_qsci.utils import uhf_from_rhf
from sd_qsci.circuit import rhf_uhf_orbital_rotation_circuit

# Qiskit for circuit construction/export
from qiskit import transpile, QuantumCircuit
from qiskit_braket_provider import BraketProvider


class BraketDevice(Enum):
    """Canonical selection of target devices (names as used by qiskit-braket-provider)."""
    # us-east-1
    ARIA_1 = "Aria-1"
    AQUILA = "Aquila"
    FORTE_1 = "Forte 1"
    FORTE_ENTERPRISE_1 = "Forte Enterprise 1"

    # us-west-1
    ANKAA_3 = "Ankaa-3"

    # eu-north-1
    GARNET = "Garnet"
    EMERALD = "Emerald"
    IBEX_Q1 = "Ibex Q1"

    # Global simulators
    SV1 = "SV1"
    TN1 = "TN1"
    DM1 = "dm1"


# Hard-coded configuration (no CLI)
MODE = "fetch"  # choose either "submit" or "fetch"
DEVICE_NAME = BraketDevice.SV1.value  # use same naming style as 04b_qiskit_to_braket.py
SHOTS = 100
BOND_LENGTH = 2.0  # used in submit mode to build circuit and output path

def main():
    if MODE == "submit":
        submit()
    elif MODE == "fetch":
        fetch()
        
def submit():
        # Build problem and circuit
        mol, rhf, uhf = build_h2_rhf_uhf(R=BOND_LENGTH)
        print(f"[info] RHF energy = {rhf.e_tot:.8f}  |  UHF energy = {uhf.e_tot:.8f}")
        qc = rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf, optimize_single_slater=True)
        qc.measure_all()
        n_qubits = qc.num_qubits
        print(f"[info] Qiskit circuit has {n_qubits} qubits. Submitting via qiskit-braket-provider...")

        # Prepare output directory
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir / "data" / "04c_h2" / now_tag()
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save circuit diagram
        write_text(data_dir / "qiskit_circuit.txt", str(qc))

        # Record initial metadata
        metadata = {
            "molecule": "H2",
            "geometry": mol.atom,
            "basis": "sto-3g",
            "n_qubits": int(n_qubits),
            "device": DEVICE_NAME,
            "SHOTS": SHOTS,
            "rhf_energy": float(rhf.e_tot),
            "uhf_energy": float(uhf.e_tot),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(data_dir / "metadata.json", metadata)

        # Submit
        print(f"[info] Submitting circuit to {DEVICE_NAME} (SHOTS={SHOTS}) via Qiskit Braket provider...")
        provider = BraketProvider()
        backend = provider.get_backend(DEVICE_NAME)
        tqc = transpile(qc, backend=backend)
        job = backend.run(tqc, shots=SHOTS)
        job_id = job.job_id()
        print(f"[info] Submitted Braket job: {job_id}")
        write_text(data_dir / "job_id.txt", str(job_id))

        # Save minimal backend/job metadata
        write_json(
            data_dir / "job_metadata.json",
            {
                "backend": backend.name,
                "device": DEVICE_NAME,
                "SHOTS": SHOTS,
                "job_id": job_id,
                "submitted_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

        print(f"[info] Submission artifacts saved under: {data_dir}")


def fetch():
        base_dir = Path(__file__).resolve().parent
        run_dir_input = base_dir / "data" / "04c_h2" / "20260119-095902"
        if not run_dir_input:
            raise SystemExit("RUN_DIR must be set to the submission folder path when MODE='fetch'")

        run_dir = Path(run_dir_input)
        if not run_dir.exists():
            raise SystemExit(f"Run directory does not exist: {run_dir}")

        # Load metadata
        job_meta_path = run_dir / "job_metadata.json"
        job_id_path = run_dir / "job_id.txt"
        if not job_meta_path.exists() or not job_id_path.exists():
            raise SystemExit("job_metadata.json or job_id.txt not found in run directory")

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
        try:
            status = str(job.status())
        except Exception:
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
        try:
            counts = result.get_counts()
        except Exception:
            try:
                counts = result.results[0].data.counts  # type: ignore[attr-defined]
            except Exception:
                counts = None

        print("\nMeasurement counts (top 10):")
        if counts:
            write_json(run_dir / "measurement_counts.json", dict(counts))
            sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
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


def _fetch_one(run_dir: Path) -> None:
    """
    Retrieve results for a single submission directory if the job completed.
    Writes status to `job_status.txt` and, when available, saves
    `measurement_counts.json` and `measurement_counts_top10.json`.
    """
    if not run_dir.exists():
        print(f"[warn] Run directory does not exist: {run_dir}")
        return

    job_meta_path = run_dir / "job_metadata.json"
    job_id_path = run_dir / "job_id.txt"
    if not job_meta_path.exists() or not job_id_path.exists():
        print(f"[warn] Missing job_metadata.json or job_id.txt in {run_dir}")
        return

    job_meta = json.loads(job_meta_path.read_text())
    job_id = job_id_path.read_text().strip()
    device_meta = job_meta.get("device", DEVICE_NAME)

    print(f"[info] Retrieving job {job_id} on backend {device_meta}...")
    provider = BraketProvider()
    backend = provider.get_backend(device_meta)

    try:
        retrieve_job = getattr(backend, "retrieve_job")
    except AttributeError:
        print("[error] Backend does not support retrieve_job(job_id)")
        return

    job = retrieve_job(job_id)
    try:
        status = str(job.status())
    except Exception:
        status = "unknown"
    write_text(run_dir / "job_status.txt", status + "\n")
    print(f"[info] Current job status: {status}")

    status_upper = status.upper()
    if not any(s in status_upper for s in ["DONE", "COMPLETED", "SUCCESS"]):
        print("[info] Job not completed yet. Skipping for now.")
        return

    print("[info] Job completed. Fetching results...")
    result = job.result()

    # Try to obtain counts in a provider-agnostic way
    try:
        counts = result.get_counts()
    except Exception:
        try:
            counts = result.results[0].data.counts  # type: ignore[attr-defined]
        except Exception:
            counts = None

    if counts:
        write_json(run_dir / "measurement_counts.json", dict(counts))
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        write_json(
            run_dir / "measurement_counts_top10.json",
            {k: v for k, v in sorted_items},
        )
        print(f"[info] Saved counts for: {run_dir}")
    else:
        print(f"[info] No counts returned for: {run_dir}")


def fetch_all() -> None:
    """
    Walk through research/04_aws_start/data/04c_h2/* and fetch results for any
    run directory that has submission artifacts (job_metadata.json & job_id.txt)
    but is missing measurement_counts.json.
    """
    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / "data" / "04c_h2"
    if not data_root.exists():
        print(f"[info] Nothing to do. Data directory not found: {data_root}")
        return

    # Consider immediate subdirectories under data/04c_h2 as run directories
    run_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    if not run_dirs:
        print(f"[info] No run directories found under: {data_root}")
        return

    to_process = []
    for rd in sorted(run_dirs):
        has_submission = (rd / "job_metadata.json").exists() and (rd / "job_id.txt").exists()
        has_counts = (rd / "measurement_counts.json").exists()
        if has_submission and not has_counts:
            to_process.append(rd)

    if not to_process:
        print("[info] All runs already fetched (measurement_counts.json present).")
        return

    print(f"[info] Found {len(to_process)} run(s) to fetch.")
    for rd in to_process:
        print(f"\n[info] Processing: {rd}")
        _fetch_one(rd)


def build_h2_rhf_uhf(R: float = 2.0) -> Tuple[gto.Mole, scf.RHF, scf.UHF]:
    """
    Build H2 at distance R (Å), run RHF then derive UHF starting from RHF.
    Returns (mol, rhf, uhf).
    """
    mol = gto.Mole()
    mol.atom = f"H 0 0 0; H 0 0 {R}"
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0  # H2 singlet
    mol.build()

    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-10
    rhf.kernel()

    uhf = uhf_from_rhf(mol, rhf)
    return mol, rhf, uhf


def submit_qiskit_job(
    qc: QuantumCircuit,
    shots: int,
    device_name: str,
):
    """
    Submit a Qiskit circuit to an AWS Braket backend without waiting.
    Returns the provider Job object (non-blocking).
    """
    provider = BraketProvider()
    backend = provider.get_backend(device_name)
    tqc = transpile(qc, backend=backend)
    job = backend.run(tqc, shots=shots)
    return job


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.write_text(content)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


# Note: region parsing and explicit AwsSession handling removed to mirror
# 04b_qiskit_to_braket.py usage which relies on default AWS configuration
# available to the qiskit-braket-provider.


if __name__ == "__main__":
    main()
