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
from qiskit import transpile
from qiskit_braket_provider import BraketProvider

# AWS Braket
from braket.circuits import Circuit as BraketCircuit
from braket.aws import AwsDevice, AwsSession
import boto3


class BraketDevice(Enum):
    """Canonical selection of target devices (deduplicated simulators)."""
    # us-east-1
    ARIA_1 = "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
    AQUILA = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"
    FORTE_1 = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
    FORTE_ENTERPRISE_1 = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1"

    # us-west-1
    ANKAA_3 = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"

    # eu-north-1
    GARNET = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
    EMERALD = "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald"
    IBEX_Q1 = "arn:aws:braket:eu-north-1::device/qpu/aqt/Ibex-Q1"

    # Global simulators
    SV1 = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    TN1 = "arn:aws:braket:::device/quantum-simulator/amazon/tn1"
    DM1 = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"


# Hard-coded configuration (no CLI)
MODE = "submit"  # choose either "submit" or "fetch"
DEVICE_ARN = BraketDevice.ANKAA_3.value
REGION = DEVICE_ARN.split(":")[3]
SHOTS = 1000
BOND_LENGTH = 2.0  # used in submit mode to build circuit and output path
RUN_DIR = None  # set path (str or Path) to the run directory when MODE == "fetch"


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
        print(qc)
        n_qubits = qc.num_qubits
        print(f"[info] Qiskit circuit has {n_qubits} qubits. Submitting via qiskit-braket-provider...")

        # Prepare output directory
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir / "data" / "04c_h2" / f"bond_length_{BOND_LENGTH:.2f}" / _now_tag()
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save circuit diagram
        _write_text(data_dir / "qiskit_circuit.txt", str(qc))

        # Record initial metadata
        metadata = {
            "molecule": "H2",
            "bond_length_angstrom": float(BOND_LENGTH),
            "basis": "sto-3g",
            "n_qubits": int(n_qubits),
            "region": REGION,
            "DEVICE_ARN": DEVICE_ARN,
            "SHOTS": SHOTS,
            "rhf_energy": float(rhf.e_tot),
            "uhf_energy": float(uhf.e_tot),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(data_dir / "metadata.json", metadata)

        # Submit (non-blocking)
        print(f"[info] Submitting circuit to {DEVICE_ARN} (SHOTS={SHOTS}) via Qiskit backend...")
        job = submit_qiskit_job(qc, SHOTS=SHOTS, DEVICE_ARN=DEVICE_ARN, region=REGION)

        # Persist job info
        try:
            job_id = getattr(job, "job_id", None) or getattr(job, "id", None) or str(job)
        except Exception:
            job_id = "unknown"
        print(f"[info] Submitted Braket job: {job_id}")
        _write_text(data_dir / "job_id.txt", str(job_id))

        # Save minimal backend/job metadata
        try:
            backend_name = getattr(getattr(job, "backend", None), "name", None)
        except Exception:
            backend_name = None
        _write_json(
            data_dir / "job_metadata.json",
            {
                "backend": backend_name or "braket-backend",
                "DEVICE_ARN": DEVICE_ARN,
                "region": REGION,
                "SHOTS": SHOTS,
                "job_id": job_id,
                "submitted_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

        print(f"[info] Submission artifacts saved under: {data_dir}")


def fetch():
        run_dir_input = RUN_DIR
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
        DEVICE_ARN_meta = job_meta.get("DEVICE_ARN", DEVICE_ARN)
        region_meta = job_meta.get("region", REGION)

        # Reconnect to backend and retrieve job
        print(f"[info] Retrieving job {job_id} on backend {DEVICE_ARN_meta}...")
        session = AwsSession(boto_session=boto3.Session(region_name=region_meta))
        provider = BraketProvider(aws_session=session)
        backend = provider.get_backend(DEVICE_ARN_meta)

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
        _write_text(run_dir / "job_status.txt", status + "\n")
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
            _write_json(run_dir / "measurement_counts.json", dict(counts))
            sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
            for bitstr, cnt in sorted_items:
                print(f"  {bitstr}: {cnt}")
            _write_json(
                run_dir / "measurement_counts_top10.json",
                {k: v for k, v in sorted_items},
            )
        else:
            print("  (no counts returned)")
            _write_text(run_dir / "measurement_counts.txt", "no counts returned\n")

        print(f"[info] Result artifacts saved under: {run_dir}")


def build_h2_rhf_uhf(R: float = 2.0) -> Tuple[gto.Mole, scf.RHF, scf.UHF]:
    """Build H2 at distance R (Å), run RHF then derive UHF starting from RHF.

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


def run_qiskit_on_braket_backend(
    qc,
    SHOTS: int = 1000,
    *,
    DEVICE_ARN: str,
    region: str | None = None,
):
    """Run a Qiskit circuit on an AWS Braket backend using qiskit-braket-provider.

    Returns a tuple (job, result).
    """
    region = region or _parse_region_from_arn(DEVICE_ARN) or "us-east-1"
    session = AwsSession(boto_session=boto3.Session(region_name=region))
    provider = BraketProvider(aws_session=session)

    backend = provider.get_backend(DEVICE_ARN)
    tqc = transpile(qc, backend=backend)
    job = backend.run(tqc, SHOTS=SHOTS)
    result = job.result()
    return job, result


def submit_qiskit_job(
    qc,
    SHOTS: int = 1000,
    *,
    DEVICE_ARN: str,
    region: str | None = None,
):
    """Submit a Qiskit circuit to an AWS Braket backend without waiting.

    Returns the provider Job object (non-blocking)."""
    region = region or _parse_region_from_arn(DEVICE_ARN) or "us-east-1"
    session = AwsSession(boto_session=boto3.Session(region_name=region))
    provider = BraketProvider(aws_session=session)
    backend = provider.get_backend(DEVICE_ARN)
    tqc = transpile(qc, backend=backend)
    job = backend.run(tqc, SHOTS=SHOTS)
    return job


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _parse_region_from_arn(arn: str) -> str | None:
    parts = arn.split(":")
    # arn:partition:service:region:account-id:resource
    region = parts[3] if len(parts) > 3 else None
    return region or None


if __name__ == "__main__":
    main()
