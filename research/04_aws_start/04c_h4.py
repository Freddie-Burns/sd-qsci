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
from matplotlib import pyplot as plt
from pyscf import gto, scf
from qiskit import transpile, QuantumCircuit
from qiskit import qpy
from qiskit.quantum_info import Statevector
from qiskit_braket_provider import BraketProvider

from sd_qsci.utils import uhf_from_rhf
from sd_qsci.circuit import rhf_uhf_orbital_rotation_circuit


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


DEVICE_NAME = BraketDevice.FORTE_1.value
SHOTS = int(1e3)
BOND_LENGTH = 2.0  # used in submit mode to build circuit and output path

def main():
    submit()
    # fetch("20260119-170454")
    # fetch_all()

        
def submit():
        # Build problem and circuit
        mol, rhf, uhf = build_h4_rhf_uhf(R=BOND_LENGTH)
        print(f"[info] RHF energy = {rhf.e_tot:.8f}  |  UHF energy = {uhf.e_tot:.8f}")

        qc = rhf_uhf_orbital_rotation_circuit(
            mol, rhf, uhf, optimize_single_slater=True
        )

        # Display circuit for checking
        qc.decompose().draw(output="mpl")
        # plt.show()

        # Keep an unmeasured copy for statevector simulation
        qc_for_sim = qc.copy()

        # Add measurements to create the sampling circuit for the AWS run
        qc.measure_all()

        n_qubits = qc.num_qubits
        print(
            f"[info] Qiskit circuit has {n_qubits} qubits. Submitting via qiskit-braket-provider..."
        )

        # Prepare output directory
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir / "data" / now_tag()
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save circuit diagrams (human-readable)
        write_text(data_dir / "qiskit_circuit_measured.txt", str(qc))
        write_text(data_dir / "qiskit_circuit_unmeasured.txt", str(qc_for_sim))

        # Record initial metadata
        metadata = {
            "molecule": "H4",
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

        # Save detailed circuit data for later exact reconstruction
        with open(data_dir / "qiskit_circuit_unmeasured.qpy", "wb") as f:
            qpy.dump(qc_for_sim, f)
        with open(data_dir / "qiskit_circuit_measured.qpy", "wb") as f:
            qpy.dump(qc, f)

        # Local statevector simulation (using the unmeasured circuit)
        # Remove final measurements if any (safety), then get statevector
        qc_for_sv = getattr(qc_for_sim, "remove_final_measurements", None)
        if callable(qc_for_sv):
            qc_nom = qc_for_sim.remove_final_measurements(inplace=False)
        else:
            qc_nom = qc_for_sim

        sv = Statevector.from_instruction(qc_nom)

        # Save as .npy and JSON (real/imag)
        np.save(data_dir / "sim_statevector.npy", sv.data)

        sv_json = {
            "n_qubits": int(n_qubits),
            "dim": int(len(sv.data)),
            "amplitudes": [
                {"index": int(i), "real": float(z.real), "imag": float(z.imag)}
                for i, z in enumerate(sv.data)
            ],
        }
        write_json(data_dir / "sim_statevector.json", sv_json)

        # Also save top-10 probabilities for a quick glance
        probs = np.abs(sv.data) ** 2
        idx_sorted = np.argsort(probs)[::-1][:10]
        top10 = {
            format(int(i), f"0{n_qubits}b"): float(probs[i]) for i in idx_sorted
        }
        write_json(data_dir / "sim_statevector_top10.json", top10)

        # Submit to AWS Braket via Qiskit provider
        print(
            f"[info] Submitting circuit to {DEVICE_NAME} (SHOTS={SHOTS}) via Qiskit Braket provider..."
        )
        provider = BraketProvider()
        backend = provider.get_backend(DEVICE_NAME)
        tqc = transpile(qc, backend=backend)

        # Save transpiled circuit artifacts as well (text + QPY)
        write_text(data_dir / "qiskit_circuit_transpiled.txt", str(tqc))
        with open(data_dir / "qiskit_circuit_transpiled.qpy", "wb") as f:
            qpy.dump(tqc, f)

        tqc.draw(output="mpl", fold=-1)
        plt.show()

        # job = backend.run(tqc, shots=SHOTS)
        # job_id = job.job_id()
        # print(f"[info] Submitted Braket job: {job_id}")
        # write_text(data_dir / "job_id.txt", str(job_id))
        #
        # # Save minimal backend/job metadata
        # write_json(
        #     data_dir / "job_metadata.json",
        #     {
        #         "backend": getattr(backend, "name", getattr(backend, "__str__", lambda: "?")()),
        #         "device": DEVICE_NAME,
        #         "SHOTS": SHOTS,
        #         "job_id": job_id,
        #         "submitted_at": datetime.now().isoformat(timespec="seconds"),
        #     },
        # )
        #
        # print(f"[info] Submission and local simulation artifacts saved under: {data_dir}")


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


def fetch_all() -> None:
    """
    Walk through research/04_aws_start/data/04c_h2/* and fetch results for any
    run directory that has submission artifacts (job_metadata.json & job_id.txt)
    but is missing measurement_counts.json.
    """
    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / "data"
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
        fetch(rd)


def build_h4_rhf_uhf(R: float = 2.0) -> Tuple[gto.Mole, scf.RHF, scf.UHF]:
    """
    Build H2 at distance R (Å), run RHF then derive UHF starting from RHF.
    Returns (mol, rhf, uhf).
    """
    mol = gto.Mole()
    mol.atom = f"H 0 0 0; H 0 0 {R}; H 0 0 {2*R}; H 0 0 {3*R}"
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0  # H4 singlet
    mol.build()

    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-10
    rhf.kernel()

    uhf = uhf_from_rhf(mol, rhf)
    return mol, rhf, uhf


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
