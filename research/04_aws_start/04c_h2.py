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

# AWS Braket
from braket.circuits import Circuit as BraketCircuit
from braket.aws import AwsDevice, AwsSession
import boto3


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


def main():
    # Hard-code your Braket target device
    device_arn = BraketDevice.ANKAA_3.value
    region = _parse_region_from_arn(device_arn)
    shots = 1000

    print("[info] Building H2 (R=2.0 Å) and computing RHF→UHF...")
    mol, rhf, uhf = build_h2_rhf_uhf(R=2.0)
    print(f"[info] RHF energy = {rhf.e_tot:.8f}  |  UHF energy = {uhf.e_tot:.8f}")

    print("[info] Constructing Qiskit UHF rotation circuit...")
    qc = make_qiskit_uhf_rotation_circuit(mol, rhf, uhf)
    print(qc)
    n_qubits = qc.num_qubits
    print(f"[info] Qiskit circuit has {n_qubits} qubits. Exporting to OpenQASM3 for Braket...")

    # Prepare output directory
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "04c_h2" / f"bond_length_2.00" / _now_tag()
    _ensure_dir(data_dir)

    # Save Qiskit circuit diagram for reference
    _write_text(data_dir / "qiskit_circuit.txt", str(qc))

    # Record initial metadata
    metadata = {
        "molecule": "H2",
        "bond_length_angstrom": 2.0,
        "basis": "sto-3g",
        "n_qubits": int(n_qubits),
        "region": region,
        "device_arn": device_arn,
        "shots": shots,
        "rhf_energy": float(rhf.e_tot),
        "uhf_energy": float(uhf.e_tot),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(data_dir / "metadata.json", metadata)

    braket_circ = qiskit_to_braket(qc)

    print(f"[info] Running circuit on Amazon Braket device {device_arn} (shots={shots})...")
    task = run_on_braket(braket_circ, shots=shots, device_arn=device_arn, region=region)

    # Persist task info immediately for later retrieval
    task_arn = task.arn
    print(f"[info] Submitted Braket task: {task_arn}")
    _write_text(data_dir / "task_arn.txt", task_arn)
    _write_json(data_dir / "task_metadata_initial.json", task.metadata())

    # Wait up to 5 minutes for completion
    task.wait_until_completed(timeout_seconds=300)

    # Upon completion, fetch result and save
    result = task.result()
    _write_json(data_dir / "task_metadata_final.json", task.metadata())

    counts = result.measurement_counts

    print("\nMeasurement counts (top 10):")
    if counts:
        _write_json(data_dir / "measurement_counts.json", dict(counts))
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for bitstr, cnt in sorted_items:
            print(f"  {bitstr}: {cnt}")
        _write_json(
            data_dir / "measurement_counts_top10.json",
            {k: v for k, v in sorted_items},
        )
    else:
        print("  (no counts returned)")
        _write_text(data_dir / "measurement_counts.txt", "no counts returned\n")

    print(f"[info] Artifacts saved under: {data_dir}")


# (aliases and dynamic resolver removed for simplicity)


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


def make_qiskit_uhf_rotation_circuit(mol: gto.Mole, rhf: scf.RHF, uhf: scf.UHF):
    """Create the Qiskit circuit that prepares HF and applies RHF→UHF rotation."""
    qc = rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf, optimize_single_slater=True)
    return qc


def qiskit_to_braket(qc) -> BraketCircuit:
    """Transpile to a simple basis and translate directly into a Braket Circuit.

    We avoid exporting OpenQASM3 to prevent external include dependencies
    like `stdgates.inc`. Only maps a small gate set used by the transpiled
    circuit: rz, sx, x, cx, and measurement. The `sx` gate is implemented as
    an Rx(pi/2) up to a global phase, which is acceptable for state prep and
    measurement workflows.
    """
    # Transpile to a compact, supported basis
    tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3)

    # Ensure measurements exist for sampling backends
    if tqc.num_clbits == 0:
        tqc.measure_all()

    # Build Braket circuit natively
    bkc = BraketCircuit()
    from math import pi

    for instr, qargs, cargs in tqc.data:
        name = instr.name
        qubits = [q._index for q in qargs]

        if name == "rz":
            theta = float(instr.params[0])
            bkc.rz(qubits[0], theta)
        elif name == "sx":
            # sqrt(X) = Rx(pi/2) up to global phase
            bkc.rx(qubits[0], pi / 2)
        elif name == "x":
            bkc.x(qubits[0])
        elif name in ("cx", "cnot"):
            bkc.cnot(qubits[0], qubits[1])
        elif name in ("barrier",):
            # ignore
            continue
        elif name in ("measure",):
            # We'll add a full measurement at the end (measure_all was applied)
            continue
        else:
            raise ValueError(f"Unsupported gate after transpile: {name}")

    # Finally, measure all qubits to collect counts
    bkc.measure(range(tqc.num_qubits))
    return bkc


def run_on_braket(
    circuit: BraketCircuit,
    shots: int = 1000,
    *,
    device_arn: str,
    region: str | None = None,
):
    """Run a Braket Circuit on a selected device and return the AwsQuantumTask."""
    region = region or _parse_region_from_arn(device_arn) or "us-east-1"
    session = AwsSession(boto_session=boto3.Session(region_name=region))
    device = AwsDevice(device_arn, aws_session=session)
    task = device.run(circuit, shots=shots)
    return task


# Removed legacy argparse-based CLI flow and try/except blocks for simplicity


if __name__ == "__main__":
    main()
