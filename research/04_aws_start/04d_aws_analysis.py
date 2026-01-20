from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from pyscf import gto, scf

from sd_qsci.analysis import run_quantum_chemistry_calculations
from sd_qsci import analysis, plot


# Minimal, hard-coded configuration. Change these for each run.
DATE_TAG = "20260119-170454"  # e.g., folder name under data/


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / DATE_TAG
    in_path = base_dir / "data" / DATE_TAG / "measurement_counts.json"
    meta_path = base_dir / "data" / DATE_TAG / "metadata.json"

    # Load JSON payload (assume a dict); minimal branching
    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Accept either direct mapping or nested under "measurement_counts"
    counts = payload.get("measurement_counts", payload)

    # Infer qubit count and construct ordered basis
    n = max(len(k) for k in counts.keys())
    order = [format(i, f"0{n}b") for i in range(2**n)]

    total = sum(int(counts.get(bs, 0)) for bs in order)
    probs = np.array([int(counts.get(bs, 0)) / total for bs in order], dtype=float)

    # Convert probabilities to amplitude vector (non-negative, no phases) and L2-normalize
    amps = np.sqrt(probs).astype(complex)
    norm = np.linalg.norm(amps)
    amps = amps / norm

    # Build a Qiskit Statevector so it can be passed to analysis.run_quantum_chemistry_calculations
    sv = Statevector(amps)

    # Load minimal metadata to rebuild molecule and RHF with PySCF
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    geometry = meta.get("geometry")
    basis = meta.get("basis", "sto-3g")

    # Rebuild molecule (assume Angstrom units)
    mol = gto.M(atom=geometry, basis=basis, unit="Angstrom", verbose=0)
    rhf = scf.RHF(mol).run()

    # Bond length: use metadata if present; otherwise compute for diatomics
    bond_length = meta.get("bond_length")
    if bond_length is None and mol.natm == 2:
        coords = mol.atom_coords(unit="Angstrom")
        bond_length = float(np.linalg.norm(coords[1] - coords[0]))

    # Run downstream analysis using the provided Statevector
    qc_results = run_quantum_chemistry_calculations(mol, rhf, bond_length=bond_length, statevector=sv)

    # Calculate convergence data
    conv_results = analysis.calc_convergence_data(qc_results, spin_symm=True)

    # Save data to CSV
    analysis.save_convergence_data(data_dir, qc_results, conv_results)

    # Create plots
    plot.energy_vs_samples(data_dir, qc_results, conv_results, ylog=True)
    plot.convergence_comparison(data_dir, qc_results, conv_results, ylog=True)

    # Minimal output summary
    np.set_printoptions(precision=8, suppress=True)
    print("Statevector length:", len(sv.data))
    print("RHF energy:", float(rhf.e_tot))
    print("UHF energy:", float(qc_results.uhf.e_tot))
    print("FCI energy:", float(qc_results.fci_energy))


if __name__ == "__main__":
    main()
