from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from pyscf import gto, scf

from sd_qsci.analysis import run_quantum_chemistry_calculations
from sd_qsci import analysis, plot


# Minimal, hard-coded configuration for a single job/date tag
DATE_TAG: str = "20260120-104531"


def main():
    base_dir = Path(__file__).resolve().parent
    out_dir = get_data_dir()

    # Load counts for the single DATE_TAG
    counts_path = base_dir / "data" / DATE_TAG / "measurement_counts.json"
    with counts_path.open("r", encoding="utf-8") as f:
        counts: dict[str, int] = json.load(f)

    # Also write back the counts (keeps output artifacts consistent with 04d)
    out_counts_path = out_dir / "measurement_counts.json"
    with out_counts_path.open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2, sort_keys=True)

    # Top-10 by count (descending)
    top10_items = sorted(counts.items(), key=lambda kv: int(kv[1]), reverse=True)[:10]
    top10_counts = {k: int(v) for k, v in top10_items}
    out_top10_path = out_dir / "measurement_counts_top10.json"
    with out_top10_path.open("w", encoding="utf-8") as f:
        json.dump(top10_counts, f, indent=2, sort_keys=True)

    # Load metadata for the single tag
    meta_path = base_dir / "data" / DATE_TAG / "metadata.json"
    with meta_path.open("r", encoding="utf-8") as mf:
        meta = json.load(mf)

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
    qc_results = run_quantum_chemistry_calculations(
        mol, rhf, bond_length=bond_length, statevector=sv
    )

    # Calculate convergence data
    conv_results = analysis.calc_convergence_data(qc_results, spin_symm=True)

    # Save data to CSV
    analysis.save_convergence_data(out_dir, qc_results, conv_results)

    # Create plots
    plot.energy_vs_samples(out_dir, qc_results, conv_results, ylog=True)
    plot.convergence_comparison(out_dir, qc_results, conv_results, ylog=True)

    # Plot statevector amplitudes as a bar graph (and full log plot)
    plot.statevector_coefficients(
        qc_results.sv.data,
        qc_results.fci_vec,
        out_dir,
        n_top=20,
        ylog=False,
        include_spin_recovered=True,
        qsci_label='Counts SV',
    )

    # Minimal output summary
    np.set_printoptions(precision=8, suppress=True)
    print("Statevector length:", len(sv.data))
    print("RHF energy:", float(rhf.e_tot))
    print("UHF energy:", float(qc_results.uhf.e_tot))
    print("FCI energy:", float(qc_results.fci_energy))


def get_data_dir():
    """
    Output directory: for a single DATE_TAG, store under that tag.
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / DATE_TAG
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


if __name__ == "__main__":
    main()
