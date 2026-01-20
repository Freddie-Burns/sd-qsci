from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from pyscf import gto, scf

from sd_qsci.analysis import run_quantum_chemistry_calculations
from sd_qsci import analysis, plot


# Minimal, hard-coded configuration. Change these for each run.
# You can pass a single tag as a string or multiple tags as a list of strings.
# When multiple tags are provided, measurement counts are combined (summed)
# and analysis is performed on the aggregated distribution.
DATE_TAGS: list[str] | str = [
    "20260119-170454",
    "20260120-104429",
    "20260120-104452",
    "20260120-104508",
    "20260120-104531",
]


def main():
    base_dir = Path(__file__).resolve().parent
    # Normalize to list of tags
    if isinstance(DATE_TAGS, str):
        tags = [DATE_TAGS]
    else:
        tags = list(DATE_TAGS)

    # Output directory: if multiple tags, store under a combined folder name
    if len(tags) == 1:
        out_tag = tags[0]
    else:
        # Keep it readable while unique
        out_tag = "combined__" + "__".join(tags)

    data_dir = base_dir / "data" / out_tag
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load counts from all tags and sum them
    per_tag_counts: list[dict[str, int]] = []
    max_key_len = 0
    meta = None
    for i, tag in enumerate(tags):
        in_path = base_dir / "data" / tag / "measurement_counts.json"
        meta_path = base_dir / "data" / tag / "metadata.json"

        with in_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        counts_i = payload.get("measurement_counts", payload)
        # Ensure values are ints and track maximum key length
        clean_counts_i: dict[str, int] = {}
        for k, v in counts_i.items():
            ks = str(k)
            max_key_len = max(max_key_len, len(ks))
            try:
                clean_counts_i[ks] = int(v)
            except Exception:
                # Fall back to float->int if needed
                clean_counts_i[ks] = int(float(v))
        per_tag_counts.append(clean_counts_i)

        # Load metadata from the first tag only (assumed identical across tags)
        if i == 0:
            with meta_path.open("r", encoding="utf-8") as mf:
                meta = json.load(mf)

    assert meta is not None, "No metadata loaded; check DATE_TAGS paths."

    # Combine and normalize keys to the global max bit-length
    combined: dict[str, int] = {}
    for counts_i in per_tag_counts:
        for k, v in counts_i.items():
            k_pad = k.zfill(max_key_len)
            combined[k_pad] = combined.get(k_pad, 0) + int(v)
    counts = combined

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
