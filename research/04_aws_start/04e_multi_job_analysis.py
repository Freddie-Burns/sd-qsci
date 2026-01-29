from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from pyscf import gto, scf

from sd_qsci.analysis import run_quantum_chemistry_calculations
from sd_qsci import analysis, plot


# Minimal, hard-coded configuration for combining MULTIPLE date tags only.
# 04d aggregates counts across multiple tags; for single-tag analysis use 04d_job_analysis.py
DATE_TAGS: list[str] = [
    "20260119-170454",
    "20260120-104429",
    "20260120-104452",
    "20260120-104508",
    "20260120-104531",
]
REORDER_BITSTRINGS = True

def main():
    base_dir = Path(__file__).resolve().parent
    out_dir = get_data_dir()

    # Load counts from all DATE_TAGS and sum them
    per_tag_counts: list[dict[str, int]] = []
    for i, tag in enumerate(DATE_TAGS):
        in_path = base_dir / "data" / tag / "measurement_counts.json"

        with in_path.open("r", encoding="utf-8") as f:
            counts_i = json.load(f)
        per_tag_counts.append(counts_i)

    # Load metadata from the first tag only (assumed identical across DATE_TAGS)
    meta_path = base_dir / "data" / DATE_TAGS[0] / "metadata.json"
    with meta_path.open("r", encoding="utf-8") as mf:
        meta = json.load(mf)

    # Combine counts directly (keys are guaranteed to have the same length)
    combined: dict[str, int] = {}
    for counts_i in per_tag_counts:
        for k, v in counts_i.items():
            combined[k] = combined.get(k, 0) + int(v)
    counts = combined

    # Save combined counts and top-10 counts into the output directory
    # (for multiple DATE_TAGS this will be the combined_* folder)
    out_counts_path = out_dir / "measurement_counts.json"
    with out_counts_path.open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2, sort_keys=True)

    # Top-10 by count (descending)
    top10_items = sorted(counts.items(), key=lambda kv: int(kv[1]), reverse=True)[:10]
    top10_counts = {k: int(v) for k, v in top10_items}
    out_top10_path = out_dir / "measurement_counts_top10.json"
    with out_top10_path.open("w", encoding="utf-8") as f:
        json.dump(top10_counts, f, indent=2, sort_keys=True)

    # Infer qubit count and construct ordered basis
    n = max(len(k) for k in counts.keys())
    order = [format(i, f"0{n}b") for i in range(2**n)]

    # reorder qubits
    if REORDER_BITSTRINGS:
        tag_path = base_dir / "data" / DATE_TAGS[0]
        tqc_path = tag_path / "qiskit_circuit_transpiled.qpy"
        with open(tqc_path, 'rb') as f:
            qc = qpy.load(f)[0]
            layout = qc.layout

        index_map = []
        reordered_counts = {}
        for i in range(n):
            index_map.append(layout.initial_layout._p2v[i]._index)

        for bitstring, count in list(counts.items()):
            bitstring = bitstring[::-1]  # reverse to match index_map ordering
            bitstring_index_pairs = zip(bitstring, index_map)
            sorted_pairs = sorted(bitstring_index_pairs, key=lambda x: x[1])
            new_bitstring = ''.join([b for b, i in sorted_pairs])
            new_bitstring = new_bitstring[
                ::-1]  # reverse back to original order
            reordered_counts[new_bitstring] = count

        counts = reordered_counts

        # Also write back the counts
        out_counts_path = out_dir / "reordered_measurement_counts.json"
        with out_counts_path.open("w", encoding="utf-8") as f:
            json.dump(counts, f, indent=2, sort_keys=True)

        # Top-10 by count (descending)
        top10_items = sorted(counts.items(), key=lambda kv: int(kv[1]),
                             reverse=True)[:10]
        top10_counts = {k: int(v) for k, v in top10_items}
        out_top10_path = out_dir / "reordered_measurement_counts_top10.json"
        with out_top10_path.open("w", encoding="utf-8") as f:
            json.dump(top10_counts, f, indent=2, sort_keys=True)

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
    analysis.save_convergence_data(out_dir, qc_results, conv_results)

    # Build a title prefix from metadata for plot titles
    title_prefix = _build_title_prefix(meta, bond_length=bond_length, basis=basis)

    # Create plots (include title prefix with molecule/device info)
    plot.energy_vs_samples(out_dir, qc_results, conv_results, title_prefix=title_prefix, ylog=True)
    plot.convergence_comparison(out_dir, qc_results, conv_results, title_prefix=title_prefix, ylog=True)

    # Plot statevector amplitudes as a bar graph (and full log plot),
    # reusing the helper from src/sd_qsci/plot.py as in 08_spin_recovery
    plot.statevector_coefficients(
        qc_results.sv.data,
        qc_results.fci_vec,
        out_dir,
        n_top=20,
        ylog=False,
        title=f"{title_prefix} — Top 20 Configuration Coefficients" if title_prefix else None,
        include_spin_recovered=True,
        qsci_label='Counts SV',
    )

    # If available, also plot the simulated statevector amplitudes from one of the tags (use the first)
    sim_sv_path = base_dir / "data" / DATE_TAGS[0] / "sim_statevector.json"
    if sim_sv_path.exists():
        try:
            with sim_sv_path.open("r", encoding="utf-8") as f:
                sim_data = json.load(f)

            # Build simulated statevector as complex numpy array matching the counts dimension
            dim = 2 ** n
            sim_vec = np.zeros(dim, dtype=complex)
            amps_list = sim_data.get("amplitudes", [])
            for item in amps_list:
                idx = int(item.get("index", -1))
                if 0 <= idx < dim:
                    re = float(item.get("real", 0.0))
                    im = float(item.get("imag", 0.0))
                    sim_vec[idx] = re + 1j * im

            # Create a combined plot (FCI, Counts SV, Counts SV (Spin), Sim SV, Sim SV (Spin))
            # and save it to the standard filenames in out_dir, overwriting the counts-only plot.
            plot.statevector_coefficients_counts_vs_sim(
                qc_results.sv.data,
                sim_vec,
                qc_results.fci_vec,
                out_dir,
                n_top=20,
                ylog=False,
                include_spin_recovered=True,
                title_prefix=title_prefix,
            )
        except Exception as e:
            print(f"Warning: Failed to load or plot simulated statevector for multi-job analysis: {e}")

    # Minimal output summary
    np.set_printoptions(precision=8, suppress=True)
    print("Statevector length:", len(sv.data))
    print("RHF energy:", float(rhf.e_tot))
    print("UHF energy:", float(qc_results.uhf.e_tot))
    print("FCI energy:", float(qc_results.fci_energy))


def get_data_dir():
    """
    Output directory: always store under a combined folder name for multiple tags.
    """
    base_dir = Path(__file__).resolve().parent
    out_tag = "combined_" + "_".join(DATE_TAGS)
    data_dir = base_dir / "data" / out_tag
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _build_title_prefix(meta: dict, *, bond_length: float | None, basis: str | None) -> str | None:
    """Construct a descriptive title prefix using molecule and device info from metadata.

    Tries multiple metadata schemas gracefully. Returns None if nothing meaningful found.
    """
    # Molecule name
    mol_name = meta.get("molecule") or meta.get("geometry_name")
    if not mol_name:
        geom = meta.get("geometry")
        symbols: list[str] = []
        if isinstance(geom, (list, tuple)):
            for item in geom:
                if isinstance(item, (list, tuple)) and item:
                    sym = item[0] if isinstance(item[0], str) else None
                    if sym:
                        symbols.append(sym)
                elif isinstance(item, dict):
                    sym = item.get("element") or item.get("symbol")
                    if isinstance(sym, str):
                        symbols.append(sym)
        if symbols:
            from collections import Counter
            cnt = Counter(symbols)
            # Compose a simple chemical formula like H2O (alphabetical order)
            mol_name = "".join(f"{el}{cnt[el] if cnt[el] > 1 else ''}" for el in sorted(cnt.keys()))
    if not mol_name:
        mol_name = "Molecule"

    # Bond length (Å)
    bl = bond_length if bond_length is not None else meta.get("bond_length_angstrom")

    # Basis
    basis_str = basis or meta.get("basis")

    # Device string
    device = meta.get("device")
    if not device:
        arn = meta.get("device_arn")
        if isinstance(arn, str) and "/" in arn:
            parts = arn.split("/")
            if len(parts) >= 2:
                provider = parts[-2]
                name = parts[-1]
                device = f"{str(provider).title()} {name}"
    if not device:
        provider = meta.get("provider")
        backend = meta.get("backend")
        if provider or backend:
            device = " ".join([p for p in [str(provider).title() if provider else None, backend] if p])

    parts: list[str] = []
    sub = []
    if bl is not None:
        try:
            sub.append(f"R = {float(bl):.2f} Å")
        except Exception:
            pass
    if basis_str:
        sub.append(str(basis_str))
    if sub:
        parts.append(f"{mol_name} ({', '.join(sub)})")
    else:
        parts.append(mol_name)
    if device:
        parts.append(f"Device: {device}")

    return " — ".join(parts) if any(parts) else None


if __name__ == "__main__":
    main()
