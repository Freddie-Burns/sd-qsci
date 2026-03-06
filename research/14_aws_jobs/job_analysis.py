from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from qiskit.quantum_info import Statevector
from pyscf import gto, scf
import matplotlib.pyplot as plt
import seaborn as sns

from sd_qsci.analysis import run_quantum_chemistry_calculations, calc_convergence_data, save_convergence_data, spin_symm_amplitudes
from sd_qsci import analysis, plot
from filter_counts import filter_particle_number_counts


FILTER_PARTICLE_NUMBER = True
SUB_DIR = "14a_h4_ankaa"


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

def _plot_all_coefficients_horizontal(
    qsci_vec: np.ndarray,
    fci_vec: np.ndarray,
    run_dir: Path,
    title_prefix: str | None = None,
    threshold: float = 1e-6,
):
    """Plot all configurations horizontal, ordered by FCI amplitude."""
    fci_abs = np.abs(fci_vec)
    qsci_symm_vec = spin_symm_amplitudes(qsci_vec)
    
    # Filter configurations where at least one amplitude is above threshold
    significant_indices = np.where(
        (fci_abs > threshold) | (np.abs(qsci_vec) > threshold) | (np.abs(qsci_symm_vec) > threshold)
    )[0]
    
    if len(significant_indices) == 0:
        print(f"[skip] No significant configurations found for horizontal plot in {run_dir}")
        return

    # Sort significant indices by FCI amplitude (descending)
    sorted_sig_indices = significant_indices[np.argsort(fci_abs[significant_indices])[::-1]]
    
    qsci_coefs = np.abs(qsci_vec[sorted_sig_indices])
    qsci_symm_coefs = np.abs(qsci_symm_vec[sorted_sig_indices])
    fci_coefs = fci_abs[sorted_sig_indices]
    
    n_qubits = int(np.log2(len(fci_vec))) if len(fci_vec) > 0 else 0
    bitstring_labels = [format(i, f"0{n_qubits}b") for i in sorted_sig_indices]
    occupation_labels = [plot._occupation_vector(bs) for bs in bitstring_labels]
    
    # Create the plot
    sns.set_style("whitegrid")
    # Height proportional to number of configurations to keep it "thin and long"
    fig_height = max(8, len(sorted_sig_indices) * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    y = np.arange(len(sorted_sig_indices))
    height = 0.28
    
    ax.barh(y + height, fci_coefs, height, label='FCI', color='green', alpha=0.8)
    ax.barh(y, qsci_coefs, height, label='Counts SV', color='purple', alpha=0.8)
    ax.barh(y - height, qsci_symm_coefs, height, label='Counts SV (Spin Recovered)', color='#D55E00', alpha=0.8)
    
    ax.set_yticks(y)
    ax.set_yticklabels(occupation_labels, fontsize=8)
    ax.invert_yaxis()  # Largest FCI at the top
    
    ax.set_xlabel('|Coefficient|', fontsize=12)
    ax.set_ylabel('Electron configuration', fontsize=12)
    
    title = "All Configuration Amplitudes (ordered by FCI)"
    if title_prefix:
        title = f"{title_prefix} — {title}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = run_dir / 'statevector_coefficients_horizontal_all.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[info] Horizontal plot saved to {out_path}")


def analyze_run(run_dir: Path):
    print(f"\n[process] Analyzing {run_dir}...")
    
    # Try combined_counts.json first, then measurement_counts.json
    counts_path = run_dir / "combined_counts.json"
    if not counts_path.exists():
        counts_path = run_dir / "measurement_counts.json"

    meta_path = run_dir / "metadata.json"
    job_meta_path = run_dir / "job_metadata.json"

    if not counts_path.exists():
        print(f"[skip] No counts file found in {run_dir}")
        return

    # Load counts
    # Try loading existing filtered counts first (new standard name: filtered_counts.json)
    # Then try legacy filtered counts (correct_particle_number_counts.json)
    # Finally, fall back to raw counts (combined_counts.json or measurement_counts.json)
    counts_path_filtered = run_dir / "filtered_counts.json"
    counts_path_legacy = run_dir / "correct_particle_number_counts.json"
    
    if counts_path_filtered.exists():
        print(f"[info] Using existing filtered counts from {counts_path_filtered}")
        with open(counts_path_filtered, "r", encoding="utf-8") as f:
            counts = json.load(f)
    elif counts_path_legacy.exists():
        print(f"[info] Using existing filtered counts from {counts_path_legacy}")
        with open(counts_path_legacy, "r", encoding="utf-8") as f:
            counts = json.load(f)
    else:
        with open(counts_path, "r", encoding="utf-8") as f:
            counts = json.load(f)

    # Load metadata (could be in metadata.json or job_metadata.json)
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta.update(json.load(f))
    if job_meta_path.exists():
        with open(job_meta_path, "r", encoding="utf-8") as f:
            meta.update(json.load(f))

    # Filter bitstrings by particle number
    if FILTER_PARTICLE_NUMBER:
        counts = filter_particle_number_counts(run_dir, counts, meta)

    # Infer qubit count and construct ordered basis
    n = max(len(k) for k in counts.keys())
    order = [format(i, f"0{n}b") for i in range(2**n)]

    total = sum(int(counts.get(bs, 0)) for bs in order)
    if total == 0:
        print(f"[error] Total counts is zero for {run_dir}")
        return
        
    probs = np.array([int(counts.get(bs, 0)) / total for bs in order], dtype=float)

    # Convert probabilities to amplitude vector (non-negative, no phases) and L2-normalize
    amps = np.sqrt(probs).astype(complex)
    norm = np.linalg.norm(amps)
    amps = amps / norm

    # Build a Qiskit Statevector
    sv = Statevector(amps)

    geometry = meta.get("geometry")
    basis = meta.get("basis", "sto-3g")

    if not geometry:
        print(f"[error] No geometry found in metadata for {run_dir}")
        return

    # Rebuild molecule
    mol = gto.M(atom=geometry, basis=basis, unit="Angstrom", verbose=0)
    rhf = scf.RHF(mol).run()

    # Bond length
    bond_length = meta.get("bond_length")
    if bond_length is None and mol.natm == 2:
        coords = mol.atom_coords(unit="Angstrom")
        bond_length = float(np.linalg.norm(coords[1] - coords[0]))

    # Run downstream analysis
    try:
        qc_results = run_quantum_chemistry_calculations(
            mol, rhf, bond_length=bond_length, statevector=sv
        )
    except Exception as e:
        print(f"[error] run_quantum_chemistry_calculations failed for {run_dir}: {e}")
        return

    # Calculate convergence data
    try:
        conv_results = calc_convergence_data(qc_results, spin_symm=True)
    except Exception as e:
        print(f"[error] calc_convergence_data failed for {run_dir}: {e}")
        return

    # Save data to CSV
    save_convergence_data(run_dir, qc_results, conv_results)

    # Build title prefix
    title_prefix = _build_title_prefix(meta, bond_length=bond_length, basis=basis)

    # Create plots
    plot.energy_vs_samples(run_dir, qc_results, conv_results, title_prefix=title_prefix, ylog=True)
    plot.convergence_comparison(run_dir, qc_results, conv_results, title_prefix=title_prefix, ylog=True)

    # Plot statevector amplitudes (top configurations)
    plot.statevector_coefficients(
        qc_results.sv.data,
        qc_results.fci_vec,
        run_dir,
        n_top=20,
        ylog=False,
        title=f"{title_prefix} — Top 20 Configuration Coefficients" if title_prefix else None,
        include_spin_recovered=True,
        qsci_label='Counts SV',
    )

    # Plot statevector amplitudes (bottom configurations)
    plot.statevector_coefficients(
        qc_results.sv.data,
        qc_results.fci_vec,
        run_dir,
        n_top=20,
        ylog=False,
        title=f"{title_prefix} — Bottom 20 Configuration Coefficients" if title_prefix else None,
        include_spin_recovered=True,
        qsci_label='Counts SV',
        filename='statevector_coefficients_rev.png',
        order='ascending',
        include_full=False,
    )

    # If available, also plot the simulated statevector amplitudes
    sim_sv_path = run_dir / "sim_statevector.json"
    if sim_sv_path.exists():
        try:
            with sim_sv_path.open("r", encoding="utf-8") as f:
                sim_data = json.load(f)

            dim = 2 ** n
            sim_vec = np.zeros(dim, dtype=complex)
            amps_list = sim_data.get("amplitudes", [])
            for item in amps_list:
                idx = int(item.get("index", -1))
                if 0 <= idx < dim:
                    re = float(item.get("real", 0.0))
                    im = float(item.get("imag", 0.0))
                    sim_vec[idx] = re + 1j * im

            plot.statevector_coefficients_counts_vs_sim(
                qc_results.sv.data,
                sim_vec,
                qc_results.fci_vec,
                run_dir,
                n_top=20,
                ylog=False,
                include_spin_recovered=True,
                title_prefix=title_prefix,
            )
        except Exception as e:
            print(f"Warning: Failed to load or plot simulated statevector: {e}")

    # Plot all configurations horizontal (ordered by FCI)
    _plot_all_coefficients_horizontal(
        qc_results.sv.data,
        qc_results.fci_vec,
        run_dir,
        title_prefix=title_prefix,
    )

    print(f"[success] Analysis completed for {run_dir}")

def analyze_all(sub_dir: str | None = None):
    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / "data"
    
    if sub_dir:
        data_root = data_root / sub_dir

    if not data_root.exists():
        print(f"[info] Data directory not found: {data_root}")
        return

    # Find all directories that contain either measurement_counts.json or combined_counts.json
    print(f"[process] Searching for run directories in {data_root}...")
    
    # We use a set to avoid duplicates if both files exist in the same directory
    run_dirs = set()
    for pattern in ["**/measurement_counts.json", "**/combined_counts.json"]:
        for p in data_root.rglob(pattern):
            # Check if this is a "batch" directory (timestamped directory within another timestamped directory)
            # We assume a timestamped directory matches a specific pattern, or more simply,
            # we check if its parent is also a directory that looks like a run directory.
            # Usually, the structure is data/molecule/timestamp/counts.json
            # A batch might be data/molecule/timestamp/batch_timestamp/counts.json
            
            # Simple heuristic: if the path from data_root has more than 2 components, 
            # and the last two components both look like run/timestamp dirs, it's a batch.
            # For now, let's just count components. 
            # data_root / molecule / timestamp -> 2 components relative to data_root
            # data_root / molecule / timestamp / batch -> 3 components
            
            rel_path = p.parent.relative_to(data_root)
            if len(rel_path.parts) > 2:
                # This is likely a batch or sub-directory we want to skip
                continue
                
            run_dirs.add(p.parent)
    
    sorted_run_dirs = sorted(list(run_dirs))
    total_runs = len(sorted_run_dirs)
    
    if total_runs == 0:
        print("[info] No directories with counts files found.")
        return

    print(f"[info] Found {total_runs} directories to analyze.")

    for i, run_dir in enumerate(sorted_run_dirs, 1):
        print(f"\n[{i}/{total_runs}] Processing: {run_dir}")
        try:
            analyze_run(run_dir)
        except Exception as e:
            print(f"[error] Failed to analyze {run_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    analyze_all(sub_dir=SUB_DIR)
