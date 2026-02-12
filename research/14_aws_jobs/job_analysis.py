from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from qiskit.quantum_info import Statevector
from pyscf import gto, scf

from sd_qsci.analysis import run_quantum_chemistry_calculations, calc_convergence_data, save_convergence_data
from sd_qsci import analysis, plot


FILTER_PARTICLE_NUMBER = True


def filter_particle_number_counts(run_dir: Path, counts: dict[str, int], meta: dict) -> dict[str, int]:
    """Filter counts to only include bitstrings with the correct Hamming weight.
    
    Determines the target Hamming weight from metadata (H4 -> 4, H6 -> 6).
    Saves filtered counts to correct_particle_number_counts.json.
    """
    mol_name = meta.get("molecule") or meta.get("geometry_name")
    
    if not mol_name:
        # Fallback to _build_title_prefix logic or just skip if cannot determine
        print(f"[warning] Could not determine molecule name for {run_dir}. Skipping filtering.")
        return counts

    target_weight = None
    if "H4" in mol_name.upper():
        target_weight = 4
    elif "H6" in mol_name.upper():
        target_weight = 6
    else:
        print(f"[warning] Unknown molecule {mol_name} for {run_dir}. Skipping filtering.")
        return counts

    # Collect particle number counts
    particle_counts = {}
    for bs, count in counts.items():
        weight = sum(int(bit) for bit in bs)
        particle_counts[weight] = particle_counts.get(weight, 0) + count

    # Save particle number counts
    stats_path = run_dir / "particle_number_counts.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        # Convert keys to strings for JSON
        json.dump({str(k): v for k, v in sorted(particle_counts.items())}, f, indent=2)

    filtered_counts = {
        bs: count for bs, count in counts.items() 
        if sum(int(bit) for bit in bs) == target_weight
    }

    output_path = run_dir / "correct_particle_number_counts.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_counts, f, indent=2, sort_keys=True)
    
    print(f"[process] Saved particle counts to {stats_path}")
    print(f"[process] Saved filtered counts to {output_path}")
    return filtered_counts


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

    # Plot statevector amplitudes
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

    print(f"[success] Analysis completed for {run_dir}")

def analyze_all():
    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / "data"
    
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
    analyze_all()
