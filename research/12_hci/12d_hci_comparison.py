"""
12d: Method comparison on H6 @ 2.0 Å
------------------------------------

Compare multiple approaches on an H6 chain at 2.0 Å (STO-3G):
- HCI (Heat-bath CI) using src/comparison/hci.run_hci
- FCI-guided subspace energies (largest-amplitude FCI configurations)
- LUCJ with 10 layers (spin-symmetric QSCI subspace energies)
- UHF orbital-rotation circuit with spin symmetry recovery (spin-symmetric QSCI)

Plot absolute energy error vs FCI against subspace size for all methods on one
figure with:
- log-scale y-axis, lower bound 1e-5 Ha
- horizontal dashed lines for RHF and UHF absolute errors
- chemical accuracy region (≤ 1.6e-3 Ha) shaded with diagonal green hatch

LUCJ convergence data are saved to CSV and reused on subsequent runs to save time.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from pyscf import gto, scf

from sd_qsci import analysis, hamiltonian
from sd_qsci.utils import uhf_from_rhf
from src.comparison.hci import run_hci


# Settings
RECOMPUTE_LUCJ: bool = True  # if False, reuse saved LUCJ CSV
BOND_LENGTH: float = 2.0
N_ATOMS: int = 6
MAX_SUBSPACE: int = 400  # x-axis extent cap

# Theme
sns.set_theme()


def build_h_chain(bond_length: float, n_atoms: int = 6) -> gto.Mole:
    coords = [(i * bond_length, 0.0, 0.0) for i in range(n_atoms)]
    geometry = '; '.join([f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in coords])
    mol = gto.Mole()
    mol.build(atom=geometry, unit='Angstrom', basis='sto-3g', charge=0, spin=0, verbose=0)
    return mol


def main():
    bond_length = BOND_LENGTH
    n_atoms = N_ATOMS

    out_dir = Path(__file__).parent / 'data' / '12d_hci_comparison' / f"bond_length_{bond_length:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV outputs
    lucj_csv = out_dir / 'lucj_layers10_convergence.csv'
    fci_sub_csv = out_dir / 'fci_subspace_energy.csv'
    hci_csv = out_dir / 'hci_convergence.csv'

    # Build system and references
    print(f"Building H{n_atoms} chain at {bond_length:.2f} Å (STO-3G)…")
    mol = build_h_chain(bond_length, n_atoms)
    rhf = scf.RHF(mol).run()
    uhf = uhf_from_rhf(mol, rhf)
    fci_energy, n_fci_configs, fci_vec = analysis.calc_fci_energy(rhf)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)

    # -----------------------------
    # LUCJ (10 layers), spin-symmetric QSCI
    # -----------------------------
    if RECOMPUTE_LUCJ or not lucj_csv.exists():
        print("Computing LUCJ (10 layers) convergence…")
        # Reuse analysis.run_quantum_chemistry_calculations to get statevector etc.
        qc_results = analysis.run_quantum_chemistry_calculations(mol, rhf, bond_length)
        # calc_convergence_data with spin symmetry
        conv = analysis.calc_convergence_data(qc_results, spin_symm=True)

        # Save a filtered convergence CSV for convenience
        df = conv.df.copy()
        df['bond_length'] = bond_length
        df['fci_energy'] = fci_energy
        df['rhf_energy'] = rhf.e_tot
        df['uhf_energy'] = uhf.e_tot
        # Keep relevant columns if present
        keep_cols = [
            'bond_length', 'subspace_size', 'qsci_energy', 'spin_symm_energy',
            'fci_subspace_energy', 'fci_energy', 'rhf_energy', 'uhf_energy'
        ]
        df = df[[c for c in keep_cols if c in df.columns]]
        df.to_csv(lucj_csv, index=False)
    else:
        print("Reusing existing LUCJ convergence CSV.")

    # -----------------------------
    # FCI-guided subspace energies
    # -----------------------------
    # Determine sizes to compute up to
    fci_max = min(MAX_SUBSPACE, int(np.count_nonzero(np.abs(fci_vec) > 0)))
    if fci_max < 1:
        fci_max = min(MAX_SUBSPACE, 1 << int(np.log2(fci_vec.size)))
    fci_sizes = np.arange(1, fci_max + 1, dtype=int)

    fci_df = pd.DataFrame()
    if fci_sub_csv.exists():
        try:
            fci_df = pd.read_csv(fci_sub_csv)
        except Exception:
            fci_df = pd.DataFrame()

    need_compute = True
    if not fci_df.empty:
        have_sizes = set(int(x) for x in fci_df.get('subspace_size', []))
        if all(int(k) in have_sizes for k in fci_sizes):
            need_compute = False
    if need_compute:
        print("Computing FCI-guided subspace energies…")
        energies = [analysis.calc_fci_subspace_energy(H, fci_vec, int(k)) for k in fci_sizes]
        fci_df = pd.DataFrame({
            'subspace_size': fci_sizes,
            'fci_subspace_energy': np.asarray(energies, dtype=float),
            'fci_energy': float(fci_energy),
        })
        fci_df.to_csv(fci_sub_csv, index=False)

    # -----------------------------
    # HCI curve via run_hci
    # -----------------------------
    print("Running HCI (heat-bath CI)…")
    hci_res = run_hci(mol, eps=1.0e-4)
    hci_series: List[dict] = list(hci_res["series"])  # type: ignore
    hci_df = pd.DataFrame(hci_series)
    hci_df.rename(columns={'ndeterminants': 'subspace_size', 'energy_ha': 'energy'}, inplace=True)
    hci_df['fci_energy'] = fci_energy
    hci_df['abs_error'] = np.abs(hci_df['energy'] - fci_energy)
    hci_df.to_csv(hci_csv, index=False)

    # -----------------------------
    # UHF rotation circuit with spin symmetry recovery
    # -----------------------------
    # Use the same qc_results logic as in 08_spin_recovery via analysis API
    qc_results_sr = analysis.run_quantum_chemistry_calculations(mol, rhf, bond_length)
    conv_sr = analysis.calc_convergence_data(qc_results_sr, spin_symm=True)
    sr_df = conv_sr.df.copy()
    sr_df['abs_error'] = np.abs(sr_df['spin_symm_energy'] - fci_energy)

    # -----------------------------
    # Plot combined comparison
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_yscale('log')
    ax.set_xlabel('Subspace size (number of configurations)')
    ax.set_ylabel('Absolute energy error vs FCI (Ha)')
    ax.set_title(f'H6 @ {bond_length:.2f} Å — Methods comparison (error vs subspace)')

    # Chemical accuracy region (≤ 1.6e-3 Ha) with diagonal green hatch over light grey
    y_lo = 1.0e-4
    chem_acc = 1.6e-3
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    bg = patches.Rectangle((0.0, y_lo), 1.0, chem_acc - y_lo, transform=trans,
                           facecolor='#D0D0D0', edgecolor='none', alpha=0.35, zorder=0)
    ax.add_patch(bg)
    hat = patches.Rectangle((0.0, y_lo), 1.0, chem_acc - y_lo, transform=trans,
                            facecolor='none', edgecolor='#2ca02c', hatch='///', linewidth=0.0, zorder=0)
    ax.add_patch(hat)
    proxy = patches.Rectangle((0, 0), 1, 1, facecolor='#D0D0D0', edgecolor='#2ca02c',
                              hatch='///', alpha=0.35, label='chemical accuracy')
    ax.add_artist(proxy)

    # Prepare and plot series
    ymax_values: List[float] = []

    # LUCJ (10 layers) — spin-symmetric series
    try:
        lucj_df = pd.read_csv(lucj_csv)
        if 'spin_symm_energy' in lucj_df.columns:
            x = lucj_df['subspace_size'].to_numpy()
            y = np.abs(lucj_df['spin_symm_energy'].to_numpy() - float(lucj_df['fci_energy'].iloc[0]))
            y = np.maximum(y, y_lo)
            # Clip to MAX_SUBSPACE on x
            mask = (x >= 1) & (x <= MAX_SUBSPACE)
            ax.plot(x[mask], y[mask], 'o-', linewidth=1.8, markersize=3.5, label='LUCJ (10 layers, spin-symm)')
            if y[mask].size:
                ymax_values.append(float(np.nanmax(y[mask])))
    except Exception:
        pass

    # FCI-guided subspace
    if not fci_df.empty:
        x_fci = fci_df['subspace_size'].to_numpy()
        y_fci = np.abs(fci_df['fci_subspace_energy'].to_numpy() - fci_df['fci_energy'].to_numpy())
        y_fci = np.maximum(y_fci, y_lo)
        mask_f = (x_fci >= 1) & (x_fci <= MAX_SUBSPACE)
        ax.plot(x_fci[mask_f], y_fci[mask_f], 's-', color='black', linewidth=1.8, markersize=3.5, label='FCI subspace (error)')
        if y_fci[mask_f].size:
            ymax_values.append(float(np.nanmax(y_fci[mask_f])))

    # HCI curve
    if not hci_df.empty:
        x_hci = hci_df['subspace_size'].to_numpy()
        y_hci = np.maximum(hci_df['abs_error'].to_numpy(), y_lo)
        mask_h = (x_hci >= 1) & (x_hci <= MAX_SUBSPACE)
        ax.plot(x_hci[mask_h], y_hci[mask_h], 'd--', linewidth=1.6, markersize=3.5, label='HCI (error)')
        if y_hci[mask_h].size:
            ymax_values.append(float(np.nanmax(y_hci[mask_h])))

    # UHF rotation + spin symmetry recovery series
    if not sr_df.empty and 'subspace_size' in sr_df.columns and 'abs_error' in sr_df.columns:
        x_sr = sr_df['subspace_size'].to_numpy()
        y_sr = np.maximum(sr_df['abs_error'].to_numpy(), y_lo)
        mask_sr = (x_sr >= 1) & (x_sr <= MAX_SUBSPACE)
        ax.plot(x_sr[mask_sr], y_sr[mask_sr], '^-', linewidth=1.6, markersize=3.5, label='UHF rotation (spin-symm)')
        if y_sr[mask_sr].size:
            ymax_values.append(float(np.nanmax(y_sr[mask_sr])))

    # RHF and UHF horizontal dashed lines (absolute error levels)
    rhf_err = max(abs(rhf.e_tot - fci_energy), y_lo)
    uhf_err = max(abs(uhf.e_tot - fci_energy), y_lo)
    ax.axhline(y=rhf_err, linestyle='--', linewidth=1.6, color='#1f77b4', label='RHF |ΔE|')
    ax.axhline(y=uhf_err, linestyle='--', linewidth=1.6, color='#2ca02c', label='UHF |ΔE|')
    ymax_values.extend([rhf_err, uhf_err])

    # Bounds
    ax.set_xlim(left=1, right=MAX_SUBSPACE)
    try:
        if ymax_values:
            y_max = max(ymax_values)
            headroom = 1.25
            ax.set_ylim(bottom=y_lo, top=max(y_lo * 1.1, y_max * headroom))
        else:
            ax.set_ylim(bottom=y_lo)
    except Exception:
        pass

    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_path = out_dir / 'h6_method_comparison_error_vs_subspace.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved data and plot under: {out_dir}")


if __name__ == '__main__':
    main()
