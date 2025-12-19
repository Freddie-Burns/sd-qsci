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
from pyscf.cc import CCSD
from qiskit_aer import Aer

from sd_qsci import analysis, circuit, hamiltonian
from sd_qsci.utils import uhf_from_rhf
from src.comparison.hci import run_hci
from src.comparison.greedy import greedy_from_results


# Settings
RECOMPUTE_LUCJ: bool = False  # if False, reuse saved LUCJ CSV
BOND_LENGTH: float = 2.0
N_ATOMS: int = 6
MAX_SUBSPACE: int = 400  # x-axis extent

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
    lucj10_csv = out_dir / 'lucj_layers10_convergence.csv'
    lucj1_csv = out_dir / 'lucj_layers1_convergence.csv'
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
    # LUCJ (10 layers) and LUCJ (1 layer), spin-symmetric QSCI
    # -----------------------------

    # Compute/cache LUCJ (10 layers)
    if RECOMPUTE_LUCJ or not lucj10_csv.exists():
        print("Computing LUCJ (10 layers) convergence…")
        ccsd = CCSD(rhf).run()
        backend = Aer.get_backend("statevector_simulator")
        lucj_qc_10 = circuit.get_lucj_circuit(ccsd_obj=ccsd, backend=backend, n_reps=10)
        one_q_gates, two_q_gates = analysis.count_gates(lucj_qc_10)
        lucj_sv_10 = circuit.simulate(lucj_qc_10)
        lucj_spin_symm_10 = analysis.spin_symm_amplitudes(lucj_sv_10.data)

        qc_results_lucj_10 = analysis.QuantumChemistryResults(
            mol=mol,
            rhf=rhf,
            uhf=uhf,
            sv=lucj_sv_10,
            H=H,
            fci_energy=fci_energy,
            n_fci_configs=n_fci_configs,
            fci_vec=fci_vec,
            bond_length=bond_length,
            spin_symm_amp=lucj_spin_symm_10,
            one_q_gates=one_q_gates,
            two_q_gates=two_q_gates,
        )

        conv10 = analysis.calc_convergence_data(qc_results_lucj_10, spin_symm=True)
        df10 = conv10.df.copy()
        df10['bond_length'] = bond_length
        df10['fci_energy'] = fci_energy
        df10['rhf_energy'] = rhf.e_tot
        df10['uhf_energy'] = uhf.e_tot
        keep_cols = [
            'bond_length', 'subspace_size', 'qsci_energy', 'spin_symm_energy',
            'fci_subspace_energy', 'fci_energy', 'rhf_energy', 'uhf_energy'
        ]
        df10 = df10[[c for c in keep_cols if c in df10.columns]]
        df10.to_csv(lucj10_csv, index=False)
    else:
        print("Reusing existing LUCJ (10 layers) convergence CSV.")

    # Compute/cache LUCJ (1 layer)
    if RECOMPUTE_LUCJ or not lucj1_csv.exists():
        print("Computing LUCJ (1 layer) convergence…")
        ccsd = CCSD(rhf).run()
        backend = Aer.get_backend("statevector_simulator")
        lucj_qc_1 = circuit.get_lucj_circuit(ccsd_obj=ccsd, backend=backend, n_reps=1)
        one_q_gates, two_q_gates = analysis.count_gates(lucj_qc_1)
        lucj_sv_1 = circuit.simulate(lucj_qc_1)
        lucj_spin_symm_1 = analysis.spin_symm_amplitudes(lucj_sv_1.data)

        qc_results_lucj_1 = analysis.QuantumChemistryResults(
            mol=mol,
            rhf=rhf,
            uhf=uhf,
            sv=lucj_sv_1,
            H=H,
            fci_energy=fci_energy,
            n_fci_configs=n_fci_configs,
            fci_vec=fci_vec,
            bond_length=bond_length,
            spin_symm_amp=lucj_spin_symm_1,
            one_q_gates=one_q_gates,
            two_q_gates=two_q_gates,
        )

        conv1 = analysis.calc_convergence_data(qc_results_lucj_1, spin_symm=True)
        df1 = conv1.df.copy()
        df1['bond_length'] = bond_length
        df1['fci_energy'] = fci_energy
        df1['rhf_energy'] = rhf.e_tot
        df1['uhf_energy'] = uhf.e_tot
        keep_cols = [
            'bond_length', 'subspace_size', 'qsci_energy', 'spin_symm_energy',
            'fci_subspace_energy', 'fci_energy', 'rhf_energy', 'uhf_energy'
        ]
        df1 = df1[[c for c in keep_cols if c in df1.columns]]
        df1.to_csv(lucj1_csv, index=False)
    else:
        print("Reusing existing LUCJ (1 layer) convergence CSV.")

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
    # Variational errors are non-negative: use signed difference without abs
    hci_df['qsci_spin_symm_error'] = (hci_df['energy'] - fci_energy)
    hci_df.to_csv(hci_csv, index=False)

    # -----------------------------
    # UHF rotation circuit with spin symmetry recovery
    # -----------------------------
    # Use the same qc_results logic as in 08_spin_recovery via analysis API
    qc_results_sr = analysis.run_quantum_chemistry_calculations(mol, rhf, bond_length)
    conv_sr = analysis.calc_convergence_data(qc_results_sr, spin_symm=True)
    sr_df = conv_sr.df.copy()
    # Use signed difference; should be ≥ 0 by variational principle
    sr_df['qsci_spin_symm_error'] = (sr_df['spin_symm_energy'] - fci_energy)
    sr_df['qsci_error'] = (sr_df['qsci_energy'] - fci_energy)

    # -----------------------------
    # Greedy (FCI-guided) series
    # -----------------------------
    greedy_selected: List[int] = []
    greedy_energies: List[float] = []
    try:
        if not sr_df.empty and 'subspace_size' in sr_df.columns:
            k_max = int(min(MAX_SUBSPACE, sr_df['subspace_size'].max()))
        else:
            k_max = int(MAX_SUBSPACE)
        greedy_selected, greedy_energies = greedy_from_results(qc_results_sr, k_max=k_max, amp_thresh=1e-5)
    except Exception:
        greedy_selected, greedy_energies = [], []

    # -----------------------------
    # Plot combined comparison
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_yscale('log')
    ax.set_xlabel('Subspace size / number of configurations')
    ax.set_ylabel('FCI energy error / Ha')
    ax.set_title(f'H$_6$ chain - bond length {bond_length:.2f} Å : simulations comparison', fontsize=14)

    # Chemical accuracy region (≤ 1.6e-3 Ha) with diagonal green hatch over light grey
    y_lo = 1.0e-4
    chem_acc = 1.6e-3
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    bg = patches.Rectangle(
        (0.0, y_lo),
        1.0, 
        chem_acc - y_lo, 
        transform=trans,
        facecolor='#D0D0D0', 
        edgecolor='none', 
        alpha=0.35, 
        zorder=0,
    )
    ax.add_patch(bg)
    hat = patches.Rectangle(
        (0.0, y_lo),
        1.0,
        chem_acc - y_lo,
        transform=trans,
        facecolor='none',
        edgecolor='#2ca02c',
        hatch='///',
        linewidth=0.0,
        zorder=0,
    )
    ax.add_patch(hat)
    proxy = patches.Rectangle(
        (0, 0),
        1,
        1,
        facecolor='#D0D0D0',
        edgecolor='#2ca02c',
        hatch='///',
        alpha=0.35,
        label='chemical accuracy',
    )
    ax.add_artist(proxy)

    # Prepare and plot series
    ymax_values: List[float] = []

    # UHF rotation + spin symmetry recovery series — plot FIRST (per request)
    if not sr_df.empty and 'subspace_size' in sr_df.columns and 'qsci_spin_symm_error' in sr_df.columns:
        x_sr = sr_df['subspace_size'].to_numpy()
        y_sr = np.maximum(sr_df['qsci_spin_symm_error'].to_numpy(), y_lo)
        mask_sr = (x_sr >= 1) & (x_sr <= MAX_SUBSPACE)
        ax.plot(
            x_sr[mask_sr],
            y_sr[mask_sr],
            '^-',
            linewidth=1.6,
            markersize=3.5,
            label=f"UHF rotation (2q={qc_results_sr.two_q_gates})",
        )
        if y_sr[mask_sr].size:
            ymax_values.append(float(np.nanmax(y_sr[mask_sr])))

    # LUCJ (1 and 10 layers) — spin-symmetric series, add 2q gate counts in labels when available
    lucj_series = [
        (
            lucj1_csv,
            f"LUCJ 1 layer (2q={qc_results_lucj_1.two_q_gates})",
            {'marker': 'o', 'linestyle': '-', 'linewidth': 1.8, 'markersize': 3.5}
        ),
        (
            lucj10_csv,
            f"LUCJ 10 layers (2q={qc_results_lucj_10.two_q_gates})",
            {'marker': 'o', 'linestyle': '-', 'linewidth': 1.8, 'markersize': 3.5}
        ),
    ]
    for csv_path, label, style in lucj_series:
        lucj_df = pd.read_csv(csv_path)
        if 'spin_symm_energy' in lucj_df.columns and not lucj_df.empty:
            x = lucj_df['subspace_size'].to_numpy()
            # Signed error vs FCI (should be ≥ 0)
            y = (lucj_df['spin_symm_energy'].to_numpy() - float(lucj_df['fci_energy'].iloc[0]))
            y = np.maximum(y, y_lo)
            mask = (x >= 1) & (x <= MAX_SUBSPACE)
            ax.plot(x[mask], y[mask], **style, label=label)
            if y[mask].size:
                ymax_values.append(float(np.nanmax(y[mask])))

    # FCI-guided subspace
    if not fci_df.empty:
        x_fci = fci_df['subspace_size'].to_numpy()
        # Signed error vs FCI (should be ≥ 0)
        y_fci = (fci_df['fci_subspace_energy'].to_numpy() - fci_df['fci_energy'].to_numpy())
        y_fci = np.maximum(y_fci, y_lo)
        mask_f = (x_fci >= 1) & (x_fci <= MAX_SUBSPACE)
        ax.plot(
            x_fci[mask_f],
            y_fci[mask_f],
            's-',
            color='black',
            linewidth=1.8,
            markersize=3.5,
            label='FCI max amp',
        )
        if y_fci[mask_f].size:
            ymax_values.append(float(np.nanmax(y_fci[mask_f])))

    # Greedy curve — plot AFTER FCI and BEFORE HCI (per request)
    if len(greedy_energies) > 0:
        x_g = np.arange(1, len(greedy_energies) + 1)
        y_g = np.maximum(np.asarray(greedy_energies) - float(fci_energy), y_lo)
        mask_g = (x_g >= 1) & (x_g <= MAX_SUBSPACE)
        ax.plot(
            x_g[mask_g],
            y_g[mask_g],
            'o-',
            linewidth=1.6,
            markersize=3.5,
            label='Greedy',
            color='#7f7f7f',
        )
        if y_g[mask_g].size:
            ymax_values.append(float(np.nanmax(y_g[mask_g])))

    # HCI curve
    if not hci_df.empty:
        x_hci = hci_df['subspace_size'].to_numpy()
        y_hci = np.maximum(hci_df['qsci_spin_symm_error'].to_numpy(), y_lo)
        mask_h = (x_hci >= 1) & (x_hci <= MAX_SUBSPACE)
        ax.plot(
            x_hci[mask_h],
            y_hci[mask_h],
            'd--',
            linewidth=1.6,
            markersize=3.5,
            label='HCI',
            color='#800020',
        )
        if y_hci[mask_h].size:
            ymax_values.append(float(np.nanmax(y_hci[mask_h])))

    # RHF and UHF horizontal dashed lines (variational errors; no abs)
    rhf_err = max((rhf.e_tot - fci_energy), y_lo)
    uhf_err = max((uhf.e_tot - fci_energy), y_lo)
    ax.axhline(y=rhf_err, linestyle='--', linewidth=1.6, color='#1f77b4', label='RHF')
    ax.axhline(y=uhf_err, linestyle='--', linewidth=1.6, color='#FFBF00', label='UHF')
    ymax_values.extend([rhf_err, uhf_err])

    # Bounds
    ax.set_xlim(left=1, right=MAX_SUBSPACE)
    y_max = max(ymax_values)
    headroom = 1.25
    ax.set_ylim(bottom=y_lo, top=max(y_lo * 1.1, y_max * headroom))

    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.99)
    plt.tight_layout()
    fig_path = out_dir / 'h6_method_comparison_error_vs_subspace.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved data and plot under: {out_dir}")


if __name__ == '__main__':
    main()
