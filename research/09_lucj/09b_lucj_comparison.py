"""
Compare UHF vs LUCJ spin-symmetric QSCI convergence (H6 chain)
----------------------------------------------------------------

This script computes the UHF and LUCJ statevectors,
builds convergence data, and plots the energy difference relative to the
overall FCI energy. It produces both linear and log-y plots, for:
- Energy difference vs subspace size
- Energy difference vs mean sample number (log x)

Outputs are saved under:
    research/09/data/09b/bond_length_XX
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, scf
from pyscf.cc import CCSD
from qiskit_aer import Aer

from sd_qsci import analysis, circuit, hamiltonian
from sd_qsci.utils import uhf_from_rhf


# Plot style configuration (toggle between Seaborn default and whitegrid)
# Set to True to use Seaborn's default theme; set to False to force whitegrid.
USE_SNS_DEFAULT: bool = True
if USE_SNS_DEFAULT: sns.set_theme()  # default seaborn style and color palette
else: sns.set_theme(style="whitegrid")

# Distinct colors for reference dashed lines (RHF, UHF, FCI)
_palette = sns.color_palette()
RHF_REF_COLOR = _palette[4 % len(_palette)] if _palette else '#000000'
UHF_REF_COLOR = _palette[5 % len(_palette)] if _palette else '#FF0000'
FCI_REF_COLOR = _palette[6 % len(_palette)] if _palette else '#00FF00'


def build_h_chain(bond_length: float, n_atoms: int = 6) -> gto.Mole:
    coords = [(i * bond_length, 0.0, 0.0) for i in range(n_atoms)]
    geometry = '; '.join([f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in coords])
    mol = gto.Mole()
    mol.build(atom=geometry, unit='Angstrom', basis='sto-3g', charge=0, spin=0, verbose=0)
    return mol


def plot_diff_vs_subspace(
    out_dir: Path,
    df_uhf: pd.DataFrame,
    df_lucj: pd.DataFrame,
    df_uhf_lucj: pd.DataFrame,
    bond_length: float,
    logy: bool,
    rhf_energy: float,
    uhf_energy: float,
    fci_energy: float,
) -> None:
    """Plot (Energy - FCI) vs Subspace Size for UHF and LUCJ spin-symmetric energies."""
    fig, ax = plt.subplots(figsize=(12, 8))

    if logy:
        ax.set_yscale('log')

    ax.plot(
        df_uhf['subspace_size'],
        df_uhf['spin_symm_energy_diff'],
        'o-',
        label='UHF spin-symmetric',
        linewidth=2,
        markersize=4,
    )

    ax.plot(
        df_lucj['subspace_size'],
        df_lucj['spin_symm_energy_diff'],
        '^-',
        label='LUCJ spin-symmetric',
        linewidth=2,
        markersize=4,
    )

    # UHF→UHF-rotation + LUCJ pipeline
    ax.plot(
        df_uhf_lucj['subspace_size'],
        df_uhf_lucj['spin_symm_energy_diff'],
        'x-',
        label='UHF rotation + LUCJ',
        linewidth=2,
        markersize=4,
    )

    # FCI subspace energy difference (same for both, use one)
    ax.plot(
        df_lucj['subspace_size'],
        df_lucj['fci_subspace_energy_diff'],
        's-',
        label='FCI subspace',
        linewidth=2,
        markersize=4,
        alpha=0.8,
    )

    ax.set_xlabel('Subspace Size (Number of Configurations)', fontsize=12)
    ax.set_ylabel('Energy difference to FCI (Hartree)' + (' [log y]' if logy else ''), fontsize=12)
    # Enforce requested lower bound on energy difference
    ax.set_ylim(bottom=1e-5)
    # Set x-axis to start at 1 and end at the maximum subspace size present
    try:
        max_x = float(np.nanmax([
            np.nanmax(df_uhf['subspace_size'].values),
            np.nanmax(df_lucj['subspace_size'].values),
            np.nanmax(df_uhf_lucj['subspace_size'].values),
        ]))
        ax.set_xlim(left=1, right=max_x)
    except Exception:
        ax.set_xlim(left=1)
    # Shade the "chemical accuracy" region (ΔE <= 1.6e-3 Ha)
    try:
        ax.axhspan(1.0e-5, 1.6e-3, facecolor='#B0B0B0', alpha=0.2, zorder=0)
    except Exception:
        # Fallback without breaking plotting in unusual backends
        pass
    # Reference horizontal lines (as in analysis.py), adjusted for ΔE axis
    try:
        ax.axhline(
            y=rhf_energy - fci_energy,
            linestyle='--', linewidth=2,
            label=f'RHF: {rhf_energy:.2f} Ha', color=RHF_REF_COLOR,
        )
        ax.axhline(
            y=uhf_energy - fci_energy,
            linestyle='--', linewidth=2,
            label=f'UHF: {uhf_energy:.2f} Ha', color=UHF_REF_COLOR,
        )
        ax.axhline(
            y=0.0,
            linestyle='--', linewidth=2,
            label=f'FCI: {fci_energy:.2f} Ha', color=FCI_REF_COLOR,
        )
    except Exception:
        pass
    title = f"H6 Chain: ΔE vs Subspace Size (Spin-Symmetric)\nBond Length = {bond_length:.2f} Å"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, which='both', alpha=0.3)

    out_path = out_dir / ('h6_qsci_convergence_diff_logy.png' if logy else 'h6_qsci_convergence_diff_linear.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_diff_vs_samples(
    out_dir: Path,
    df_uhf: pd.DataFrame,
    df_lucj: pd.DataFrame,
    df_uhf_lucj: pd.DataFrame,
    bond_length: float,
    logy: bool,
    rhf_energy: float,
    uhf_energy: float,
    fci_energy: float,
) -> None:
    """Plot (Energy - FCI) vs Mean Sample Number for UHF and LUCJ spin-symmetric energies."""
    fig, ax = plt.subplots(figsize=(12, 8))

    if logy:
        ax.set_yscale('log')

    ax.plot(
        df_uhf['mean_sample_number'],
        df_uhf['spin_symm_energy_diff'],
        'o-',
        label='UHF spin-symmetric',
        linewidth=2,
        markersize=4,
    )

    ax.plot(
        df_lucj['mean_sample_number'],
        df_lucj['spin_symm_energy_diff'],
        '^-',
        label='LUCJ spin-symmetric',
        linewidth=2,
        markersize=4,
    )

    ax.plot(
        df_uhf_lucj['mean_sample_number'],
        df_uhf_lucj['spin_symm_energy_diff'],
        'x-',
        label='UHF rotation + LUCJ',
        linewidth=2,
        markersize=4,
    )

    ax.set_xscale('log')
    ax.set_xlabel('Mean Sample Number (log x)', fontsize=12)
    ax.set_ylabel('Energy difference to FCI (Hartree)' + (' [log y]' if logy else ''), fontsize=12)
    # Enforce requested axis bounds
    ax.set_ylim(bottom=1e-5)
    # Set x-axis to start at 1 and end at the maximum mean sample number present
    try:
        max_x = float(np.nanmax([
            np.nanmax(df_uhf['mean_sample_number'].values),
            np.nanmax(df_lucj['mean_sample_number'].values),
            np.nanmax(df_uhf_lucj['mean_sample_number'].values),
        ]))
        # ax.set_xlim(left=1, right=max_x)
        ax.set_xlim(left=1, right=1e10)
    except Exception:
        ax.set_xlim(left=1)
    # Shade the "chemical accuracy" region (ΔE <= 1.6e-3 Ha)
    try:
        ax.axhspan(1.0e-5, 1.6e-3, facecolor='#B0B0B0', alpha=0.2, zorder=0)
    except Exception:
        pass
    # Reference horizontal lines (as in analysis.py), adjusted for ΔE axis
    try:
        ax.axhline(
            y=rhf_energy - fci_energy,
            linestyle='--', linewidth=2,
            label=f'RHF: {rhf_energy:.2f} Ha', color=RHF_REF_COLOR,
        )
        ax.axhline(
            y=uhf_energy - fci_energy,
            linestyle='--', linewidth=2,
            label=f'UHF: {uhf_energy:.2f} Ha', color=UHF_REF_COLOR,
        )
        ax.axhline(
            y=0.0,
            linestyle='--', linewidth=2,
            label=f'FCI: {fci_energy:.2f} Ha', color=FCI_REF_COLOR,
        )
    except Exception:
        pass
    title = f"H6 Chain: ΔE vs Mean Samples (Spin-Symmetric)\nBond Length = {bond_length:.2f} Å"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, which='both', alpha=0.3)

    out_path = out_dir / ('h6_energy_vs_samples_diff_logy.png' if logy else 'h6_energy_vs_samples_diff_linear.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_energy_vs_subspace_linear(
    out_dir: Path,
    df_uhf: pd.DataFrame,
    df_lucj: pd.DataFrame,
    df_uhf_lucj: pd.DataFrame,
    bond_length: float,
    rhf_energy: float,
    uhf_energy: float,
    fci_energy: float,
) -> None:
    """Plot actual energies vs Subspace Size (linear y) for UHF/LUCJ spin-symmetric and FCI subspace."""
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        df_uhf['subspace_size'],
        df_uhf['spin_symm_energy'],
        'o-',
        label='UHF spin-symmetric',
        linewidth=2,
        markersize=4,
    )

    ax.plot(
        df_lucj['subspace_size'],
        df_lucj['spin_symm_energy'],
        '^-',
        label='LUCJ spin-symmetric',
        linewidth=2,
        markersize=4,
    )

    ax.plot(
        df_uhf_lucj['subspace_size'],
        df_uhf_lucj['spin_symm_energy'],
        'x-',
        label='UHF rotation + LUCJ',
        linewidth=2,
        markersize=4,
    )

    ax.plot(
        df_lucj['subspace_size'],
        df_lucj['fci_subspace_energy'],
        's-',
        label='FCI subspace',
        linewidth=2,
        markersize=4,
        alpha=0.8,
    )

    ax.set_xlabel('Subspace Size (Number of Configurations)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    # Set x-axis to start at 1 and end at the maximum subspace size present
    try:
        max_x = float(np.nanmax([
            np.nanmax(df_uhf['subspace_size'].values),
            np.nanmax(df_lucj['subspace_size'].values),
        ]))
        ax.set_xlim(left=1, right=max_x)
    except Exception:
        ax.set_xlim(left=1)
    # Reference horizontal lines (as in analysis.py)
    try:
        ax.axhline(
            y=rhf_energy,
            linestyle='--', linewidth=2,
            label=f'RHF: {rhf_energy:.2f} Ha', color=RHF_REF_COLOR,
        )
        ax.axhline(
            y=uhf_energy,
            linestyle='--', linewidth=2,
            label=f'UHF: {uhf_energy:.2f} Ha', color=UHF_REF_COLOR,
        )
        ax.axhline(
            y=fci_energy,
            linestyle='--', linewidth=2,
            label=f'FCI: {fci_energy:.2f} Ha', color=FCI_REF_COLOR,
        )
    except Exception:
        pass
    title = f"H6 Chain: Energy vs Subspace Size (Spin-Symmetric)\nBond Length = {bond_length:.2f} Å"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, which='both', alpha=0.3)

    out_path = out_dir / 'h6_qsci_convergence_energy_linear.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_energy_vs_samples_linear(
    out_dir: Path,
    df_uhf: pd.DataFrame,
    df_lucj: pd.DataFrame,
    df_uhf_lucj: pd.DataFrame,
    bond_length: float,
    rhf_energy: float,
    uhf_energy: float,
    fci_energy: float,
) -> None:
    """Plot actual energies vs Mean Sample Number (log x, linear y) for UHF/LUCJ spin-symmetric and FCI subspace."""
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        df_uhf['mean_sample_number'],
        df_uhf['spin_symm_energy'],
        'o-',
        label='UHF spin-symmetric',
        linewidth=2,
        markersize=4,
    )

    ax.plot(
        df_lucj['mean_sample_number'],
        df_lucj['spin_symm_energy'],
        '^-',
        label='LUCJ spin-symmetric',
        linewidth=2,
        markersize=4,
    )

    ax.plot(
        df_uhf_lucj['mean_sample_number'],
        df_uhf_lucj['spin_symm_energy'],
        'x-',
        label='UHF rotation + LUCJ',
        linewidth=2,
        markersize=4,
    )

    ax.plot(
        df_lucj['mean_sample_number'],
        df_lucj['fci_subspace_energy'],
        's-',
        label='FCI subspace',
        linewidth=2,
        markersize=4,
        alpha=0.8,
    )

    ax.set_xscale('log')
    ax.set_xlabel('Mean Sample Number (log x)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    # Set x-axis to start at 1 and end at the maximum mean sample number present
    try:
        max_x = float(np.nanmax([
            np.nanmax(df_uhf['mean_sample_number'].values),
            np.nanmax(df_lucj['mean_sample_number'].values),
        ]))
        # ax.set_xlim(left=1, right=max_x)
        ax.set_xlim(left=1, right=1e10)
    except Exception:
        ax.set_xlim(left=1)
    # Reference horizontal lines (as in analysis.py)
    try:
        ax.axhline(
            y=rhf_energy,
            linestyle='--', linewidth=2,
            label=f'RHF: {rhf_energy:.2f} Ha', color=RHF_REF_COLOR,
        )
        ax.axhline(
            y=uhf_energy,
            linestyle='--', linewidth=2,
            label=f'UHF: {uhf_energy:.2f} Ha', color=UHF_REF_COLOR,
        )
        ax.axhline(
            y=fci_energy,
            linestyle='--', linewidth=2,
            label=f'FCI: {fci_energy:.2f} Ha', color=FCI_REF_COLOR,
        )
    except Exception:
        pass
    title = f"H6 Chain: Energy vs Mean Samples (Spin-Symmetric)\nBond Length = {bond_length:.2f} Å"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, which='both', alpha=0.3)

    out_path = out_dir / 'h6_energy_vs_samples_energy_linear.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    # Configuration
    bond_length = 2.0
    base_dir = Path(__file__).parent
    stem = Path(__file__).stem
    out_dir = base_dir / 'data' / stem / f"bond_length_{bond_length:.2f}"

    print("Building H6 chain and running RHF/UHF/FCI...")
    mol = build_h_chain(bond_length, n_atoms=6)
    rhf = scf.RHF(mol).run()

    # UHF pipeline (orbital rotation circuit)
    qc_uhf = analysis.run_quantum_chemistry_calculations(mol, rhf, bond_length)
    conv_uhf = analysis.calc_convergence_data(qc_uhf, spin_symm=True)
    df_uhf = conv_uhf.df.copy()

    # LUCJ pipeline
    print("Building LUCJ circuit and simulating statevector...")
    uhf = uhf_from_rhf(mol, rhf)
    ccsd = CCSD(rhf).run()
    backend = Aer.get_backend("statevector_simulator")
    qc = circuit.get_lucj_circuit(ccsd_obj=ccsd, backend=backend, n_reps=1)
    sv = circuit.simulate(qc)
    spin_symm_amp = analysis.spin_symm_amplitudes(sv.data)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)
    fci_energy, n_fci_configs, fci_vec = analysis.calc_fci_energy(rhf)

    qc_lucj = analysis.QuantumChemistryResults(
        mol=mol,
        rhf=rhf,
        uhf=uhf,
        sv=sv,
        H=H,
        fci_energy=fci_energy,
        n_fci_configs=n_fci_configs,
        fci_vec=fci_vec,
        bond_length=bond_length,
        spin_symm_amp=spin_symm_amp,
    )
    conv_lucj = analysis.calc_convergence_data(qc_lucj, spin_symm=True)
    df_lucj = conv_lucj.df.copy()

    # UHF rotation + LUCJ pipeline (no HF init in LUCJ)
    print("Building UHF-rotation + LUCJ circuit and simulating statevector...")
    qc2 = circuit.uhf_rotation_then_lucj_circuit(
        ccsd_obj=ccsd,
        backend=backend,
        rhf=rhf,
        uhf=uhf,
        n_reps=1,
    )
    sv2 = circuit.simulate(qc2)
    spin_symm_amp2 = analysis.spin_symm_amplitudes(sv2.data)
    qc_uhf_lucj = analysis.QuantumChemistryResults(
        mol=mol,
        rhf=rhf,
        uhf=uhf,
        sv=sv2,
        H=H,
        fci_energy=fci_energy,
        n_fci_configs=n_fci_configs,
        fci_vec=fci_vec,
        bond_length=bond_length,
        spin_symm_amp=spin_symm_amp2,
    )
    conv_uhf_lucj = analysis.calc_convergence_data(qc_uhf_lucj, spin_symm=True)
    df_uhf_lucj = conv_uhf_lucj.df.copy()

    # Clarify what "max subspace size" means in these plots to avoid confusion:
    # It is determined by the number of non-negligible amplitudes (above tolerance)
    # in the given statevector, NOT the full Hilbert space dimension.
    # Therefore, the largest UHF subspace shown here does not span the entire
    # Hilbert space, and its QSCI energy need not equal the FCI energy.
    try:
        hilbert_dim = len(qc_uhf.sv.data)
        uhf_support = int(np.count_nonzero(np.abs(qc_uhf.sv.data) > analysis.DEFAULT_SV_TOL))
        lucj_support = int(np.count_nonzero(np.abs(qc_lucj.sv.data) > analysis.DEFAULT_SV_TOL))
        print("Note on subspace sizes:")
        print(f"  Hilbert space dimension: {hilbert_dim}")
        print(f"  UHF nonzero support (>|{analysis.DEFAULT_SV_TOL:g}|): {uhf_support}")
        print(f"  LUCJ nonzero support (>|{analysis.DEFAULT_SV_TOL:g}|): {lucj_support}")
        print("  Plotted 'max subspace size' corresponds to the support size above, not the full Hilbert space.")
    except Exception:
        pass

    # Ensure LUCJ variants and FCI subspace data extend to UHF's maximum subspace size
    max_uhf_size = int(df_uhf['subspace_size'].max())
    max_lucj_size = int(df_lucj['subspace_size'].max())
    max_uhf_lucj_size = int(df_uhf_lucj['subspace_size'].max())
    if max_lucj_size < max_uhf_size:
        # Extend LUCJ dataframe with computed values up to max_uhf_size
        add_rows = []
        spin_data = qc_lucj.spin_symm_amp
        for size in range(max_lucj_size + 1, max_uhf_size + 1):
            symm_energy = analysis.calc_qsci_energy_with_size(qc_lucj.H, spin_data, size)
            fci_sub_energy = analysis.calc_fci_subspace_energy(qc_lucj.H, qc_lucj.fci_vec, size)
            idx = np.argsort(np.abs(spin_data))[-size:]
            min_coeff = np.min(np.abs(spin_data[idx])) if size > 0 else 0.0
            mean_sample_number = np.inf if min_coeff == 0 else 1.0 / (min_coeff ** 2)
            add_rows.append({
                'subspace_size': size,
                'qsci_energy': np.nan,  # not used in this script
                'spin_symm_energy': symm_energy,
                'fci_subspace_energy': fci_sub_energy,
                'mean_sample_number': mean_sample_number,
            })

        if add_rows:
            df_lucj = pd.concat([df_lucj, pd.DataFrame(add_rows)], ignore_index=True)
            df_lucj = df_lucj.sort_values('subspace_size').reset_index(drop=True)

    if max_uhf_lucj_size < max_uhf_size:
        # Extend UHF+LUCJ dataframe similarly
        add_rows = []
        spin_data = qc_uhf_lucj.spin_symm_amp
        for size in range(max_uhf_lucj_size + 1, max_uhf_size + 1):
            symm_energy = analysis.calc_qsci_energy_with_size(qc_uhf_lucj.H, spin_data, size)
            fci_sub_energy = analysis.calc_fci_subspace_energy(qc_uhf_lucj.H, qc_uhf_lucj.fci_vec, size)
            idx = np.argsort(np.abs(spin_data))[-size:]
            min_coeff = np.min(np.abs(spin_data[idx])) if size > 0 else 0.0
            mean_sample_number = np.inf if min_coeff == 0 else 1.0 / (min_coeff ** 2)
            add_rows.append({
                'subspace_size': size,
                'qsci_energy': np.nan,
                'spin_symm_energy': symm_energy,
                'fci_subspace_energy': fci_sub_energy,
                'mean_sample_number': mean_sample_number,
            })

        if add_rows:
            df_uhf_lucj = pd.concat([df_uhf_lucj, pd.DataFrame(add_rows)], ignore_index=True)
            df_uhf_lucj = df_uhf_lucj.sort_values('subspace_size').reset_index(drop=True)

    # Compute differences to overall FCI energy
    fci_ref = qc_uhf.fci_energy  # same for both
    for df in (df_uhf, df_lucj, df_uhf_lucj):
        df['spin_symm_energy_diff'] = df['spin_symm_energy'] - fci_ref
        df['fci_subspace_energy_diff'] = df['fci_subspace_energy'] - fci_ref

    # For log-y plots, replace exact zeros with NaN to avoid log(0)
    for df in (df_uhf, df_lucj, df_uhf_lucj):
        for col in ('spin_symm_energy_diff', 'fci_subspace_energy_diff'):
            if col in df:
                df[col] = df[col].replace(0.0, np.nan)

    print(f"Saving plots to: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ΔE vs subspace size
    plot_diff_vs_subspace(
        out_dir, df_uhf, df_lucj, df_uhf_lucj, bond_length, logy=False,
        rhf_energy=qc_uhf.rhf.e_tot, uhf_energy=qc_uhf.uhf.e_tot, fci_energy=fci_ref,
    )
    plot_diff_vs_subspace(
        out_dir, df_uhf, df_lucj, df_uhf_lucj, bond_length, logy=True,
        rhf_energy=qc_uhf.rhf.e_tot, uhf_energy=qc_uhf.uhf.e_tot, fci_energy=fci_ref,
    )

    # ΔE vs samples
    plot_diff_vs_samples(
        out_dir, df_uhf, df_lucj, df_uhf_lucj, bond_length, logy=False,
        rhf_energy=qc_uhf.rhf.e_tot, uhf_energy=qc_uhf.uhf.e_tot, fci_energy=fci_ref,
    )
    plot_diff_vs_samples(
        out_dir, df_uhf, df_lucj, df_uhf_lucj, bond_length, logy=True,
        rhf_energy=qc_uhf.rhf.e_tot, uhf_energy=qc_uhf.uhf.e_tot, fci_energy=fci_ref,
    )

    # Also plot actual energies with linear y-axis
    plot_energy_vs_subspace_linear(
        out_dir, df_uhf, df_lucj, df_uhf_lucj, bond_length,
        rhf_energy=qc_uhf.rhf.e_tot, uhf_energy=qc_uhf.uhf.e_tot, fci_energy=fci_ref,
    )
    plot_energy_vs_samples_linear(
        out_dir, df_uhf, df_lucj, df_uhf_lucj, bond_length,
        rhf_energy=qc_uhf.rhf.e_tot, uhf_energy=qc_uhf.uhf.e_tot, fci_energy=fci_ref,
    )

    print("Done.")


if __name__ == "__main__":
    main()
