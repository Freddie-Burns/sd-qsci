from __future__ import annotations

from math import log2
from pathlib import Path
from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sd_qsci import spin
from sd_qsci.analysis import QuantumChemistryResults, ConvergenceResults, \
    spin_closed_subspace_sizes, spin_symm_amplitudes, calc_qsci_energy_with_size


def convergence_comparison(
    data_dir: Path,
    qc_results: QuantumChemistryResults,
    conv_results: ConvergenceResults,
    title_prefix: Optional[str] = None,
    ylog: bool = False,
    label_raw: str = 'UHF State',
    label_spin: str = 'UHF State Spin Recovered',
):
    """
    Create and save convergence comparison plot.

    Plots QSCI and FCI subspace energies against subspace size, comparing
    with RHF, UHF, and FCI reference energies.

    Parameters
    ----------
    data_dir : Path
        Directory to save the PNG file.
    qc_results : QuantumChemistryResults
        Quantum chemistry calculation results.
    conv_results : ConvergenceResults
        Convergence analysis results.
    title_prefix : str, optional
        Prefix to add to the plot title. Default is None.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        conv_results.df['subspace_size'],
        conv_results.df['qsci_energy'],
        'o-',
        label=label_raw,
        linewidth=2,
        markersize=4,
        color='#0072B2',
    )

    # Only plot spin-recovery points at subspace sizes that are closed under
    # spin symmetry (i.e., include all members of each spin-symmetric set).
    spin_closed_sizes = set(spin_closed_subspace_sizes(qc_results.sv.data))
    df_symm = conv_results.df[conv_results.df['subspace_size'].isin(spin_closed_sizes)]

    ax.plot(
        df_symm['subspace_size'],
        df_symm['spin_symm_energy'],
        '^-',
        label=label_spin,
        linewidth=2,
        markersize=4,
        color='#D55E00',
    )
    ax.plot(
        conv_results.df['subspace_size'],
        conv_results.df['fci_subspace_energy'],
        's-',
        label='FCI',
        linewidth=2,
        markersize=4,
        color='#009E73',
    )

    ax.axhline(
        y=qc_results.rhf.e_tot,
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'RHF: {qc_results.rhf.e_tot:.2f} Ha',
    )
    ax.axhline(
        y=qc_results.uhf.e_tot,
        color='orange',
        linestyle='--',
        linewidth=2,
        label=f'UHF: {qc_results.uhf.e_tot:.2f} Ha',
    )
    ax.axhline(
        y=qc_results.fci_energy,
        color='green',
        linestyle='--',
        linewidth=2,
        label=f'FCI: {qc_results.fci_energy:.2f} Ha',
    )

    ax.set_xlabel('Subspace Size (Number of Configurations)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    if ylog:
        # Use symmetric log to allow negative energies while still showing magnitude
        ax.set_yscale('symlog', linthresh=1e-6)
    bond_info = f"Bond Length = {qc_results.bond_length:.2f} Å" if qc_results.bond_length is not None else ""
    title = (f"Energy Convergence Comparison\n{bond_info}")
    if title_prefix:
        title = f"{title_prefix}: " + title
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(data_dir) / 'h6_qsci_convergence.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def energy_vs_samples(
    data_dir: Path,
    qc_results: QuantumChemistryResults,
    conv_results: ConvergenceResults,
    title_prefix: Optional[str] = None,
    ylog: bool = False,
    label_raw: str = 'UHF State',
    label_spin: str = 'UHF State Spin Recovered',
):
    """
    Create and save energy vs mean-sample-number plot.

    Plots QSCI energy on a semilog scale against the mean sample number required
    for each subspace size.

    Parameters
    ----------
    data_dir : Path
        Directory to save the PNG file.
    qc_results : QuantumChemistryResults
        Quantum chemistry calculation results.
    conv_results : ConvergenceResults
        Convergence analysis results.
    title_prefix : str, optional
        Prefix to add to the plot title. Default is None.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Build mean-sample-number series for raw and spin-recovered selections
    sizes = list(conv_results.df['subspace_size'])

    def mean_samples_for_sizes(data: np.ndarray, sizes_list: list[int]) -> list[float]:
        vals = []
        # Determine selection order by amplitude magnitude
        order = np.argsort(np.abs(data))
        for size in sizes_list:
            idx = order[-int(size):]
            min_coeff = float(np.min(np.abs(data[idx]))) if len(idx) > 0 else 0.0
            ms = float(1.0 / (min_coeff ** 2)) if min_coeff > 0 else np.inf
            vals.append(ms)
        return vals

    # Raw (UHF-rotated) amplitudes across all sizes
    ms_raw = mean_samples_for_sizes(qc_results.sv.data, sizes)

    # Spin-recovered amplitudes only at spin-closed subspace sizes
    spin_sizes_all = sorted(set(spin_closed_subspace_sizes(qc_results.sv.data)))
    # clip to available computed range
    if len(sizes) > 0:
        max_size = int(sizes[-1])
        spin_sizes = [int(s) for s in spin_sizes_all if int(s) <= max_size]
    else:
        spin_sizes = []
    ms_symm = mean_samples_for_sizes(qc_results.spin_symm_amp, spin_sizes) if len(spin_sizes) else []

    # Plot raw QSCI energy vs raw mean sample number
    ax.semilogx(
        ms_raw,
        list(conv_results.df['qsci_energy']), 'o-',
        label=label_raw,
        linewidth=2,
        markersize=4,
        color='#0072B2',
    )

    # Plot spin-recovered energy vs mean sample number (only at spin-closed sizes)
    if len(spin_sizes):
        df_symm = conv_results.df[conv_results.df['subspace_size'].isin(spin_sizes)]
        ax.semilogx(
            ms_symm,
            list(df_symm['spin_symm_energy']), 's-',
            label=label_spin,
            linewidth=2,
            markersize=4,
            color='#D55E00',
        )

    # RHF energy reference line
    ax.axhline(
        y=qc_results.rhf.e_tot,
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'RHF: {qc_results.rhf.e_tot:.2f} Ha',
    )
    # UHF energy reference line
    ax.axhline(
        y=qc_results.uhf.e_tot,
        color='orange',
        linestyle='--',
        linewidth=2,
        label=f'UHF: {qc_results.uhf.e_tot:.2f} Ha',
    )
    # FCI energy reference line
    ax.axhline(
        y=qc_results.fci_energy,
        color='green',
        linestyle='--',
        linewidth=2,
        label=f'FCI: {qc_results.fci_energy:.2f} Ha',
    )

    # Label axes
    ax.set_xlabel('Mean Sample Number (log scale)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    if ylog:
        ax.set_yscale('symlog', linthresh=1e-6)

    # Create title with bond length if available
    if qc_results.bond_length is None: bond_info = ""
    else: bond_info = f"Bond Length = {qc_results.bond_length:.2f} Å"
    title = (f"Energy vs Mean Sample Number\n{bond_info}")
    if title_prefix: title = f"{title_prefix}: " + title
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    out_path = Path(data_dir) / 'h6_energy_vs_samples.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def total_spin_vs_subspace(
    data_dir: Path,
    qc_results: QuantumChemistryResults,
    conv_results: ConvergenceResults,
    title_prefix: Optional[str] = None,
    ylog: bool = False,
):
    """
    Plot total spin <S^2> versus subspace size with/without spin symmetry recovery.

    For each subspace size present in conv_results, builds the QSCI ground-state
    vector using (a) the raw circuit statevector amplitudes and (b) the spin-
    symmetry recovered amplitudes, then evaluates the expectation value of the
    total spin operator S^2.

    Parameters
    ----------
    data_dir : Path
        Directory to save the PNG file.
    qc_results : QuantumChemistryResults
        Quantum chemistry results containing the Hamiltonian and statevectors.
    conv_results : ConvergenceResults
        Convergence data that provides the subspace sizes to evaluate.
    title_prefix : str, optional
        Prefix to add to the plot title. Default is None.
    """
    sns.set_style("whitegrid")

    # Build S^2 operator in the full Fock space (RHF ordering assumed)
    n_spatial_orbs = qc_results.mol.nao
    S2 = spin.total_spin_S2(n_spatial_orbs)

    sizes = list(conv_results.df['subspace_size'])
    s2_raw = []
    s2_symm = []

    # Ensure we have spin-symmetric amplitudes
    if qc_results.spin_symm_amp is None:
        qc_results.spin_symm_amp = spin_symm_amplitudes(qc_results.sv.data)
    spin_symm_amp = qc_results.spin_symm_amp

    for size in sizes:
        # Raw amplitudes
        _, psi_raw, _ = calc_qsci_energy_with_size(
            qc_results.H, qc_results.sv.data, int(size), return_vector=True
        )
        s2_val_raw = spin.expectation(S2, psi_raw)
        s2_raw.append(float(np.real(s2_val_raw)))

    # For spin-symmetry recovered amplitudes, evaluate only at spin-closed sizes
    symm_sizes = sorted(set(spin_closed_subspace_sizes(qc_results.sv.data)))
    # Keep only sizes within the computed convergence range to ensure
    # the x-axis aligns with available subspace sizes
    if len(sizes) > 0:
        max_size = int(sizes[-1])
        symm_sizes = [int(s) for s in symm_sizes if int(s) <= max_size]
    for size in symm_sizes:
        _, psi_symm, _ = calc_qsci_energy_with_size(
            qc_results.H, spin_symm_amp, int(size), return_vector=True
        )
        s2_val_symm = spin.expectation(S2, psi_symm)
        s2_symm.append(float(np.real(s2_val_symm)))

    # Use a consistent figure size with other plots for uniform font/line scales
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        sizes,
        s2_raw,
        'o-',
        label='raw amplitudes',
        linewidth=2,
        markersize=4,
        color='#0072B2',
    )
    ax.plot(
        symm_sizes,
        s2_symm,
        's-',
        label='spin recovered',
        linewidth=2,
        markersize=4,
        color='#D55E00',
    )

    ax.set_xlabel('Subspace Size (Number of Configurations)', fontsize=12)
    ax.set_ylabel(r'Total spin $\langle S^2 \rangle$', fontsize=12)
    if ylog:
        ax.set_yscale('log')
    bond_info = f"Bond Length = {qc_results.bond_length:.2f} Å" if qc_results.bond_length is not None else ""
    title = (f"Total Spin vs Subspace Size\n{bond_info}")
    if title_prefix:
        title = f"{title_prefix}: " + title
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(data_dir) / 'total_spin_vs_subspace.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def statevector_coefficients(
    qsci_vec: np.ndarray,
    fci_vec: np.ndarray,
    data_dir: Path,
    n_top: int = 20,
    ylog: bool = False,
):
    """
    Plot comparison of QSCI, spin-recovered QSCI, and FCI statevector coefficients.

    Creates two plots: one showing the top n_top configurations as a bar chart,
    and another showing all significant configurations on a log scale.
    The QSCI series is shown both before and after spin-symmetry recovery.

    Parameters
    ----------
    qsci_vec : np.ndarray
        QSCI statevector.
    fci_vec : np.ndarray
        FCI statevector.
    data_dir : Path
        Directory to save the PNG files.
    n_top : int, optional
        Number of top configurations to show in the first plot. Default is 20.

    Returns
    -------
    dict
        Dictionary with statistics:
        - 'n_significant_fci': Number of significant FCI configurations
        - 'n_significant_qsci': Number of significant QSCI configurations
        - 'max_fci_coef': Maximum FCI coefficient magnitude
        - 'max_qsci_coef': Maximum QSCI coefficient magnitude
        - 'overlap': Overlap between FCI and QSCI vectors
    """
    fci_abs = np.abs(fci_vec)
    # Build spin-recovered amplitudes from the provided QSCI vector
    qsci_symm_vec = spin_symm_amplitudes(qsci_vec)
    top_indices = np.argsort(fci_abs)[-n_top:][::-1]

    qsci_coefs = np.abs(qsci_vec[top_indices])
    qsci_symm_coefs = np.abs(qsci_symm_vec[top_indices])
    fci_coefs = fci_abs[top_indices]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(n_top)
    width = 0.28

    # Three side-by-side bars: FCI, QSCI, QSCI Spin-Recovered
    ax.bar(x - width, fci_coefs, width, label='FCI', color='green', alpha=0.8)
    ax.bar(x, qsci_coefs, width, label='QSCI', color='purple', alpha=0.8)
    ax.bar(x + width, qsci_symm_coefs, width, label='QSCI (Spin Recovered)', color='#D55E00', alpha=0.8)

    # X-axis labels should be the computational basis bitstrings (zero-padded to n_qubits)
    n_qubits = int(log2(len(fci_vec))) if len(fci_vec) > 0 else 0
    bitstring_labels = [format(i, f"0{n_qubits}b") for i in top_indices]
    occupation_labels = [_occupation_vector(x) for x in bitstring_labels]

    ax.set_xlabel('Computational basis state', fontsize=12)
    ax.set_ylabel('|Coefficient|', fontsize=12)
    if ylog:
        ax.set_yscale('log')
    ax.set_title(f'Top {n_top} Configuration Coefficients', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    # Angle the bitstring labels slightly for readability
    ax.set_xticklabels(occupation_labels, rotation=35, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    (Path(data_dir) / 'statevector_coefficients.png').parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(data_dir) / 'statevector_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(14, 8))

    significant_mask = (fci_abs > 1e-10) | (np.abs(qsci_vec) > 1e-10) | (np.abs(qsci_symm_vec) > 1e-10)
    significant_indices = np.where(significant_mask)[0]

    sort_order = np.argsort(fci_abs[significant_indices])[::-1]
    sorted_indices = significant_indices[sort_order]

    fci_sig = fci_abs[sorted_indices]
    qsci_sig = np.abs(qsci_vec[sorted_indices])
    qsci_symm_sig = np.abs(qsci_symm_vec[sorted_indices])

    x_all = np.arange(len(sorted_indices))

    ax2.semilogy(x_all, fci_sig, 'o-', label='FCI', color='green', markersize=3, linewidth=1)
    ax2.semilogy(x_all, qsci_sig, 's-', label='QSCI', color='purple', markersize=3, linewidth=1, alpha=0.7)
    ax2.semilogy(x_all, qsci_symm_sig, '^-', label='QSCI (Spin Recovered)', color='#D55E00', markersize=3, linewidth=1, alpha=0.8)

    ax2.set_xlabel('Configuration Index (sorted by FCI amplitude)', fontsize=12)
    ax2.set_ylabel('|Coefficient| (log scale)', fontsize=12)
    ax2.set_title('FCI vs QSCI Statevector Coefficients (All Significant Configurations)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    (Path(data_dir) / 'statevector_coefficients_full.png').parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(data_dir) / 'statevector_coefficients_full.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # Return some stats for programmatic use
    stats = {
        'n_significant_fci': int(np.sum(fci_abs > 1e-10)),
        'n_significant_qsci': int(np.sum(np.abs(qsci_vec) > 1e-10)),
        'max_fci_coef': float(np.max(fci_abs)),
        'max_qsci_coef': float(np.max(np.abs(qsci_vec))),
        'max_qsci_spin_recovered_coef': float(np.max(np.abs(qsci_symm_vec))),
        'overlap': float(np.abs(np.vdot(fci_vec, qsci_vec))),
        'overlap_spin_recovered': float(np.abs(np.vdot(fci_vec, qsci_symm_vec)))
    }
    return stats


def _occupation_vector(bitstring: str) -> str:
    """
    Take a configuration bitstring and
    return the spatial occupation vector as a string, for labelling.
    """
    n = len(bitstring)
    alpha, beta = bitstring[:n//2], bitstring[n//2:]
    occ_vec = ''
    for i in range(n):
        if alpha[i] == '1' and beta[i] == '1':
            occ_vec += '2'
        elif alpha[i] == '1':
            occ_vec += '\u03b1'
        elif beta[i] == '1':
            occ_vec += '\u03b2'
        else:
            occ_vec += '0'
    return occ_vec
