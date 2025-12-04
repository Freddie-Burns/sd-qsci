"""
Analysis utilities for QSCI convergence and statevector handling.

This module extracts reusable functions from the research scripts so they can
be reused for arbitrary molecules/geometries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, fci
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import Statevector

from sd_qsci.utils import uhf_from_rhf
from sd_qsci import circuit, hamiltonian


# Defaults for tolerances (can be overridden by callers)
DEFAULT_SV_TOL = 1e-6
DEFAULT_FCI_TOL = 1e-6


@dataclass
class QuantumChemistryResults:
    """Container for quantum chemistry calculation results.

    Fields:
      mol: PySCF molecule
      rhf: RHF object
      uhf: UHF object
      sv: Statevector (from circuit.simulate)
      H: full Hamiltonian matrix (numpy array or sparse matrix)
      fci_energy: float
      n_fci_configs: int
      fci_vec: np.ndarray (full Fock-space FCI vector)
      bond_length: float | None
    """
    mol: gto.Mole
    rhf: object
    uhf: object
    sv: Statevector
    H: np.ndarray
    fci_energy: float
    n_fci_configs: int
    fci_vec: np.ndarray
    bond_length: Optional[float]


@dataclass
class ConvergenceResults:
    """Container for convergence analysis results."""
    df: pd.DataFrame
    max_size: int
    n_configs_below_uhf: Optional[int]
    n_configs_reach_fci: Optional[int]


def calculate_convergence_data(qc_results: QuantumChemistryResults,
                               sv_tol: float = DEFAULT_SV_TOL,
                               fci_tol: float = DEFAULT_FCI_TOL) -> ConvergenceResults:
    """
    Calculate QSCI and FCI subspace energies for varying subspace sizes.

    Parameters
    ----------
    qc_results: QuantumChemistryResults
    sv_tol: float
        Threshold for considering statevector amplitudes as present.
    fci_tol: float
        Tolerance for considering a QSCI energy equal to FCI.

    Returns
    -------
    ConvergenceResults
    """
    max_idx = np.argwhere(np.abs(qc_results.sv.data) > sv_tol).ravel()
    max_size = len(max_idx)

    subspace_sizes = list(range(1, max_size + 1))
    qsci_energies = []
    fci_subspace_energies = []
    mean_sample_numbers = []

    n_configs_below_uhf = None
    n_configs_reach_fci = None

    for size in subspace_sizes:
        energy = calc_qsci_energy_with_size(qc_results.H, qc_results.sv, size)
        qsci_energies.append(energy)

        fci_sub_energy = calc_fci_subspace_energy(qc_results.H, qc_results.fci_vec, size)
        fci_subspace_energies.append(fci_sub_energy)

        idx = np.argsort(np.abs(qc_results.sv.data))[-size:]
        min_coeff = np.min(np.abs(qc_results.sv.data[idx]))
        mean_sample_number = 1.0 / (min_coeff ** 2)
        mean_sample_numbers.append(mean_sample_number)

        if n_configs_below_uhf is None and energy < qc_results.uhf.e_tot:
            n_configs_below_uhf = size

        if n_configs_reach_fci is None and abs(energy - qc_results.fci_energy) < fci_tol:
            n_configs_reach_fci = size

    df = pd.DataFrame({
        'subspace_size': subspace_sizes,
        'qsci_energy': qsci_energies,
 'fci_subspace_energy': fci_subspace_energies,
        'mean_sample_number': mean_sample_numbers
    })

    return ConvergenceResults(df=df, max_size=max_size,
                              n_configs_below_uhf=n_configs_below_uhf,
                              n_configs_reach_fci=n_configs_reach_fci)


def calc_fci_energy(rhf, tol: float = 1e-10) -> Tuple[float, int, np.ndarray]:
    """
    Compute FCI ground state and map PySCF's CI vector into the full Fock space.

    Returns (fci_energy, n_configs, fci_vec_full)
    """
    ci_solver = fci.FCI(rhf)
    fci_energy, fci_vec_ci = ci_solver.kernel()

    mol = rhf.mol
    nelec = mol.nelec

    fci_vec = fci_to_fock_space(fci_vec_ci, mol, nelec)
    n_configs = int(np.count_nonzero(np.abs(fci_vec) > tol))

    return fci_energy, n_configs, fci_vec


def calc_fci_subspace_energy(H, fci_vec, n_configs: int):
    """
    Energy of subspace spanned by the n_configs largest-amplitude FCI configurations.
    """
    idx = np.argsort(np.abs(fci_vec))[-n_configs:]
    H_sub = H[np.ix_(idx, idx)]

    if H_sub.shape[0] <= 2:
        eigenvalues = eigh(H_sub.toarray() if hasattr(H_sub, 'toarray') else H_sub, eigvals_only=True)
        E0 = float(np.min(eigenvalues))
    else:
        vals = eigsh(H_sub, k=1, which='SA', return_eigenvectors=False)
        E0 = float(vals[0])

    return E0


def calc_qsci_energy_with_size(H, statevector: Statevector, n_configs: int, return_vector: bool = False):
    """
    Compute QSCI energy by diagonalizing Hamiltonian restricted to the largest n_configs
    components of the provided statevector.
    """
    idx = np.argsort(np.abs(statevector.data))[-n_configs:]
    H_sub = H[np.ix_(idx, idx)]

    if H_sub.shape[0] <= 2:
        eigenvalues, eigenvectors = eigh(H_sub.toarray() if hasattr(H_sub, 'toarray') else H_sub)
        E0 = float(np.min(eigenvalues))
        psi0 = eigenvectors[:, 0]
    else:
        vals, vecs = eigsh(H_sub, k=1, which='SA')
        E0 = float(vals[0])
        psi0 = vecs[:, 0]

    if return_vector:
        psi0_full = np.zeros(statevector.data.shape, dtype=complex)
        psi0_full[idx] = psi0
        return E0, psi0_full, idx

    return E0


def fci_to_fock_space(fci_vec, mol: gto.Mole, nelec) -> np.ndarray:
    """
    Map PySCF's CI vector into the full Fock space vector (BLOCK spin ordering).
    """
    from pyscf.fci import cistring

    nmo = mol.nao
    n_alpha, n_beta = nelec
    n_spin_orbitals = 2 * nmo

    alpha_strs = cistring.make_strings(range(nmo), n_alpha)
    beta_strs = cistring.make_strings(range(nmo), n_beta)

    fock_vec = np.zeros(2 ** n_spin_orbitals, dtype=complex)

    fci_vec_flat = fci_vec.flatten()

    for i_alpha, alpha_str in enumerate(alpha_strs):
        for i_beta, beta_str in enumerate(beta_strs):
            fock_idx = (alpha_str << nmo) | beta_str
            ci_idx = i_alpha * len(beta_strs) + i_beta
            fock_vec[fock_idx] = fci_vec_flat[ci_idx]

    return fock_vec


def run_quantum_chemistry_calculations(mol: gto.Mole, rhf, bond_length: Optional[float] = None) -> QuantumChemistryResults:
    """
    Run UHF, build orbital-rotation circuit, simulate statevector and construct Hamiltonian.

    Returns QuantumChemistryResults.
    """
    uhf = uhf_from_rhf(mol, rhf)
    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
    sv = circuit.simulate(qc)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)

    sv_energy = (sv.data.conj().T @ H @ sv.data).real
    if not np.isclose(sv_energy, uhf.e_tot):
        raise RuntimeError("Orbital rotation verification failed: statevector energy != UHF energy")

    fci_energy, n_fci_configs, fci_vec = calc_fci_energy(rhf)

    return QuantumChemistryResults(
        mol=mol,
        rhf=rhf,
        uhf=uhf,
        sv=sv,
        H=H,
        fci_energy=fci_energy,
        n_fci_configs=n_fci_configs,
        fci_vec=fci_vec,
        bond_length=bond_length
    )


def plot_convergence_comparison(data_dir: Path, qc_results: QuantumChemistryResults,
                                conv_results: ConvergenceResults, title_prefix: Optional[str] = None):
    """Create and save convergence comparison plot as PNG in data_dir."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(conv_results.df['subspace_size'], conv_results.df['qsci_energy'], 'o-',
            label='QSCI (UHF-based selection)', linewidth=2, markersize=4, color='purple')

    ax.plot(conv_results.df['subspace_size'], conv_results.df['fci_subspace_energy'], 's-',
            label='FCI subspace (FCI-based selection)', linewidth=2, markersize=4, color='darkgreen')

    ax.axhline(y=qc_results.rhf.e_tot, color='blue', linestyle='--', linewidth=2,
               label=f'RHF: {qc_results.rhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.uhf.e_tot, color='orange', linestyle='--', linewidth=2,
               label=f'UHF: {qc_results.uhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.fci_energy, color='green', linestyle='--', linewidth=2,
               label=f'FCI: {qc_results.fci_energy:.6f} Ha')

    ax.set_xlabel('Subspace Size (Number of Configurations)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    bond_info = f"Bond Length = {qc_results.bond_length:.2f} Å" if qc_results.bond_length is not None else ""
    title = (f"Energy Convergence Comparison\n{bond_info}")
    if title_prefix:
        title = f"{title_prefix}: " + title
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    (Path(data_dir) / 'h6_qsci_convergence.png').parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(data_dir) / 'h6_qsci_convergence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_energy_vs_samples(data_dir: Path, qc_results: QuantumChemistryResults,
                           conv_results: ConvergenceResults, title_prefix: Optional[str] = None):
    """Create and save energy vs mean-sample-number plot."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.semilogx(conv_results.df['mean_sample_number'], conv_results.df['qsci_energy'], 'o-',
                label='QSCI (UHF-based selection)', linewidth=2, markersize=4, color='purple')

    ax.axhline(y=qc_results.rhf.e_tot, color='blue', linestyle='--', linewidth=2,
               label=f'RHF: {qc_results.rhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.uhf.e_tot, color='orange', linestyle='--', linewidth=2,
               label=f'UHF: {qc_results.uhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.fci_energy, color='green', linestyle='--', linewidth=2,
               label=f'FCI: {qc_results.fci_energy:.6f} Ha')

    ax.set_xlabel('Mean Sample Number (log scale)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    bond_info = f"Bond Length = {qc_results.bond_length:.2f} Å" if qc_results.bond_length is not None else ""
    title = (f"Energy vs Mean Sample Number\n{bond_info}")
    if title_prefix:
        title = f"{title_prefix}: " + title
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    (Path(data_dir) / 'h6_energy_vs_samples.png').parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(data_dir) / 'h6_energy_vs_samples.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_statevector_coefficients(qsci_vec: np.ndarray, fci_vec: np.ndarray, data_dir: Path, n_top: int = 20):
    """Plot comparison of coefficients and save two PNG files."""
    fci_abs = np.abs(fci_vec)
    top_indices = np.argsort(fci_abs)[-n_top:][::-1]

    qsci_coefs = np.abs(qsci_vec[top_indices])
    fci_coefs = fci_abs[top_indices]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(n_top)
    width = 0.35

    ax.bar(x - width/2, fci_coefs, width, label='FCI', color='green', alpha=0.8)
    ax.bar(x + width/2, qsci_coefs, width, label='QSCI', color='purple', alpha=0.8)

    ax.set_xlabel('Configuration Index (sorted by FCI amplitude)', fontsize=12)
    ax.set_ylabel('|Coefficient|', fontsize=12)
    ax.set_title(f'Top {n_top} Configuration Coefficients', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in top_indices], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    (Path(data_dir) / 'statevector_coefficients.png').parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(data_dir) / 'statevector_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(14, 8))

    significant_mask = (fci_abs > 1e-10) | (np.abs(qsci_vec) > 1e-10)
    significant_indices = np.where(significant_mask)[0]

    sort_order = np.argsort(fci_abs[significant_indices])[::-1]
    sorted_indices = significant_indices[sort_order]

    fci_sig = fci_abs[sorted_indices]
    qsci_sig = np.abs(qsci_vec[sorted_indices])

    x_all = np.arange(len(sorted_indices))

    ax2.semilogy(x_all, fci_sig, 'o-', label='FCI', color='green', markersize=3, linewidth=1)
    ax2.semilogy(x_all, qsci_sig, 's-', label='QSCI', color='purple', markersize=3, linewidth=1, alpha=0.7)

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
        'overlap': float(np.abs(np.vdot(fci_vec, qsci_vec)))
    }
    return stats


def save_convergence_data(data_dir: Path, qc_results: QuantumChemistryResults, conv_results: ConvergenceResults):
    """Save convergence dataframe and a small summary CSV."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    conv_results.df.to_csv(Path(data_dir) / 'h6_qsci_convergence.csv', index=False)

    summary_data = {
        'bond_length': qc_results.bond_length,
        'rhf_energy': qc_results.rhf.e_tot,
        'uhf_energy': qc_results.uhf.e_tot,
        'fci_energy': qc_results.fci_energy,
        'n_fci_configs': qc_results.n_fci_configs,
        'n_configs_below_uhf': conv_results.n_configs_below_uhf if conv_results.n_configs_below_uhf else 'Never',
        'n_configs_reach_fci': conv_results.n_configs_reach_fci if conv_results.n_configs_reach_fci else 'Never',
        'max_subspace_size': conv_results.max_size,
        'min_qsci_energy': conv_results.df['qsci_energy'].min(),
        'energy_diff_to_fci': conv_results.df['qsci_energy'].min() - qc_results.fci_energy
    }
    summary_df = pd.DataFrame(list(summary_data.items()), columns=['quantity', 'value'])
    summary_df.to_csv(Path(data_dir) / 'h6_summary.csv', index=False)


def setup_data_directory(base: Optional[Path] = None) -> Path:
    """Create and return a data directory adjacent to this module (or under base if given)."""
    if base is None:
        data_dir = Path(__file__).parent.parent / 'research_data' / Path(__file__).stem
    else:
        data_dir = Path(base)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


__all__ = [
    'QuantumChemistryResults', 'ConvergenceResults', 'calculate_convergence_data',
    'calc_fci_energy', 'calc_fci_subspace_energy', 'calc_qsci_energy_with_size',
    'fci_to_fock_space', 'run_quantum_chemistry_calculations',
    'plot_convergence_comparison', 'plot_energy_vs_samples', 'plot_statevector_coefficients',
    'save_convergence_data', 'setup_data_directory'
]

