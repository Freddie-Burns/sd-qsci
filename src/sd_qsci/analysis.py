"""
Analysis utilities for QSCI convergence and statevector handling.

This module extracts reusable functions from the research scripts so they can
be reused for arbitrary molecules/geometries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from math import log2
from typing import Optional

import numpy as np
import pandas as pd
from pyscf import gto, fci, scf
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import Statevector

from sd_qsci.utils import uhf_from_rhf
from sd_qsci import circuit, hamiltonian, spin


# Defaults for tolerances (can be overridden by callers)
DEFAULT_SV_TOL = 1e-16
DEFAULT_FCI_TOL = 1e-16


@dataclass
class QuantumChemistryResults:
    """
    Container for quantum chemistry calculation results.

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
      spin_symm_amp: np.ndarray | None (spin-symmetric amplitudes)
      n_qubits: int | None
      one_q_gates: int | None
      two_q_gates: int | None
    """
    mol: gto.Mole
    rhf: scf.RHF
    uhf: scf.UHF
    sv: Statevector
    H: np.ndarray
    fci_energy: float
    n_fci_configs: int
    fci_vec: np.ndarray
    bond_length: Optional[float] = None
    spin_symm_amp: Optional[np.ndarray] = None
    # Optional metadata about the state preparation circuit
    n_qubits: Optional[int] = None
    one_q_gates: Optional[int] = None
    two_q_gates: Optional[int] = None


@dataclass
class ConvergenceResults:
    """Container for convergence analysis results."""
    df: pd.DataFrame
    max_size: int
    n_configs_below_uhf: Optional[int]
    n_configs_reach_fci: Optional[int]
    n_configs_below_uhf_symm: Optional[int]
    n_configs_reach_fci_symm: Optional[int]


def calc_convergence_data(
    qc_results: QuantumChemistryResults,
    sv_tol: float = DEFAULT_SV_TOL,
    fci_tol: float = DEFAULT_FCI_TOL,
    spin_symm: bool = False,
) -> ConvergenceResults:
    """
    Calculate QSCI and FCI subspace energies for varying subspace sizes.

    Parameters
    ----------
    qc_results : QuantumChemistryResults
        Quantum chemistry calculation results.
    sv_tol : float, optional
        Threshold for considering statevector amplitudes as present.
        Default is DEFAULT_SV_TOL.
    fci_tol : float, optional
        Tolerance for considering a QSCI energy equal to FCI.
        Default is DEFAULT_FCI_TOL.

    Returns
    -------
    ConvergenceResults
        Container with convergence analysis results.
    """
    max_idx = np.argwhere(np.abs(qc_results.sv.data) > sv_tol).ravel()
    max_size = len(max_idx)

    subspace_sizes = list(range(1, max_size + 1))
    qsci_energies = []
    symm_energies = []
    qsci_S2_vals = []
    symm_S2_vals = []
    fci_subspace_energies = []
    mean_sample_numbers = []

    n_configs_below_uhf = None
    n_configs_reach_fci = None
    n_configs_below_uhf_symm = None
    n_configs_reach_fci_symm = None

    print("sv norm:", np.linalg.norm(qc_results.sv.data))
    print("spin symm amp norm:", np.linalg.norm(qc_results.spin_symm_amp))

    if spin_symm:
        data = qc_results.spin_symm_amp
    else:
        data = qc_results.sv.data

    # Build S^2 once for this system
    n_bits = int(log2(len(qc_results.sv.data)))
    n_spatial = n_bits // 2
    S2 = spin.total_spin_S2(n_spatial)

    for size in subspace_sizes:
        # Plain QSCI
        qsci_energy, psi_plain, _ = calc_qsci_energy_with_size(
            qc_results.H, qc_results.sv.data, size, return_vector=True
        )
        qsci_energies.append(qsci_energy)
        qsci_S2_vals.append(float(spin.expectation(S2, psi_plain).real))

        # Spin-symmetric QSCI (diagonalise on spin-symmetrised amplitudes)
        symm_energy, psi_symm, _ = calc_qsci_energy_with_size(
            qc_results.H, qc_results.spin_symm_amp, size, return_vector=True
        )
        symm_energies.append(symm_energy)
        symm_S2_vals.append(float(spin.expectation(S2, psi_symm).real))

        fci_sub_energy = calc_fci_subspace_energy(qc_results.H, qc_results.fci_vec, size)
        fci_subspace_energies.append(fci_sub_energy)

        idx = np.argsort(np.abs(data))[-size:]
        min_coeff = np.min(np.abs(data[idx]))
        mean_sample_number = 1.0 / (min_coeff ** 2)
        mean_sample_numbers.append(mean_sample_number)

        if n_configs_below_uhf is None and qsci_energy < qc_results.uhf.e_tot:
            n_configs_below_uhf = size
        if n_configs_reach_fci is None and abs(qsci_energy - qc_results.fci_energy) < fci_tol:
            n_configs_reach_fci = size
        if n_configs_below_uhf_symm is None and symm_energy < qc_results.uhf.e_tot:
            n_configs_below_uhf_symm = size
        if n_configs_reach_fci_symm is None and abs(symm_energy - qc_results.fci_energy) < fci_tol:
            n_configs_reach_fci_symm = size

    df = pd.DataFrame({
        'subspace_size': subspace_sizes,
        'qsci_energy': qsci_energies,
        'qsci_S2': qsci_S2_vals,
        'spin_symm_energy': symm_energies,
        'spin_symm_S2': symm_S2_vals,
        'fci_subspace_energy': fci_subspace_energies,
        'mean_sample_number': mean_sample_numbers
    })

    return ConvergenceResults(
        df=df,
        max_size=max_size,
        n_configs_below_uhf=n_configs_below_uhf,
        n_configs_reach_fci=n_configs_reach_fci,
        n_configs_below_uhf_symm=n_configs_below_uhf_symm,
        n_configs_reach_fci_symm=n_configs_reach_fci_symm,
    )


def calc_fci_energy(rhf, tol: float = 1e-10) -> tuple[float, int, np.ndarray]:
    """
    Compute FCI ground state and map PySCF's CI vector into the full Fock space.

    Parameters
    ----------
    rhf : scf.RHF
        Restricted Hartree-Fock object.
    tol : float, optional
        Tolerance for considering an FCI amplitude as nonzero. Default is 1e-10.

    Returns
    -------
    fci_energy : float
        FCI ground state energy.
    n_configs : int
        Number of significant FCI configurations.
    fci_vec : np.ndarray
        Full Fock-space FCI vector.
    """
    ci_solver = fci.FCI(rhf)
    fci_energy, fci_vec_ci = ci_solver.kernel()

    mol = rhf.mol
    nelec = mol.nelec

    fci_vec = fci_to_fock_space(fci_vec_ci, mol, nelec)
    n_configs = int(np.count_nonzero(np.abs(fci_vec) > tol))

    return fci_energy, n_configs, fci_vec

# Todo move to comparison directory
def calc_fci_subspace_energy(H: np.ndarray, fci_vec: np.ndarray, n_configs: int):
    """
    Energy of subspace spanned by the n_configs largest-amplitude FCI configurations.

    Parameters
    ----------
    H : np.ndarray or sparse matrix
        Full Hamiltonian matrix.
    fci_vec : np.ndarray
        Full Fock-space FCI vector.
    n_configs : int
        Number of configurations to include in the subspace.

    Returns
    -------
    float
        Ground state energy of the subspace.
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


def calc_qsci_energy_with_size(
    H,
    data: np.ndarray,
    n_configs: int,
    return_vector: bool = False,
    spin_symmetry: bool = False,
    enforce_singlet: bool = False,
    singlet_tol: float = 1e-6,
):
    """
    Compute QSCI energy by diagonalising Hamiltonian in a subspace.

    Diagonalises the Hamiltonian restricted to the largest n_configs components
    of the provided statevector.

    Parameters
    ----------
    H : np.ndarray or sparse matrix
        Full Hamiltonian matrix.
    data : np.ndarray
        Quantum statevector data from circuit simulation.
    n_configs : int
        Number of configurations (largest amplitudes) to include in the subspace.
    return_vector : bool, optional
        If True, also return the full-space vector and configuration indices.
        Default is False.

    Returns
    -------
    float
        Ground state energy of the subspace.
    np.ndarray, optional
        Full-space statevector (returned only if return_vector is True).
    np.ndarray, optional
        Indices of configurations in the subspace (returned only if return_vector is True).
    """
    # n configs ordered by highest amplitude first
    idx = np.argsort(np.abs(data))[-n_configs:][::-1]

    if spin_symmetry:
        n_bits = int(log2(len(data)))
        idx = spin_symm_indices(idx, n_bits)[:n_configs]

    H_sub = H[np.ix_(idx, idx)]

    # Solve for the lowest eigenpair only (singlet enforcement removed in analysis)
    N = H_sub.shape[0]
    if N <= 3:
        evals, evecs = eigh(H_sub.toarray() if hasattr(H_sub, 'toarray') else H_sub)
        E0 = float(evals[0])
        psi0 = evecs[:, 0]
    else:
        vals, vecs = eigsh(H_sub, k=1, which='SA')
        E0 = float(vals[0])
        psi0 = vecs[:, 0]

    if return_vector:
        psi0_full = np.zeros(data.shape, dtype=complex)
        psi0_full[idx] = psi0
        return E0, psi0_full, idx

    return E0


def fci_to_fock_space(fci_vec, mol: gto.Mole, nelec) -> np.ndarray:
    """
    Map PySCF's CI vector into the full Fock space vector.

    Converts PySCF's restricted CI vector to a full Fock-space representation
    using BLOCK spin ordering.

    Parameters
    ----------
    fci_vec : np.ndarray
        FCI vector from PySCF.
    mol : gto.Mole
        PySCF molecule object.
    nelec : tuple
        Tuple of (n_alpha, n_beta) electron counts.

    Returns
    -------
    np.ndarray
        Full Fock-space FCI vector.
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


def run_quantum_chemistry_calculations(
        mol: gto.Mole,
        rhf: scf.RHF,
        bond_length: Optional[float],
        statevector: Optional["Statevector | np.ndarray"] = None,
        uhf: Optional[scf.UHF] = None
) -> QuantumChemistryResults:
    """
    Run complete quantum chemistry calculations and circuit simulation.

    Performs UHF calculation, builds orbital-rotation circuit, and constructs
    the full Hamiltonian. By default it simulates the statevector from the
    circuit. Alternatively, a precomputed statevector can be provided to bypass
    simulation (e.g., derived from hardware measurement post-processing).

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object.
    rhf : scf.RHF
        Restricted Hartree-Fock object.
    bond_length : float, optional
        Bond length for reference. Default is None.

    Parameters
    -------
    mol : gto.Mole
        PySCF molecule object.
    rhf : scf.RHF
        Restricted Hartree-Fock object.
    bond_length : float, optional
        Bond length for reference. Default is None.
    statevector : qiskit.quantum_info.Statevector or numpy.ndarray, optional
        If provided, this statevector is used instead of simulating the
        circuit. If a numpy array is provided, it will be wrapped into a
        qiskit Statevector. The array is expected to be L2-normalized.

    Returns
    -------
    QuantumChemistryResults
        Container with all quantum chemistry calculation results.

    Raises
    ------
    RuntimeError
        If orbital rotation verification fails (statevector energy != UHF
        energy) when the statevector is simulated internally.
    """
    if uhf is None:
        uhf = uhf_from_rhf(mol, rhf)
    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)

    # Determine the statevector: either simulate or use provided one
    simulated_internally = statevector is None
    if simulated_internally:
        sv = circuit.simulate(qc)
    else:
        if isinstance(statevector, Statevector):
            sv = statevector
        else:
            # Assume numpy-like array to be wrapped as a Statevector
            sv = Statevector(statevector)

    spin_symm_amp = spin_symm_amplitudes(sv.data)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)

    # Use np.vdot for robust dot product across array shapes
    sv_energy = np.vdot(sv.data, H.dot(sv.data)).real if hasattr(H, 'dot') else np.vdot(sv.data, H @ sv.data).real
    # Only enforce verification when we simulated internally
    if simulated_internally and not np.isclose(sv_energy, uhf.e_tot):
        raise RuntimeError("Orbital rotation verification failed: statevector energy != UHF energy")

    fci_energy, n_fci_configs, fci_vec = calc_fci_energy(rhf)
    n_qubits = int(getattr(qc, 'num_qubits', 0))
    one_q_gates, two_q_gates = count_gates(qc)

    return QuantumChemistryResults(
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
        n_qubits=n_qubits,
        one_q_gates=one_q_gates,
        two_q_gates=two_q_gates,
    )


def save_convergence_data(
    data_dir: Path,
    qc_results: QuantumChemistryResults,
    conv_results: ConvergenceResults,
) -> None:
    """
    Save convergence data and summary to CSV files.

    Saves the convergence dataframe to 'qsci_convergence.csv' and a
    summary of key quantities to 'summary.csv'.

    Parameters
    ----------
    data_dir : Path
        Directory to save the CSV files.
    qc_results : QuantumChemistryResults
        Quantum chemistry calculation results.
    conv_results : ConvergenceResults
        Convergence analysis results.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    conv_results.df.to_csv(Path(data_dir) / 'qsci_convergence.csv', index=False)

    # Chemical accuracy tolerance (in Hartree)
    CHEM_ACCURACY_TOL = 1.6e-3

    # Compute first subspace sizes that reach chemical accuracy for both series
    df = conv_results.df
    fci_E = qc_results.fci_energy

    qsci_abs_diff = (df['qsci_energy'] - fci_E).abs()
    spin_abs_diff = (df['spin_symm_energy'] - fci_E).abs() if 'spin_symm_energy' in df.columns else None

    def first_size_within_tol(abs_diff_series, sizes, tol):
        mask = abs_diff_series <= tol
        if mask.any():
            return int(sizes[mask].iloc[0])
        return 'Never'

    n_configs_chemacc_qsci = first_size_within_tol(qsci_abs_diff, df['subspace_size'], CHEM_ACCURACY_TOL)
    n_configs_chemacc_spin = (
        first_size_within_tol(spin_abs_diff, df['subspace_size'], CHEM_ACCURACY_TOL)
        if spin_abs_diff is not None else 'N/A'
    )

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
        'energy_diff_to_fci': conv_results.df['qsci_energy'].min() - qc_results.fci_energy,
        'chemical_accuracy_tol': CHEM_ACCURACY_TOL,
        'n_configs_chemacc_qsci': n_configs_chemacc_qsci,
        'n_configs_chemacc_spin_symm': n_configs_chemacc_spin,
    }
    # Optionally include circuit gate counts if available (use safe getattr for optional fields)
    n_qubits = getattr(qc_results, 'n_qubits', None)
    one_q_gates = getattr(qc_results, 'one_q_gates', None)
    two_q_gates = getattr(qc_results, 'two_q_gates', None)
    total_gates = getattr(qc_results, 'total_gates', None)

    if n_qubits is not None:
        summary_data['n_qubits'] = int(n_qubits)
    if one_q_gates is not None:
        summary_data['one_q_gates'] = int(one_q_gates)
    if two_q_gates is not None:
        summary_data['two_q_gates'] = int(two_q_gates)
    # Preserve backward compatibility if some scripts provide total_gates; otherwise omit
    if total_gates is not None:
        summary_data['total_gates'] = int(total_gates)
    summary_df = pd.DataFrame(list(summary_data.items()), columns=['quantity', 'value'])
    summary_df.to_csv(Path(data_dir) / 'summary.csv', index=False)


def setup_data_directory(base: Optional[Path] = None) -> Path:
    """
    Create and return a data directory.

    Creates a data directory adjacent to this module, or under the specified
    base directory if provided.

    Parameters
    ----------
    base : Path, optional
        Base directory for data storage. If None, uses a directory relative
        to this module. Default is None.

    Returns
    -------
    Path
        Path to the data directory (created if it doesn't exist).
    """
    if base is None:
        data_dir = Path(__file__).parent.parent / 'research_data' / Path(__file__).stem
    else:
        data_dir = Path(base)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def spin_symm_indices(idx, n_bits):
    new_indices = []
    bitstring = bin(idx)[2:].zfill(n_bits)
    bitstrings = spin.spin_symmetric_configs(bitstring)
    for bs in bitstrings:
        new_indices.append(int(bs, 2))
    return new_indices


def spin_symm_amplitudes(sv_data: np.ndarray) -> np.ndarray:
    """
    Make amplitudes equal to largest for all spin-symmetric configurations.
    """
    sv_data_new = sv_data.copy()
    calculated_indices = set()
    indices = np.argsort(np.abs(sv_data_new))[::-1]
    n_bits = int(np.log2(sv_data_new.size))
    for i in indices:
        if i not in calculated_indices:
            symm_indices = spin_symm_indices(i, n_bits)
            for j in symm_indices:
                calculated_indices.add(j)
                sv_data_new[j] = sv_data[i]
    return sv_data_new


def spin_closed_subspace_sizes(sv_data: np.ndarray) -> list[int]:
    """
    Compute the sequence of subspace sizes that are closed under spin symmetry.

    The subspace growth follows the natural importance ordering given by
    descending absolute value of the raw statevector amplitudes. Whenever a new
    configuration is selected, all of its spin-symmetric partners must be
    included before emitting the next valid size. This ensures that each
    reported size corresponds to a subspace that contains complete spin
    orbits.

    Parameters
    ----------
    sv_data : np.ndarray
        Full statevector amplitudes (complex). The magnitudes are used to
        determine the selection order.

    Returns
    -------
    list[int]
        A strictly increasing list of subspace sizes (counts of unique
        configurations) where each size corresponds to a union of complete
        spin-symmetric sets.
    """
    n_bits = int(np.log2(sv_data.size))
    # Sort indices by decreasing amplitude magnitude
    sorted_idx = np.argsort(np.abs(sv_data))[::-1]

    selected: set[int] = set()
    sizes: list[int] = []

    for idx in sorted_idx:
        if idx in selected:
            continue
        # Add the entire spin-symmetric orbit for this index
        orbit = spin_symm_indices(int(idx), n_bits)
        for j in orbit:
            selected.add(j)
        sizes.append(len(selected))

    return sizes


def count_gates(qc: QuantumCircuit) -> tuple[int, int]:
    """
    Count the number of single-qubit and two-qubit operations in a transpiled
    Qiskit QuantumCircuit.
    """
    tqc = transpile(
        qc,
        backend=Aer.get_backend("aer_simulator"),
        basis_gates=['rz', 'sx', 'cx'],
        optimization_level=3
    )
    one_q = two_q = 0
    for instr, qargs, _ in tqc.data:
        if len(qargs) == 1:
            one_q += 1
        elif len(qargs) == 2:
            two_q += 1
    return one_q, two_q


__all__ = [
    'QuantumChemistryResults',
    'ConvergenceResults',
    'calc_convergence_data',
    'calc_fci_energy',
    'calc_fci_subspace_energy',
    'calc_qsci_energy_with_size',
    'fci_to_fock_space',
    'run_quantum_chemistry_calculations',
    'save_convergence_data',
    'setup_data_directory',
]
