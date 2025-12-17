import numpy as np
from math import log2
from pyscf import fci
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import Statevector
from sd_qsci import spin


def fci_energy(rhf):
    """
    Calculate Full Configuration Interaction (FCI) energy.

    Parameters
    ----------
    rhf : scf.RHF
        Converged RHF calculation object.

    Returns
    -------
    float
        FCI ground state energy in Hartree.
    """
    ci_solver = fci.FCI(rhf)
    fci_energy, fci_vec = ci_solver.kernel()
    return fci_energy


def qsci_energy(
    H: csr_matrix,
    statevector: Statevector,
    enforce_singlet: bool = False,
    singlet_tol: float = 1e-6,
):
    """
    Calculate Quantum Subspace Configuration Interaction (QSCI) energy.

    Extracts the significant configurations from the statevector (based on
    amplitude threshold), constructs the Hamiltonian in this reduced subspace,
    and solves for the ground state energy.

    Parameters
    ----------
    H : scipy.sparse matrix
        Full Hamiltonian matrix in the computational basis (Fock space).
    statevector : circuit.Statevector
        Quantum statevector with amplitudes for all basis configurations.

    Returns
    -------
    E0 : float
        QSCI ground state energy in Hartree.
    idx : np.ndarray
        Array of configuration indices used in the QSCI subspace.

    Notes
    -----
    - Configurations with |amplitude| < 1e-12 are filtered out
    - For small subspaces (≤2 dimensions), uses dense eigenvalue solver
    - For larger subspaces, uses sparse eigenvalue solver (eigsh)
    """
    idx = np.argwhere(np.abs(statevector.data) > 1e-12).ravel()
    H_sub = H[np.ix_(idx, idx)]

    # Gather candidates
    if H_sub.shape[0] <= 2:
        evals, evecs = eigh(H_sub.toarray())
        candidates = [(float(evals[i]), evecs[:, i]) for i in range(len(evals))]
    else:
        k = 1 if not enforce_singlet else min(max(2, 5), H_sub.shape[0] - 1)
        vals, vecs = eigsh(H_sub, k=k, which='SA')
        order = np.argsort(vals)
        candidates = [(float(vals[i]), vecs[:, i]) for i in order]

    # Default selection
    E0, psi0_sub = candidates[0]

    if enforce_singlet:
        n_bits = int(log2(len(statevector.data)))
        n_spatial = n_bits // 2
        S2 = spin.total_spin_S2(n_spatial)
        full_dim = len(statevector.data)
        for E, psi_sub in candidates:
            psi_full = np.zeros(full_dim, dtype=complex)
            psi_full[idx] = psi_sub
            s2 = spin.expectation(S2, psi_full)
            if abs(s2.real) <= singlet_tol:
                E0, psi0_sub = E, psi_sub
                break

    return E0, idx