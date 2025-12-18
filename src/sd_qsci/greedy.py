from __future__ import annotations

from typing import Iterable, Optional, Tuple, List

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh


def _lowest_eigenvalue(H_sub) -> float:
    """
    Compute the smallest eigenvalue of a (projected) Hamiltonian.

    Handles both dense and sparse inputs. Falls back to dense routine for
    very small matrices where it is typically faster and more robust.
    """
    n = H_sub.shape[0]
    if n <= 2:
        # Convert sparse to dense if needed
        A = H_sub.toarray() if issparse(H_sub) else H_sub
        vals = eigh(A, eigvals_only=True)
        return float(vals.min())
    # Use sparse solver for efficiency on larger subspaces
    vals = eigsh(H_sub, k=1, which='SA', return_eigenvectors=False)
    return float(vals[0])


def greedy_best_subspace(
    H,
    fci_vec: np.ndarray,
    k_max: int,
    amp_thresh: float = 1e-5,
) -> Tuple[List[int], List[float]]:
    """
    Build a subspace greedily by adding one configuration at a time to
    minimise the lowest eigenvalue of the projected Hamiltonian.

    At iteration k, among the remaining candidate configurations (filtered by
    |fci_vec[i]| >= amp_thresh), choose the one that yields the smallest
    ground-state energy when added to the current subspace, then repeat.

    Parameters
    ----------
    H : array-like or sparse matrix (n x n)
        Full Hamiltonian in the determinant/Fock basis used by ``fci_vec``.
    fci_vec : np.ndarray (n,)
        Ground-state FCI coefficients mapped to the same basis as H.
    k_max : int
        Target subspace size to reach.
    amp_thresh : float, optional
        Only determinants with |fci_vec[i]| >= amp_thresh are considered.

    Returns
    -------
    selected : list[int]
        Ordered indices of selected configurations (length <= k_max).
    energies : list[float]
        Sequence of lowest eigenvalues for each subspace size from 1..len(selected).
    """
    n = len(fci_vec)
    if k_max <= 0:
        return [], []

    # Candidate pool filtered by amplitude threshold
    candidates = np.flatnonzero(np.abs(fci_vec) >= amp_thresh).tolist()
    if not candidates:
        return [], []

    # Clamp k_max to available candidates
    k_target = min(k_max, len(candidates))

    selected: List[int] = []
    energies: List[float] = []

    # Pre-compute diagonal for a fast first pick if desired
    # The 1x1 projected energy is simply H[ii]
    diag = np.array(H.diagonal() if hasattr(H, 'diagonal') else np.diag(H))

    # Iteration 1: pick determinant with minimal diagonal energy among candidates
    first_idx = int(min(candidates, key=lambda i: diag[i]))
    selected.append(first_idx)
    energies.append(float(diag[first_idx]))

    # Remaining iterations
    while len(selected) < k_target:
        best_j: Optional[int] = None
        best_E: Optional[float] = None

        remaining = [j for j in candidates if j not in selected]
        if not remaining:
            break

        for j in remaining:
            trial = selected + [j]
            H_sub = H[np.ix_(trial, trial)]
            E0 = _lowest_eigenvalue(H_sub)
            if (best_E is None) or (E0 < best_E):
                best_E = E0
                best_j = j

        # Update selection with the best candidate of this iteration
        assert best_j is not None and best_E is not None
        selected.append(best_j)
        energies.append(best_E)

    return selected, energies


def greedy_from_results(qc_results, k_max: int, amp_thresh: float = 1e-5) -> Tuple[List[int], List[float]]:
    """
    Convenience wrapper to run greedy selection directly from
    ``QuantumChemistryResults``.

    Parameters
    ----------
    qc_results : sd_qsci.analysis.QuantumChemistryResults
        Container with Hamiltonian ``H`` and full-space FCI vector ``fci_vec``.
    k_max : int
        Target subspace size.
    amp_thresh : float, optional
        Amplitude threshold for candidate filtering. Default 1e-5.

    Returns
    -------
    (selected, energies) as in ``greedy_best_subspace``.
    """
    return greedy_best_subspace(qc_results.H, qc_results.fci_vec, k_max, amp_thresh)
