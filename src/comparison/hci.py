from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tempfile
import os

import pyci
from pyscf import scf, tools


def run_hci(
    mol,
    eps: float = 1.0e-4,
    max_cycles: Optional[int] = None,
    tol: float = 1.0e-9,
) -> Dict[str, object]:
    """
    Run Heat-bath CI (HCI) starting from RHF for a given PySCF molecule.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        A built PySCF molecule object.
    eps : float, optional
        Heat-bath threshold used by `pyci.add_hci` to select connected determinants.
    nelec : tuple[int, int] | None, optional
        Number of alpha and beta electrons `(na, nb)`. If None, defaults to `mol.nelec`.
    max_cycles : int | None, optional
        Maximum number of HCI expansion/solve cycles. If None, iterate until no new determinants are added.
    tol : float, optional
        Eigen-solver tolerance passed to `op.solve`.

    Returns
    -------
    dict
        A dictionary containing:
        - `eps`: the eps used
        - `series`: list of dicts with keys `iteration`, `ndeterminants`, `energy_ha`
        - `hf_energy_ha`: RHF total energy (Hartree)
        - `final_energy_ha`: final HCI ground-state energy (Hartree)
        - `final_ndeterminants`: number of determinants in the final wavefunction

    Notes
    -----
    - RHF is used to generate molecular orbitals; integrals are written to a temporary
      FCIDUMP file which is then read by `pyci.hamiltonian`.
    - The initial point (iteration 0) corresponds to the Hartree–Fock determinant only.
    """

    # Run RHF to obtain MO basis and energy
    mf = scf.RHF(mol).run()
    e_rhf = float(mf.e_tot)

    # Prepare FCIDUMP in a temporary file for pyci
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".fcidump", prefix="pyci_")
    os.close(tmp_fd)

    # Try block ensures temp file is deleted even if an exception is raised
    try:
        tools.fcidump.from_scf(mf, tmp_path)
        ham = pyci.hamiltonian(tmp_path)

        # Initialize wavefunction with Hartree–Fock determinant
        wfn = pyci.fullci_wfn(ham.nbasis, *mol.nelec)
        wfn.add_hartreefock_det()

        # Create sparse operator and solve initial problem
        op = pyci.sparse_op(ham, wfn)
        e_vals, e_vecs = op.solve(n=1, tol=tol)

        # Record initial HF-only point
        series: List[Dict[str, float | int]] = [
            {"iteration": 0, "ndeterminants": int(len(wfn)), "energy_ha": float(e_vals[0])}
        ]

        niter = 0
        while True:
            if max_cycles is not None and niter >= max_cycles:
                break

            dets_added = pyci.add_hci(ham, wfn, e_vecs[0], eps=eps)
            if not dets_added:
                break

            op.update(ham, wfn)
            e_vals, e_vecs = op.solve(n=1, tol=tol)
            niter += 1
            series.append({
                "iteration": niter,
                "ndeterminants": int(len(wfn)),
                "energy_ha": float(e_vals[0])
            })

        result: Dict[str, object] = {
            "eps": float(eps),
            "series": series,
            "hf_energy_ha": e_rhf,
            "final_energy_ha": float(series[-1]["energy_ha"]),
            "final_ndeterminants": int(series[-1]["ndeterminants"]),
        }
        return result

    # Clean the temporary FCIDUMP file
    finally:
        try: os.remove(tmp_path)
        except OSError: pass


__all__ = ["run_hci"]
