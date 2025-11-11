"""
Hamiltonian construction and manipulation for quantum systems.

This module provides tools for building and working with many-body Hamiltonians
in the fermionic second quantization formalism.

Main Entry Points
-----------------
hamiltonian_from_pyscf : Build Hamiltonian directly from PySCF output
    Converts a PySCF RHF molecule object to a full Fock-space Hamiltonian matrix.
    Handles spin-orbital expansion and integral transformations automatically.

hamiltonian_matrix : Build Hamiltonian from pre-computed integrals
    Lower-level entry point for users who already have MO integrals in the
    desired format. Requires one- and two-electron integrals.

Utilities
---------
ladder_operators : Generate fermionic ladder operators
    Creates annihilation and creation operator matrices for manipulating
    fermionic states. Useful for manual Hamiltonian construction or
    custom analysis.

rhf_energy_from_mo_integrals : Compute RHF energy from MO integrals
    Validation function to verify that MO integrals are correct by
    reproducing the RHF energy.

Examples
--------
Build Hamiltonian from PySCF:

>>> from pyscf import gto, scf
>>> from sd-qsci.hamiltonian import hamiltonian_from_pyscf
>>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g')
>>> rhf = scf.RHF(mol).run()
>>> H = hamiltonian_from_pyscf(mol, rhf)

Build Hamiltonian from integrals:

>>> from sd-qsci.hamiltonian import hamiltonian_matrix
>>> H = hamiltonian_matrix(h1, g2_phys, enuc=0.0)
"""

from .pyscf_glue import hamiltonian_from_pyscf
from .fermion_hamiltonian import hamiltonian_matrix, ladder_operators
from .checks import rhf_energy_from_mo_integrals

__all__ = [
    "hamiltonian_from_pyscf",
    "hamiltonian_matrix",
    "ladder_operators",
    "rhf_energy_from_mo_integrals",
]
