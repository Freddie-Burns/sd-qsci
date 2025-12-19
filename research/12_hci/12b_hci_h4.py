import os
import math
import pyci
from pyscf import gto, scf, tools, fci as pyscf_fci


RECALCULATE = True


def main():
    # run_h4()
    run_h6()


def run_h6():
    # Build H6 (linear chain) system information using PySCF instead of test data
    here = os.path.dirname(__file__)
    filename = os.path.join(here, "h6_sto3g.fcidump")
    if RECALCULATE or not os.path.exists(filename):
        _write_h6_fcidump(filename, bond_length=2.0, basis="sto-3g")

    occs = (3, 3)                        # (alpha, beta) occupation numbers for H4
    ham = pyci.hamiltonian(filename)     # initialize second-quantized operator instance
    e_dict = {}
    hartree_fock(ham, occs, e_dict)
    fci(ham, occs, e_dict)
    hci(ham, occs, e_dict)


def run_h4():
    # Build H4 (linear chain) system information using PySCF instead of test data
    here = os.path.dirname(__file__)
    filename = os.path.join(here, "h4_sto3g.fcidump")
    if RECALCULATE or not os.path.exists(filename):
        _write_h4_fcidump(filename, bond_length=2.0, basis="sto-3g")

    occs = (2, 2)                        # (alpha, beta) occupation numbers for H4
    ham = pyci.hamiltonian(filename)     # initialize second-quantized operator instance
    e_dict = {}
    hartree_fock(ham, occs, e_dict)
    fci(ham, occs, e_dict)
    hci(ham, occs, e_dict)


def _write_h4_fcidump(path: str, bond_length: float = 1, basis: str = "sto-3g") -> None:
    """Create an H4 linear chain with PySCF and write its FCIDUMP to the given path.

    Parameters:
        path: Output file path for FCIDUMP
        bond_length: H–H spacing in Angstrom along x-axis
        basis: Basis set name for PySCF
    """
    # Define a linear H4 chain along the x-axis: (0, 0, 0), (b, 0, 0), (2b, 0, 0), (3b, 0, 0)
    atoms = "; ".join([f"H {i * bond_length:.8f} 0.0 0.0" for i in range(4)])
    mol = gto.Mole()
    mol.build(atom=atoms, unit="Angstrom", basis=basis, spin=0, charge=0, verbose=0)

    mf = scf.RHF(mol).run()
    e_rhf = mf.e_tot
    cisolver = pyscf_fci.FCI(mol, mf.mo_coeff)
    e_fci, fcivec = cisolver.kernel()

    print("\nPySCF reference energies for H4 chain")
    print(f"  Bond length: {bond_length} Ang  |  Basis: {basis}")
    print(f"  RHF energy: {e_rhf:.8f} Ha")
    print(f"  FCI energy: {e_fci:.8f} Ha")
    print(f"  FCI determinants: {fcivec.size}")

    # Write standard FCIDUMP (one- and two-electron integrals in the RHF MO basis)
    tools.fcidump.from_scf(mf, path)


def _write_h6_fcidump(path: str, bond_length: float = 1, basis: str = "sto-3g") -> None:
    """Create an H6 linear chain with PySCF and write its FCIDUMP to the given path.

    Parameters:
        path: Output file path for FCIDUMP
        bond_length: H–H spacing in Angstrom along x-axis
        basis: Basis set name for PySCF
    """
    # Define a linear H4 chain along the x-axis: (0, 0, 0), (b, 0, 0), (2b, 0, 0), (3b, 0, 0)
    atoms = "; ".join([f"H {i * bond_length:.8f} 0.0 0.0" for i in range(6)])
    mol = gto.Mole()
    mol.build(atom=atoms, unit="Angstrom", basis=basis, spin=0, charge=0, verbose=0)

    mf = scf.RHF(mol).run()
    e_rhf = mf.e_tot
    cisolver = pyscf_fci.FCI(mol, mf.mo_coeff)
    e_fci, fcivec = cisolver.kernel()

    print("\nPySCF reference energies for H4 chain")
    print(f"  Bond length: {bond_length} Ang  |  Basis: {basis}")
    print(f"  RHF energy: {e_rhf:.8f} Ha")
    print(f"  FCI energy: {e_fci:.8f} Ha")
    print(f"  FCI determinants: {fcivec.size}")

    # Write standard FCIDUMP (one- and two-electron integrals in the RHF MO basis)
    tools.fcidump.from_scf(mf, path)


def hartree_fock(ham, occs, e_dict):
    # contruct empty fci wave function class instance from # of basis functions and occupation
    wfn0 = pyci.fullci_wfn(ham.nbasis, *occs)
    wfn0.add_hartreefock_det()        # add Hartree-Fock determinant to wave function

    # initialize sparse matrix operator (hamiltonian into wave function)
    op = pyci.sparse_op(ham, wfn0)

    # solve for the lowest eigenvalue and eigenvector
    e_vals, e_vecs0 = op.solve(n=1, tol=1.0e-9)
    e_dict["HF"] = (len(wfn0), e_vals[0])

    print('\n')
    print("Ground state wave function: HF")
    print("Number of determinants: {}".format(len(wfn0)))
    print("Energy: {}".format(e_vals[0]))


def fci(ham, occs, e_dict):
    wfn1 = pyci.fullci_wfn(ham.nbasis, *occs)
    wfn1.add_all_dets()  # add all determinants to the wave function

    # Solve the CI matrix problem
    op = pyci.sparse_op(ham, wfn1)

    e_vals, e_vecs1 = op.solve(n=1, tol=1.0e-9)
    e_dict["Full-CI"] = (len(wfn1), e_vals[0])

    print('\n')
    print("Ground state wave function: Full CI")
    print(f"Number of determinants: {len(wfn1)}")
    print(f"Full CI energy: {e_vals[0]:.8f} a.u.")


def hci(ham, occs, e_dict):
    wfn5 = pyci.fullci_wfn(ham.nbasis, *occs)

    # Add Hartree-Fock determinant
    wfn5.add_hartreefock_det()
    dets_added = 1

    # Create CI matrix operator and initial Hartree-Fock solution
    op = pyci.sparse_op(ham, wfn5)
    e_vals, e_vecs5 = op.solve(n=1, tol=1.0e-9)

    # Run HCI iterations
    niter = 0
    eps = 5.0e-4
    while dets_added:
        # Add connected determinants to wave function via HCI
        dets_added = pyci.add_hci(ham, wfn5, e_vecs5[0], eps=eps)
        # Update CI matrix operator
        op.update(ham, wfn5)
        # Solve CI matrix problem
        e_vals, e_vecs5 = op.solve(n=1, tol=1.0e-9)
        niter += 1
    e_dict["HCI"] = (len(wfn5), e_vals[0])

    print('\n')
    print("Ground state wave function: HCI")
    print(f"Number of determinants: {len(wfn5)}")
    print(f"HCI energy: {e_vals[0]:.8f} a.u.")
    print(f"Number of iterations used: {niter}")


if __name__ == "__main__":
    main()
