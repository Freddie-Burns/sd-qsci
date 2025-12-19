import pyci
from pyci.test import datafile  # optional, data file location for tests


def main():
    # load system information
    filename = datafile("be_ccpvdz.fcidump")
    occs = (2,2)                        # (alpha, beta) occupation numbers
    ham = pyci.hamiltonian(filename)    # initialize second-quantized operator instance
    e_dict = {}
    hartree_fock(ham, occs, e_dict)
    fci(ham, occs, e_dict)
    hci(ham, occs, e_dict)


def hartree_fock(ham, occs, e_dict):
    # contruct empty fci wave function class instance from # of basis functions and occupation
    wfn0 = pyci.fullci_wfn(ham.nbasis, *occs)
    wfn0.add_hartreefock_det()        # add Hartree-Fock determinant to wave function

    # initialize sparse matrix operator (hamiltonian into wave function)
    op = pyci.sparse_op(ham, wfn0)

    # solve for the lowest eigenvalue and eigenvector
    e_vals, e_vecs0 = op.solve(n=1, tol=1.0e-9)
    e_dict["HF"] = (len(wfn0), e_vals[0])

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

    print("Ground state wave function: HCI")
    print(f"Number of determinants: {len(wfn5)}")
    print(f"HCI energy: {e_vals[0]:.8f} a.u.")
    print(f"Number of iterations used: {niter}")


if __name__ == "__main__":
    main()
