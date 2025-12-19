import os
from pathlib import Path
import csv
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import pyci
import seaborn as sns
from pyscf import gto, scf, tools, fci as pyscf_fci


RECALCULATE = True
# Use Seaborn's default theme across all plots
sns.set_theme()


def main():
    # Build H6 (linear chain) system information using PySCF instead of test data
    filename = Path(__file__).parent / "h6_sto3g.fcidump"
    bond_length = 2.0
    if RECALCULATE or not os.path.exists(filename):
        _write_h6_fcidump(filename, bond_length=bond_length, basis="sto-3g")

    # (alpha, beta) occupation numbers for H6
    occs = (3, 3)
    ham = pyci.hamiltonian(str(filename))
    e_dict = {}
    hartree_fock(ham, occs, e_dict)
    fci(ham, occs, e_dict)

    # HCI sweep over multiple eps values
    eps_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    # Output directory for results
    outdir = Path(__file__).parent / "data" / "12c_hci" / f"bond_length_{bond_length:.2f}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Use FCI energy as reference for error plots
    e_ref_ha = float(e_dict["Full-CI"][1])

    hci_sweep(ham, occs, eps_values, outdir, e_ref_ha=e_ref_ha)


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
    print(f"Full CI energy: {e_vals[0]:.8f} Ha")


def hci_sweep(ham, occs, eps_values: List[float], outdir: Path, e_ref_ha: Optional[float] = None) -> None:
    """Run HCI for multiple eps thresholds, saving per-iteration energy and ndets, and plot.

    For each eps value, we record after each HCI expansion/solve cycle the current
    number of determinants and the ground-state energy. We write a CSV per eps and
    a combined plot of energy vs determinants overlaying all eps curves.
    """
    all_series: List[Tuple[float, List[Tuple[int, int, float]]]] = []

    for eps in eps_values:
        wfn = pyci.fullci_wfn(ham.nbasis, *occs)
        wfn.add_hartreefock_det()
        dets_added = 1

        op = pyci.sparse_op(ham, wfn)
        e_vals, e_vecs = op.solve(n=1, tol=1.0e-9)

        # Record initial point as iteration 0
        series: List[Tuple[int, int, float]] = [(0, len(wfn), float(e_vals[0]))]

        niter = 0
        while dets_added:
            dets_added = pyci.add_hci(ham, wfn, e_vecs[0], eps=eps)
            if not dets_added:
                break
            op.update(ham, wfn)
            e_vals, e_vecs = op.solve(n=1, tol=1.0e-9)
            niter += 1
            series.append((niter, len(wfn), float(e_vals[0])))

        # Save series for this eps to CSV
        eps_tag = f"{eps:.1e}".replace("+", "").replace("-0", "-")
        csv_path = outdir / f"h6_hci_convergence_eps_{eps_tag}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "ndeterminants", "energy_ha", "eps"])
            for it, nd, e in series:
                writer.writerow([it, nd, f"{e:.12f}", eps])

        print("\nHCI sweep | eps = {}".format(eps))
        print("  Final determinants: {}".format(series[-1][1]))
        print("  Final energy: {:.8f} Ha".format(series[-1][2]))
        print("  Iterations: {}".format(series[-1][0]))

        all_series.append((eps, series))

    # Plot absolute energy error vs determinants for each eps (log y) with chemical accuracy shaded
    fig, ax = plt.subplots(figsize=(7, 5))
    # Default reference: lowest energy found among all points if not provided
    if e_ref_ha is None:
        e_ref_ha = min(e for (_, series) in all_series for (_, __, e) in series)

    min_err = None
    max_err = None
    for eps, series in all_series:
        ndets = [nd for (_, nd, __) in series]
        # Signed variational error vs reference (should be ≥ 0)
        errs = [(e - e_ref_ha) for (_, __, e) in series]
        ax.plot(ndets, errs, marker="o", linewidth=1.5, markersize=3, label=f"eps={eps:.1e}")
        smin = min(errs)
        smax = max(errs)
        min_err = smin if min_err is None else min(min_err, smin)
        max_err = smax if max_err is None else max(max_err, smax)

    # Log scale on y
    ax.set_yscale('log')

    ax.set_xlabel("Number of determinants")
    ax.set_ylabel("Energy error vs FCI (Ha)")
    ax.set_title("HCI energy error vs determinants for varying eps (H6, STO-3G)")
    # ax.grid(True, which='both', alpha=0.3)

    # Shade chemical accuracy region (1 kcal/mol) below threshold
    chem_acc_ha = 1.6e-3
    # Ensure limits are reasonable before shading
    if min_err is not None and max_err is not None:
        bottom = max(min_err * 0.5, 1e-12)
        top = max(max_err * 1.2, chem_acc_ha * 2)
        ax.set_ylim(bottom=bottom, top=top)

    y0, y1 = ax.get_ylim()
    ax.axhspan(y0, min(chem_acc_ha, y1), color='tab:green', alpha=0.15, label='Chemical accuracy (1 kcal/mol)')

    ax.legend()
    fig.tight_layout()
    plot_path = outdir / "h6_hci_error_vs_determinants_by_eps.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
