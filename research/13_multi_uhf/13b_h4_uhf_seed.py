import numpy as np
from matplotlib import pyplot as plt
from pyscf import gto, scf

import multi_uhf_utils as utils


# Define H4 in a line
mol = gto.M(
    atom='''
    H 0 0 0
    H 0 0 2
    H 0 0 4
    H 0 0 6
    ''',
    basis='sto-3g',
    spin=0,
    charge=0
)

def compare_solutions(solutions):
    """Compare multiple UHF solutions."""
    for label, mf in solutions.items():
        print(f"\n{label}:")
        print(f"Alpha MO coefficients shape: {mf.mo_coeff[0].shape}")
        print(f"Beta MO coefficients shape: {mf.mo_coeff[1].shape}")

        # Occupied orbitals
        print(f"Alpha occupied orbitals: {mf.mo_occ[0]}")
        print(f"Beta occupied orbitals: {mf.mo_occ[1]}")

        # Energy of occupied orbitals
        nocc_alpha = int(mf.mo_occ[0].sum())
        nocc_beta = int(mf.mo_occ[1].sum())
        print(f"Alpha orbital energies: {mf.mo_energy[0][:nocc_alpha]}")
        print(f"Beta orbital energies: {mf.mo_energy[1][:nocc_beta]}")


def plot_mo_coefficients(mf, title):
    """Plot MO coefficients to visualize the orbitals"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    nao = mf.mo_coeff[0].shape[0]

    # Alpha MOs
    axes[0].imshow(mf.mo_coeff[0], cmap='RdBu', aspect='auto')
    axes[0].set_title(f'{title} - Alpha MOs')
    axes[0].set_xlabel('MO index')
    axes[0].set_ylabel('AO index')
    axes[0].set_yticks(range(nao))

    # Beta MOs
    axes[1].imshow(mf.mo_coeff[1], cmap='RdBu', aspect='auto')
    axes[1].set_title(f'{title} - Beta MOs')
    axes[1].set_xlabel('MO index')
    axes[1].set_ylabel('AO index')
    axes[1].set_yticks(range(nao))

    plt.colorbar(axes[0].images[0], ax=axes[0])
    plt.colorbar(axes[1].images[0], ax=axes[1])
    plt.tight_layout()
    return fig


# Try different spin patterns
patterns = {
    "up up down down": [1, 1, -1, -1],
    "up down down up": [1, -1, -1, 1],
    "up down up down": [1, -1, 1, -1],
    "down up up down": [-1, 1, 1, -1],
    "all up (then relax)": [1, 1, 1, 1],
}

solutions = {}
for label, pattern in patterns.items():
    mf = utils.solve_with_spin_pattern(mol, pattern, label)
    solutions[label] = mf

# Compare all solutions
print("\n" + "=" * 60)
print("SUMMARY OF ALL SOLUTIONS")
print("=" * 60)
for label, mf in solutions.items():
    if mf.converged:
        print(
            f"{label:20s}: E = {mf.e_tot:.8f} Ha, <S^2> = {mf.spin_square()[0]:.4f}")

compare_solutions(solutions)

# Plot each solution
for label, mf in solutions.items():
    fig = plot_mo_coefficients(mf, label)
    plt.savefig(f'mo_coeffs_{label.replace(" ", "_")}.png')
