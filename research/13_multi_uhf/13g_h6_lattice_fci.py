"""
Look at the spin frustrated h6 triagonal lattice for both singlet and
triplet states to see which is the ground.
This will hopefully help to guide my orbital rotation calculations.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyscf import gto, fci

# Setup
bond_length = 2.0
n_atoms = 6
print(f"Running H{n_atoms} triangular lattice bond length: {bond_length:.2f} Angstrom")

# Save under this script's full stem name inside this script's folder
stem = Path(__file__).stem
data_dir = Path(__file__).parent / 'data' / stem
data_dir.mkdir(parents=True, exist_ok=True)
data_file = data_dir / "energies.npz"

bond_lengths = np.linspace(0.5, 3, 26)

if data_file.exists():
    print(f"Loading data from {data_file}")
    data = np.load(data_file)
    existing_bond_lengths = data['bond_lengths'].tolist()
    singlet_energies_map = dict(zip(existing_bond_lengths, data['singlet'].tolist()))
    triplet_energies_map = dict(zip(existing_bond_lengths, data['triplet'].tolist()))
    quintet_energies_map = dict(zip(existing_bond_lengths, data['quintet'].tolist()))
    septet_energies_map = dict(zip(existing_bond_lengths, data['septet'].tolist()))
else:
    singlet_energies_map = {}
    triplet_energies_map = {}
    quintet_energies_map = {}
    septet_energies_map = {}

for a in bond_lengths:
    # Use a small tolerance for float comparison
    match = None
    for existing_a in singlet_energies_map.keys():
        if np.isclose(a, existing_a):
            match = existing_a
            break

    if match is not None:
        print(f"Using cached data for bond length: {a:.2f}")
        continue

    print(f"Calculating for bond length: {a:.2f}")
    h = a * np.sqrt(3) / 2
    geometry = f'''
        H 0 0 {0*a}
        H 0 0 {2*a}
        H 0 0 {4*a}
        H 0 {h} {1*a}
        H 0 {h} {3*a}
        H 0 {2*h} {2*a}
        '''

    singlet = gto.M(
        atom=geometry,
        basis='sto-3g',
        spin=0,
        charge=0
    )
    triplet = gto.M(
        atom=geometry,
        basis='sto-3g',
        spin=2,
        charge=0
    )
    quintet = gto.M(
        atom=geometry,
        basis='sto-3g',
        spin=4,
        charge=0
    )
    septet = gto.M(
        atom=geometry,
        basis='sto-3g',
        spin=6,
        charge=0
    )

    # First run mean-field calculations
    mf_singlet = singlet.RHF().run()
    mf_triplet = triplet.ROHF().run()  # Use ROHF for open-shell
    mf_quintet = quintet.ROHF().run()  # Use ROHF for open-shell
    mf_septet  = septet.ROHF().run()   # Use ROHF for open-shell

    # Run FCI on singlet
    cisolver_singlet = fci.FCI(mf_singlet)
    e_singlet, ci_singlet = cisolver_singlet.kernel()
    # Run FCI on triplet
    cisolver_triplet = fci.FCI(mf_triplet)
    e_triplet, ci_triplet = cisolver_triplet.kernel()
    # Run FCI on quintet
    cisolver_quintet = fci.FCI(mf_quintet)
    e_quintet, ci_quintet = cisolver_quintet.kernel()
    # Run FCI on septet
    cisolver_septet = fci.FCI(mf_septet)
    e_septet, ci_septet = cisolver_septet.kernel()

    # Compare energies
    print(f"Singlet FCI energy: {e_singlet:.8f} Ha")
    print(f"Triplet FCI energy: {e_triplet:.8f} Ha")
    print(f"Quintet FCI energy: {e_quintet:.8f} Ha")
    print(f"Septet FCI energy:  {e_septet:.8f} Ha")

    singlet_energies_map[a] = e_singlet
    triplet_energies_map[a] = e_triplet
    quintet_energies_map[a] = e_quintet
    septet_energies_map[a] = e_septet

# Sort results by bond length for saving and plotting
sorted_bond_lengths = sorted(singlet_energies_map.keys())
singlet_energies = [singlet_energies_map[b] for b in sorted_bond_lengths]
triplet_energies = [triplet_energies_map[b] for b in sorted_bond_lengths]
quintet_energies = [quintet_energies_map[b] for b in sorted_bond_lengths]
septet_energies = [septet_energies_map[b] for b in sorted_bond_lengths]

# Save data
np.savez(data_file, bond_lengths=sorted_bond_lengths, singlet=singlet_energies, triplet=triplet_energies, quintet=quintet_energies, septet=septet_energies)

# Plotting
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
plt.plot(sorted_bond_lengths, singlet_energies, 'o-', linewidth=2, label='Singlet (S=0)')
plt.plot(sorted_bond_lengths, triplet_energies, 's--', markersize=4, label='Triplet (S=1)')
plt.plot(sorted_bond_lengths, quintet_energies, '^-', label='Quintet (S=2)')
plt.plot(sorted_bond_lengths, septet_energies, 'd-', label='Septet (S=3)')

plt.xlabel('Bond Length (Angstrom)')
plt.ylabel('Energy (Ha)')
plt.title(f'H{n_atoms} Triangular Lattice FCI Energy vs Bond Length')
plt.legend()
plt.grid(True)

plot_path = data_dir / "energy_vs_bond_length.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
plt.show()

