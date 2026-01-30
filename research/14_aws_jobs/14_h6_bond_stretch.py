"""
Stretch an H6 chain and plot the energy vs bond length.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyscf import gto, scf, fci
from sd_qsci import utils

# Setup
STEM = Path(__file__).stem
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / STEM
DATA_DIR.mkdir(parents=True, exist_ok=True)

# UHF patterns
patterns = {
    "antiferromagnetic": [1, -1, 1, -1, 1, -1],  # antiferromagnetic: lowest energy soln
    "ferromagnetic": [1, 1, 1, -1, -1, -1],  # ferromagnetic: higher energy soln
}

# Bond lengths to stretch from 0.5 to 2.5 Angstrom
bond_lengths = np.linspace(0.5, 3.0, 26)
rhf_energies = []
uhf_afm_energies = []
uhf_fm_energies = []
fci_energies = []

print(f"Stretching H6 chain from 0.5 to 2.5 Angstrom...")

for a in bond_lengths:
    # Define H6 in a line with bond length a
    mol = gto.M(
        atom=f'''
        H 0 0 {0*a}
        H 0 0 {1*a}
        H 0 0 {2*a}
        H 0 0 {3*a}
        H 0 0 {4*a}
        H 0 0 {5*a}
        ''',
        basis='sto-3g',
        spin=0,
        charge=0,
        verbose=0
    )
    
    rhf = scf.RHF(mol).run()
    rhf_energies.append(rhf.e_tot)
    
    # UHF antiferromagnetic
    uhf_afm = utils.solve_with_spin_pattern(mol, patterns["antiferromagnetic"], "antiferromagnetic")
    uhf_afm_energies.append(uhf_afm.e_tot)
    
    # UHF ferromagnetic
    uhf_fm = utils.solve_with_spin_pattern(mol, patterns["ferromagnetic"], "ferromagnetic")
    uhf_fm_energies.append(uhf_fm.e_tot)
    
    # FCI
    cisolver = fci.FCI(rhf)
    fci_e, fci_vec = cisolver.kernel()
    fci_energies.append(fci_e)
    
    print(f"Bond length: {a:.2f} A, RHF: {rhf.e_tot:.6f}, UHF-AFM: {uhf_afm.e_tot:.6f}, UHF-FM: {uhf_fm.e_tot:.6f}, FCI: {fci_e:.6f}")

# Convert to numpy arrays
rhf_energies = np.array(rhf_energies)
uhf_afm_energies = np.array(uhf_afm_energies)
uhf_fm_energies = np.array(uhf_fm_energies)
fci_energies = np.array(fci_energies)

# Plotting
plt.figure(figsize=(10, 7))
plt.plot(bond_lengths, rhf_energies, 'o-', label='RHF Energy')
plt.plot(bond_lengths, uhf_afm_energies, 's-', label='UHF Energy (AFM)')
plt.plot(bond_lengths, uhf_fm_energies, '^-', label='UHF Energy (FM)')
plt.plot(bond_lengths, fci_energies, 'x--', label='FCI Energy', color='black', alpha=0.7)
plt.xlabel('Bond Length (Angstrom)')
plt.ylabel('Total Energy (Hartree)')
plt.title('H6 Chain: Energy vs Bond Length')
plt.grid(True)
plt.legend()

# Save the figure
plot_path = DATA_DIR / "h6_bond_stretch.png"
plt.savefig(plot_path)
print(f"\nPlot saved to {plot_path}")

# Save the data
data_path = DATA_DIR / "h6_bond_stretch_data.csv"
np.savetxt(data_path, np.column_stack((bond_lengths, rhf_energies, uhf_afm_energies, uhf_fm_energies, fci_energies)), 
           header="bond_length_angstrom,rhf_energy_hartree,uhf_afm_energy_hartree,uhf_fm_energy_hartree,fci_energy_hartree", 
           delimiter=",", comments="")
print(f"Data saved to {data_path}")
