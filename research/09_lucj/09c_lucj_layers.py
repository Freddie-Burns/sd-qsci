"""
LUCJ layers study (09c)
-----------------------

Plot log energy error to FCI for LUCJ circuits with layers 1–10 against
increasing subspace size (spin-symmetric QSCI), using a plasma colormap and a
lower bound of 1e-5 Ha on the y-axis. Also plot the FCI subspace energy vs
subspace size.

Outputs are saved under:
    research/09_lucj/data/09c_lucj_layers/bond_length_2.00/
"""

from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, scf
from pyscf.cc import CCSD
from qiskit_aer import Aer

from sd_qsci import analysis, circuit, hamiltonian
from sd_qsci.utils import uhf_from_rhf


# Plot theme
sns.set_theme()


def build_h_chain(bond_length: float, n_atoms: int = 6) -> gto.Mole:
    coords = [(i * bond_length, 0.0, 0.0) for i in range(n_atoms)]
    geometry = '; '.join([f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in coords])
    mol = gto.Mole()
    mol.build(atom=geometry, unit='Angstrom', basis='sto-3g', charge=0, spin=0, verbose=0)
    return mol


def main():
    bond_length = 2.0
    n_atoms = 6

    stem = Path(__file__).stem
    out_dir = Path(__file__).parent / 'data' / stem / f"bond_length_{bond_length:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building H{n_atoms} chain, bond length {bond_length:.2f} Å …")
    mol = build_h_chain(bond_length, n_atoms)
    rhf = scf.RHF(mol).run()
    uhf = uhf_from_rhf(mol, rhf)
    ccsd = CCSD(rhf).run()

    # Reference quantities
    fci_energy, n_fci_configs, fci_vec = analysis.calc_fci_energy(rhf)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)

    backend = Aer.get_backend("statevector_simulator")

    # Collect per-layer convergence data (spin-symmetric QSCI energy vs subspace size)
    layer_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    max_subspace = 0

    for n_layers in range(1, 11):
        print(f"Simulating LUCJ with {n_layers} layer(s)…")
        qc = circuit.get_lucj_circuit(ccsd_obj=ccsd, backend=backend, n_reps=n_layers)
        sv = circuit.simulate(qc)
        spin_symm_amp = analysis.spin_symm_amplitudes(sv.data)

        qc_results = analysis.QuantumChemistryResults(
            mol=mol,
            rhf=rhf,
            uhf=uhf,
            sv=sv,
            H=H,
            fci_energy=fci_energy,
            n_fci_configs=n_fci_configs,
            fci_vec=fci_vec,
            bond_length=bond_length,
            spin_symm_amp=spin_symm_amp,
        )

        conv = analysis.calc_convergence_data(qc_results, spin_symm=True)

        subspace = conv.df['subspace_size'].to_numpy()
        # Absolute error vs full FCI for spin-symmetric QSCI energy
        err = np.abs(conv.df['spin_symm_energy'].to_numpy() - fci_energy)

        layer_results[n_layers] = (subspace, err)
        if subspace.size > 0:
            max_subspace = max(max_subspace, int(subspace.max()))

    # Compute FCI subspace energies up to the maximum subspace encountered
    fci_sub_sizes = np.arange(1, max_subspace + 1, dtype=int)
    fci_sub_energies = np.array([
        analysis.calc_fci_subspace_energy(H, fci_vec, int(k)) for k in fci_sub_sizes
    ], dtype=float)

    # Plot 1: log error (spin-symmetric LUCJ-QSCI) vs subspace size, layers colored by plasma
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_yscale('log')
    ax.set_xlabel('Subspace Size (Number of Configurations)')
    ax.set_ylabel('Absolute energy error to FCI (Ha) [log]')
    ax.set_title(f'H6 Chain: LUCJ layers (1–10) error vs subspace size\nBond length = {bond_length:.2f} Å')
    ax.set_ylim(bottom=1e-5)

    cmap = plt.cm.get_cmap('plasma', 10)
    for n_layers, (x, y) in layer_results.items():
        # enforce lower bound visually (without modifying underlying values)
        y_plot = np.maximum(y, 1e-5)
        ax.plot(x, y_plot, marker='o', linestyle='-', linewidth=1.8, markersize=3.5,
                color=cmap(n_layers - 1), label=f'{n_layers} layer' + ('s' if n_layers > 1 else ''))

    ax.legend(title='LUCJ layers', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'lucj_layers_error_vs_subspace.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: FCI subspace energy vs subspace size, with full FCI energy reference
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(fci_sub_sizes, fci_sub_energies, 's-', linewidth=1.8, markersize=3.5, label='FCI subspace energy')
    ax2.axhline(y=fci_energy, linestyle='--', linewidth=2.0, color='black', label=f'FCI: {fci_energy:.6f} Ha')
    ax2.set_xlabel('Subspace Size (Number of Configurations)')
    ax2.set_ylabel('Energy (Ha)')
    ax2.set_title(f'H6 Chain: FCI subspace energy vs subspace size\nBond length = {bond_length:.2f} Å')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_dir / 'fci_subspace_energy_vs_subspace.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"Saved plots to: {out_dir}")


if __name__ == '__main__':
    main()
