"""
H6 spin-symmetric QSCI: singlet-enforcement comparison
======================================================

Purpose
-------
Compare spin-symmetric QSCI energies obtained without and with explicit
singlet enforcement when diagonalising the Hamiltonian in an amplitude-
selected subspace for a 6-hydrogen chain at a fixed bond length.

What this script does
---------------------
1. Runs RHF, UHF and FCI with PySCF for H6 (STO-3G).
2. Builds the UHF statevector via the orbital-rotation circuit and computes
   its spin-symmetrised amplitudes.
3. Baseline: reproduces the usual convergence analysis (no singlet enforcing)
   for both plain QSCI and spin-symmetric QSCI.
4. New comparison: for the same subspace sizes, recomputes the spin-symmetric
   QSCI energies but enforces a spin singlet by checking ⟨S²⟩ ≈ 0 and rejecting
   non‑singlet eigenvectors.

Outputs
-------
Baseline (unchanged):
- h6_qsci_convergence.csv, h6_summary.csv
- h6_qsci_convergence.png, h6_energy_vs_samples.png
- statevector_coefficients.png, statevector_coefficients_full.png

New comparison artifacts:
- spin_symm/spin_symm_enforce_compare.csv: spin-symmetric energies with and
  without enforcing singlet, across subspace sizes.
- spin_symm_enforce_compare.png: plot showing both spin-symmetric series plus
  reference UHF/FCI lines.

Notes
-----
- Singlet enforcing is applied only to the spin-symmetric series; the input
  amplitudes are already spin-symmetrised, so the diagonalisation uses those
  directly and verifies ⟨S²⟩ ≈ 0 within a tolerance.
- BLOCK spin ordering: [α0…α(M−1), β0…β(M−1)].
"""


from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, scf

from sd_qsci import analysis, circuit, plot
from sd_qsci.utils import uhf_from_rhf


def main():
    """
    Run H6 chain energy calculations and analyze QSCI convergence.
    """
    bond_length = 2
    n_atoms = 6
    print(f"Running H{n_atoms} chain bond length: {bond_length:.2f} Angstrom")
    run_full_analysis(bond_length, n_atoms)


def run_full_analysis(bond_length, n_atoms):
    # Save under this script's full stem name inside this script's folder
    stem = Path(__file__).stem
    data_dir = Path(__file__).parent / 'data' / stem

    # Run quantum chemistry calculations
    mol = build_h_chain(bond_length, n_atoms)
    rhf = scf.RHF(mol).run()
    qc_results = analysis.run_quantum_chemistry_calculations(mol, rhf, bond_length)

    # Calculate convergence data (baseline: without singlet enforcing)
    conv_results = analysis.calc_convergence_data(qc_results, spin_symm=True)

    # Save data to CSV
    analysis.save_convergence_data(data_dir, qc_results, conv_results)

    # Create plots for baseline
    plot.energy_vs_samples(data_dir, qc_results, conv_results)
    plot.convergence_comparison(data_dir, qc_results, conv_results, ylog=True)

    # Compare spin-symmetric QSCI with and without singlet enforcing
    # We'll sweep the same subspace sizes as in conv_results and compute a second
    # energy series where we enforce the singlet condition when selecting the
    # eigenvector from the subspace diagonalisation.
    subspace_sizes = conv_results.df['subspace_size'].tolist()
    spin_symm_no_enforce = conv_results.df['spin_symm_energy'].tolist()
    spin_symm_enforced = []

    for size in subspace_sizes:
        e_enf = analysis.calc_qsci_energy_with_size(
            qc_results.H,
            qc_results.spin_symm_amp,  # already spin-symmetric amplitudes
            size,
            return_vector=False,
            spin_symmetry=False,       # data is already spin symmetrised
            enforce_singlet=True,
            singlet_tol=1e-6,
        )
        spin_symm_enforced.append(e_enf)

    # Save comparison CSV
    df_compare = pd.DataFrame({
        'subspace_size': subspace_sizes,
        'spin_symm_energy_no_enforce': spin_symm_no_enforce,
        'spin_symm_energy_enforced': spin_symm_enforced,
    })
    (data_dir / 'spin_symm').mkdir(parents=True, exist_ok=True)
    df_compare.to_csv(data_dir / 'spin_symm' / 'spin_symm_enforce_compare.csv', index=False)

    # Plot comparison
    sns.set_context('talk')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(subspace_sizes, spin_symm_no_enforce, label='Spin-symm QSCI (no enforce)', color='tab:orange')
    ax.plot(subspace_sizes, spin_symm_enforced, label='Spin-symm QSCI (enforce singlet)', color='tab:blue')
    ax.axhline(qc_results.fci_energy, color='green', linestyle='--', linewidth=1.2, label='FCI')
    ax.axhline(qc_results.uhf.e_tot, color='red', linestyle=':', linewidth=1.0, label='UHF')

    ax.set_xlabel('Subspace size (number of configurations)')
    ax.set_ylabel('Energy (Ha)')
    ax.set_title('Spin-symmetric QSCI: with vs without singlet enforcing')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    data_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(data_dir / 'spin_symm_enforce_compare.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Compute final QSCI wavefunction and plot coefficients
    print(f"\nComputing QSCI ground state wavefunction with {conv_results.max_size} configurations...")
    qsci_energy_final, qsci_vec, qsci_indices = analysis.calc_qsci_energy_with_size(
        qc_results.H,
        qc_results.sv,
        conv_results.max_size, return_vector=True,
    )

    plot.statevector_coefficients(
        qc_results.sv.data,
        qc_results.fci_vec,
        data_dir,
        n_top=20,
    )
    plot.total_spin_vs_subspace(
        data_dir=data_dir,
        qc_results=qc_results,
        conv_results=conv_results,
        title_prefix="H6 Chain",
    )

    # Print summary and note enforced vs non-enforced minima for spin-symmetric series
    print_summary(data_dir, qc_results, conv_results, qsci_energy_final)
    min_no_enf = float(np.min(spin_symm_no_enforce))
    min_enf = float(np.min(spin_symm_enforced))
    print("\nSpin-symmetric QSCI comparison:")
    print(f"  Min without enforcing: {min_no_enf:.8f} Ha")
    print(f"  Min with enforcing   : {min_enf:.8f} Ha")
    print(f"  Difference (enf - no): {min_enf - min_no_enf:+.2e} Ha")


def print_top_configs(bond_length=2, n_atoms=6):
    """
    Printing the top amplitudes from the rotated UHF statevector.
    This is to quickly see if the spin recovery is required/working.
    """
    # Run quantum chemistry calculations
    mol = build_h_chain(bond_length, n_atoms)
    rhf = scf.RHF(mol).run()

    uhf = uhf_from_rhf(mol, rhf)
    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
    sv = circuit.simulate(qc)
    print("sv norm:", np.linalg.norm(sv.data))

    sorted_idx = np.argsort(np.abs(sv.data))[::-1]
    max_idx = sorted_idx[:20]
    n_bits = int(np.log2(sv.data.size))

    # For the highest amplitude configurations print their amplitude, index,
    # bitstring, and occupation vector in a table format.
    print("Norm" + ' '*6 + "Int" + ' '*5 + "Bitstring" + ' '*(n_bits-4) + "Occ")
    for i in max_idx:
        bitstring = bin(i)[2:].zfill(n_bits)
        occ_vec = occupation_vector(bitstring, n_bits)
        sv_amp = np.abs(sv.data[i])
        bitstring_ab = bitstring[:n_bits//2] + ' ' + bitstring[n_bits//2:]
        print(f"{sv_amp:.4f}    {i:4d}    {bitstring_ab}    {occ_vec}")
    print('\n')

    symm_amp = analysis.spin_symm_amplitudes(sv.data)
    print("symm amp norm:", np.linalg.norm(symm_amp))

    sorted_idx = np.argsort(np.abs(symm_amp))[::-1]
    max_idx = sorted_idx[:20]
    n_bits = int(np.log2(symm_amp.size))

    # For the highest amplitude configurations after spin recovery, print their
    # amplitude, index, bitstring, and occupation vector in a table format.
    print("Norm" + ' '*6 + "Int" + ' '*5 + "Bitstring" + ' '*(n_bits-4) + "Occ")
    for i in max_idx:
        bitstring = bin(i)[2:].zfill(n_bits)
        occ_vec = occupation_vector(bitstring, n_bits)
        sv_amp = np.abs(symm_amp[i])
        bitstring_ab = bitstring[:n_bits//2] + ' ' + bitstring[n_bits//2:]
        print(f"{sv_amp:.4f}    {i:4d}    {bitstring_ab}    {occ_vec}")


def occupation_vector(bitstring, n_bits) -> str:
    """
    Given a bitstring, return the occupation vector.
    Closed shells are represented by 0 or 2 for unoccupied or occupied.
    Open shells are represented by α or β for up or down spin occupancy.

    The RHF molecular orbitals are ordered from highest to lowest energy.
    This is inline with the Qiskit qubit ordering convention.

    Parameters
    ----------
    bitstring: str
        Binary representation of the bitstring.

    Returns
    -------
    occ_vec: str
        Occupation vector representation of the bitstring.
    """
    # Ensure string format
    bitstring = str(bitstring)
    alpha, beta = bitstring[:n_bits//2], bitstring[n_bits//2:]
    occ_vec = ""
    for i in range(n_bits//2):
        if alpha[i] == "1" and beta[i] == "1":
            occ_vec += '2'
        elif alpha[i] == "1":
            occ_vec += '\u03b1'
        elif beta[i] == "1":
            occ_vec += '\u03b2'
        else:
            occ_vec += '0'
    return occ_vec


def build_h_chain(bond_length, n_atoms=6) -> gto.Mole:
    """
    Build a chain of hydrogen atoms.
    """
    coords = [(i * bond_length, 0, 0) for i in range(n_atoms)]
    geometry = '; '.join([f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in coords])
    mol = gto.Mole()
    mol.build(
        atom=geometry,
        unit='Angstrom',
        basis='sto-3g',
        charge=0,
        spin=0,
        verbose=0,
    )
    return mol


def print_summary(
        data_dir: Path,
        qc_results: analysis.QuantumChemistryResults,
        conv_results: analysis.ConvergenceResults,
        qsci_energy_final: float,
) -> None:
    """
    Print summary of results to console.
    """
    print(f"\nReference Energies:")
    print(f"  RHF: {qc_results.rhf.e_tot:.8f} Ha")
    print(f"  UHF: {qc_results.uhf.e_tot:.8f} Ha")
    print(f"  FCI: {qc_results.fci_energy:.8f} Ha")
    print(f"  QSCI (max subspace): {qsci_energy_final:.8f} Ha")
    print(f"\nFCI Solution:")
    print(f"  Number of configurations: {qc_results.n_fci_configs}")
    print(f"\nQSCI Convergence:")
    print(f"  Max subspace size: {conv_results.max_size}")
    print(f"  Min QSCI energy: {conv_results.df['qsci_energy'].min():.8f} Ha")
    print(f"  Energy difference to FCI: {conv_results.df['qsci_energy'].min() - qc_results.fci_energy:.2e} Ha")
    print(f"\nMilestones:")
    print(f"  Configs to fall below UHF: {conv_results.n_configs_below_uhf if conv_results.n_configs_below_uhf else 'Never achieved'}")
    print(f"  Configs to reach FCI: {conv_results.n_configs_reach_fci if conv_results.n_configs_reach_fci else 'Never achieved'}")
    print(f"\nData saved to '{data_dir}' directory:")
    print("  - h6_qsci_convergence.csv (full energy data)")
    print("  - h6_summary.csv (summary statistics)")
    print("  - h6_qsci_convergence.png (plot)")
    print("  - h6_energy_vs_samples.png (energy vs mean sample number)")
    print("  - statevector_coefficients.png (top 20 coefficients bar chart)")
    print("  - statevector_coefficients_full.png (all significant coefficients)")


if __name__ == "__main__":
    main()
