"""
Research script to build and analyze a LUCJ circuit for an H6 hydrogen chain.

What it does:
- Builds a linear H6 molecule at a chosen bond length and runs RHF → UHF (via
  spin‑unrestriction helper) and CCSD with PySCF.
- Constructs a LUCJ variational circuit from the CCSD data and simulates it on
  the Qiskit Aer statevector simulator.
- Derives the molecular Hamiltonian and computes reference FCI energy and
  amplitudes.
- Evaluates QSCI convergence data and saves summary CSVs and plots, including:
  - h6_qsci_convergence.png / .csv
  - h6_energy_vs_samples.png
  - statevector_coefficients.png and statevector_coefficients_full.png
  - lucj_circuit.png (diagram of the LUCJ circuit)

Outputs are written under research/data/09a_lucj_circuit/bond_length_XX.XX.
Run this file directly to reproduce the analysis for the configured bond length.
"""

from pathlib import Path

from pyscf import gto, scf
from pyscf.cc import CCSD
from qiskit_aer import Aer
from qiskit.visualization import circuit_drawer

from sd_qsci import analysis, circuit, hamiltonian
from sd_qsci.utils import uhf_from_rhf


# Script-specific tolerances
# SV_TOL = 1e-2
CHEM_ACC = 1.6e-3


def main():
    """
    Run H6 chain energy calculations and analyze QSCI convergence.
    """
    # Setup
    bond_length = 2.0
    n_atoms = 6

    # Save under grouped short code (e.g., '09a_lucj_circuit') inside this script's folder
    stem = Path(__file__).stem
    data_dir = Path(__file__).parent / 'data' / stem / f"bond_length_{bond_length:.2f}"

    print(f"Running H{n_atoms} chain bond length: {bond_length:.2f} Angstrom")

    # Run quantum chemistry calculations
    mol = build_h_chain(bond_length, n_atoms)
    rhf = scf.RHF(mol).run()
    uhf = uhf_from_rhf(mol, rhf)
    ccsd = CCSD(rhf).run()

    backend = Aer.get_backend("statevector_simulator")
    qc = circuit.get_lucj_circuit(ccsd_obj=ccsd, backend=backend, n_reps=1)

    # Ensure output directory exists and save an image of the LUCJ circuit
    data_dir.mkdir(parents=True, exist_ok=True)
    circuit_path = data_dir / 'lucj_circuit.png'
    try:
        circuit_drawer(qc, output='mpl', filename=str(circuit_path), fold=-1, idle_wires=False)
        print(f"Saved LUCJ circuit image to: {circuit_path}")
    except Exception as e:
        # Fall back silently if visualization is unavailable
        print(f"Warning: Failed to save LUCJ circuit image ({e})")

    sv = circuit.simulate(qc)
    spin_symm_amp = analysis.spin_symm_amplitudes(sv.data)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)

    fci_energy, n_fci_configs, fci_vec = analysis.calc_fci_energy(rhf)

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

    # Calculate convergence data
    conv_results = analysis.calc_convergence_data(qc_results, spin_symm=True)

    # Save data to CSV
    analysis.save_convergence_data(data_dir, qc_results, conv_results)

    # Create plots
    # Label plots to reflect that the statevector comes from a LUCJ circuit
    analysis.plot_energy_vs_samples(
        data_dir,
        qc_results,
        conv_results,
        label_raw='LUCJ State',
        label_spin='LUCJ State (Spin Recovered)'
    )
    analysis.plot_convergence_comparison(
        data_dir,
        qc_results,
        conv_results,
        ylog=True,
        label_raw='LUCJ State',
        label_spin='LUCJ State (Spin Recovered)'
    )

    # Compute final QSCI wavefunction and plot coefficients
    print(f"\nComputing QSCI ground state wavefunction with {conv_results.max_size} configurations...")
    qsci_energy_final, qsci_vec, qsci_indices = analysis.calc_qsci_energy_with_size(
        qc_results.H,
        qc_results.sv,
        conv_results.max_size, return_vector=True,
    )

    analysis.plot_statevector_coefficients(
        qc_results.sv.data,
        qc_results.fci_vec,
        data_dir,
        n_top=20,
    )
    analysis.plot_total_spin_vs_subspace(
        data_dir=data_dir,
        qc_results=qc_results,
        conv_results=conv_results,
        title_prefix="H6 Chain"
    )

    print_summary(data_dir, qc_results, conv_results, qsci_energy_final)


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
    print(f"  Configs to reach FCI (±{CHEM_ACC:.0e} Ha): {conv_results.n_configs_reach_fci if conv_results.n_configs_reach_fci else 'Never achieved'}")
    print(f"\nData saved to '{data_dir}' directory:")
    print("  - h6_qsci_convergence.csv (full energy data)")
    print("  - h6_summary.csv (summary statistics)")
    print("  - h6_qsci_convergence.png (plot)")
    print("  - h6_energy_vs_samples.png (energy vs mean sample number)")
    print("  - statevector_coefficients.png (top 20 coefficients bar chart)")
    print("  - statevector_coefficients_full.png (all significant coefficients)")
    print("  - lucj_circuit.png (LUCJ circuit diagram)")


if __name__ == "__main__":
    main()
