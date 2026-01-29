"""
Create orbital rotational circuits for multiple UHF solutions.
"""

from pathlib import Path

import numpy as np
from pyscf import gto, scf

import multi_uhf_utils as utils
from sd_qsci import analysis, circuit, plot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches, transforms
from sd_qsci.analysis import spin_closed_subspace_sizes


def compare_convergence_custom(data_dir, results_dict, title_prefix=None):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    y_lo = 1.0e-4
    chem_acc = 1.6e-3
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.add_patch(patches.Rectangle((0.0, y_lo), 1.0, chem_acc - y_lo, transform=trans, facecolor='#D0D0D0', alpha=0.35, zorder=0))
    ax.add_patch(patches.Rectangle((0.0, y_lo), 1.0, chem_acc - y_lo, transform=trans, facecolor='none', edgecolor='#2ca02c', hatch='///', linewidth=0.0, zorder=0))
    ax.set_ylim(bottom=y_lo)
    
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for i, (label, (qc_res, conv_res)) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        fci_E = qc_res.fci_energy
        ax.plot(conv_res.df['subspace_size'], conv_res.df['qsci_energy'] - fci_E, 'o-', label=f"{label} QSCI", color=color)
        spin_closed_sizes = set(spin_closed_subspace_sizes(qc_res.sv.data))
        df_symm = conv_res.df[conv_res.df['subspace_size'].isin(spin_closed_sizes)]
        ax.plot(df_symm['subspace_size'], df_symm['spin_symm_energy'] - fci_E, '^-', label=f"{label} Spin-Rec", color=color, alpha=0.7)

    # Plot FCI and RHF from the last results (should be the same for all)
    ax.plot(conv_res.df['subspace_size'], conv_res.df['fci_subspace_energy'] - fci_E, 's-', label='FCI Subspace', color='black', alpha=0.5)
    ax.axhline(y=qc_res.rhf.e_tot - fci_E, color='grey', linestyle='--', label='RHF')

    ax.set_yscale('log')
    ax.set_xlabel('Subspace Size')
    ax.set_ylabel('Energy Error (Ha)')
    title = "Convergence Comparison"
    if title_prefix: title = f"{title_prefix}: {title}"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(data_dir / 'compare_convergence.png', dpi=300)
    plt.close()

def compare_energy_vs_samples_custom(data_dir, results_dict, title_prefix=None):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    y_lo = 1.0e-4
    chem_acc = 1.6e-3
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.add_patch(patches.Rectangle((0.0, y_lo), 1.0, chem_acc - y_lo, transform=trans, facecolor='#D0D0D0', alpha=0.35, zorder=0))
    ax.add_patch(patches.Rectangle((0.0, y_lo), 1.0, chem_acc - y_lo, transform=trans, facecolor='none', edgecolor='#2ca02c', hatch='///', linewidth=0.0, zorder=0))
    ax.set_ylim(bottom=y_lo)

    def mean_samples_for_sizes(data, sizes_list):
        order = np.argsort(np.abs(data))
        vals = []
        for size in sizes_list:
            idx = order[-int(size):]
            min_coeff = float(np.min(np.abs(data[idx]))) if len(idx) > 0 else 0.0
            vals.append(float(1.0 / (min_coeff ** 2)) if min_coeff > 0 else np.inf)
        return vals

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for i, (label, (qc_res, conv_res)) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        fci_E = qc_res.fci_energy
        sizes = list(conv_res.df['subspace_size'])
        ms_raw = mean_samples_for_sizes(qc_res.sv.data, sizes)
        ax.semilogx(ms_raw, conv_res.df['qsci_energy'] - fci_E, 'o-', label=f"{label} QSCI", color=color)
        
        spin_sizes_all = sorted(set(spin_closed_subspace_sizes(qc_res.sv.data)))
        max_size = sizes[-1] if sizes else 0
        spin_sizes = [int(s) for s in spin_sizes_all if int(s) <= max_size]
        if spin_sizes:
            ms_symm = mean_samples_for_sizes(qc_res.spin_symm_amp, spin_sizes)
            df_symm = conv_res.df[conv_res.df['subspace_size'].isin(spin_sizes)]
            ax.semilogx(ms_symm, df_symm['spin_symm_energy'] - fci_E, '^-', label=f"{label} Spin-Rec", color=color, alpha=0.7)

    ax.set_xlabel('Mean Samples')
    ax.set_ylabel('Energy Error (Ha)')
    title = "Energy vs Samples Comparison"
    if title_prefix: title = f"{title_prefix}: {title}"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(data_dir / 'compare_energy_vs_samples.png', dpi=300)
    plt.close()


# Setup
bond_length = 2.0
n_atoms = 4
print(f"Running H{n_atoms} chain bond length: {bond_length:.2f} Angstrom")

# Save under this script's full stem name inside this script's folder
stem = Path(__file__).stem
data_dir = Path(__file__).parent / 'data' / stem / f"bond_length_{bond_length:.2f}_spin_symm"

# Run quantum chemistry calculations
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

patterns = {
    "up down up down": [1, -1, 1, -1], # antiferromagnetic: lowest energy soln
    "up up down down": [1, 1, -1, -1], # ferromagnetic: higher energy soln
}

rhf = scf.RHF(mol).run()
all_results = {}

for label, pattern in patterns.items():
    print(f"\n--- Running for pattern: {label} ---")
    uhf = utils.solve_with_spin_pattern(mol, pattern, label)
    qc_results = analysis.run_quantum_chemistry_calculations(
        mol, rhf, bond_length, uhf=uhf)
    
    # Calculate convergence data
    conv_results = analysis.calc_convergence_data(qc_results, spin_symm=True)
    
    # Store results
    all_results[label] = (qc_results, conv_results)
    
    # Save individual data/plots
    pattern_dir = data_dir / label.replace(" ", "_")
    analysis.save_convergence_data(pattern_dir, qc_results, conv_results)
    plot.energy_vs_samples(pattern_dir, qc_results, conv_results)
    plot.convergence_comparison(pattern_dir, qc_results, conv_results, ylog=True)
    
    # Compute final QSCI wavefunction and plot coefficients
    print(f"Computing QSCI ground state wavefunction with {conv_results.max_size} configurations...")
    qsci_energy_final, qsci_vec, qsci_indices = analysis.calc_qsci_energy_with_size(
        qc_results.H,
        qc_results.sv,
        conv_results.max_size, return_vector=True,
    )
    
    plot.statevector_coefficients(
        qc_results.sv.data,
        qc_results.fci_vec,
        pattern_dir,
        n_top=20,
    )
    plot.total_spin_vs_subspace(
        data_dir=pattern_dir,
        qc_results=qc_results,
        conv_results=conv_results,
        title_prefix=f"H{n_atoms} Chain ({label})"
    )

# Create comparison plots
print(f"\nCreating comparison plots in {data_dir}...")
compare_convergence_custom(data_dir, all_results, title_prefix=f"H{n_atoms} Chain Comparison")
compare_energy_vs_samples_custom(data_dir, all_results, title_prefix=f"H{n_atoms} Chain Comparison")
