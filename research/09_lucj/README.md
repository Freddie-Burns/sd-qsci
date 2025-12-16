LUCJ circuits and comparisons (09)

Purpose
- Explore Low-Depth Unitary Coupled-Cluster with Jastrow (LUCJ) circuits and compare with alternatives.

<figure>
  <img src="data/09b_lucj_comparison/bond_length_2.00/h6_qsci_convergence_energy_linear.png" alt="LUCJ comparison — QSCI energy convergence" width="65%" />
  <figcaption>
    Energy convergence at 2.00 Å comparing LUCJ variants/parameters; highlights how low-depth ansätze approach reference energies under QSCI evaluation.
  </figcaption>
</figure>

Scripts
- 09a_lucj_circuit.py — construct and analyze a LUCJ circuit across bond lengths
- 09b_lucj_comparison.py — compare LUCJ variants/parameters

Outputs
- Saved under data/09a and data/09b respectively, with per-run subfolders (e.g., bond_length_2.00)

Usage
- From repository root:
  - python research/09_lucj/09a_lucj_circuit.py
  - python research/09_lucj/09b_lucj_comparison.py