H6 lattice and chain (03)

Purpose
- Explore QSCI behavior on H6 in lattice and chain geometries:
  - 03a_h6_lattice_n_dets.py — counts and scans number of determinants/configurations
  - 03b_h6_lattice_weights.py — investigates sampling/weighting strategies
  - 03c_h6_chain.py — chain geometry study vs bond length

![H6 lattice weights – QSCI convergence](data/03b_h6_lattice_weights/h6_qsci_convergence.png)
![H6 chain – energy vs samples](data/03c_h6_chain/h6_energy_vs_samples.png)

Outputs
- Saved under data/03a, data/03b, data/03c respectively, with per-run subfolders (e.g., bond_length_2.00)

Usage
- From repository root:
  - python research/03_h6_lattice_dev/03a_h6_lattice_n_dets.py
  - python research/03_h6_lattice_dev/03b_h6_lattice_weights.py
  - python research/03_h6_lattice_dev/03c_h6_chain.py