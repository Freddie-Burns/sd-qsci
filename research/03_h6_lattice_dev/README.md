H6 lattice and chain (03)

Purpose
- Explore QSCI behavior on H6 in lattice and chain geometries:
  - 03a_h6_lattice_n_dets.py — counts and scans number of determinants/configurations
  - 03b_h6_lattice_weights.py — investigates sampling/weighting strategies
  - 03c_h6_chain.py — chain geometry study vs bond length

<figure>
  <img src="data/03b_h6_lattice_weights/h6_qsci_convergence.png" alt="H6 lattice weights – QSCI convergence" width="65%" />
  <figcaption>
    Effect of sampling/weighting strategies on QSCI convergence for the H6 lattice; demonstrates how reweighting accelerates convergence.
  </figcaption>
</figure>
<figure>
  <img src="data/03c_h6_chain/h6_energy_vs_samples.png" alt="H6 chain – energy vs samples" width="65%" />
  <figcaption>
    H6 chain: energy error decreases with the number of sampled configurations, illustrating sample-efficiency trade-offs.
  </figcaption>
</figure>

Outputs
- Saved under data/03a, data/03b, data/03c respectively, with per-run subfolders (e.g., bond_length_2.00)

Usage
- From repository root:
  - python research/03_h6_lattice_dev/03a_h6_lattice_n_dets.py
  - python research/03_h6_lattice_dev/03b_h6_lattice_weights.py
  - python research/03_h6_lattice_dev/03c_h6_chain.py