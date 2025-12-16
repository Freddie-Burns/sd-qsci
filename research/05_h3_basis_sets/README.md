H3+ minimal basis set studies (05)

Purpose
- Compare QSCI behavior for H3+ across small Pople/STO bases:
  - 05a_h3_sto3g.py — STO-3G
  - 05b_h3_sto6g.py — STO-6G
  - 05c_h3_basis_sets.py — simple comparison sweep

![H3+ STO-3G — QSCI convergence](data/05a_h3_sto3g/bond_length_2.00/h6_qsci_convergence.png)

Outputs
- Saved under data/05a, data/05b, data/05c respectively, with per-run subfolders (e.g., bond_length_2.00)

Usage
- From repository root:
  - python research/05_h3_basis_sets/05a_h3_sto3g.py
  - python research/05_h3_basis_sets/05b_h3_sto6g.py
  - python research/05_h3_basis_sets/05c_h3_basis_sets.py