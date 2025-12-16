Spin symmetry recovery (08)

Purpose
- Investigate spin-symmetry recovery techniques and their impact on QSCI results.

<figure>
  <img src="data/08_spin_recovery/bond_length_2.00_spin_symm/h6_qsci_convergence.png" alt="Spin symmetry recovery — QSCI convergence" width="65%" />
  <figcaption>
    Impact of spin‑symmetry recovery at 2.00 Å: restores spin purity and improves QSCI convergence relative to the unconstrained reference.
  </figcaption>
</figure>

Scripts
- 08_spin_recovery.py — end-to-end analysis for a small system (configurable bond length and atom count)

Outputs
- Saved under data/08/ ... with per-run subfolders (e.g., bond_length_2.00_spin_symm)

Usage
- From repository root:
  - python research/08_spin_recovery/08_spin_recovery.py