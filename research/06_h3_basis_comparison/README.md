H3+ basis set comparison (06)

Purpose
- Systematically compare QSCI and reference energies across many basis sets for H3+.
- Includes a quick test script and a full analysis.

![H3+ basis sets — energy levels overview](data/06_h3_basis_comparison/basis_energy_levels.png)

Scripts
- 06_h3_basis_comparison.py — full analysis across many basis sets
- 06_h3_basis_comparison_test.py — lighter, quick validation

Outputs
- Saved under data/06/ ... for the full script and data/06_test/ ... for the test script.
  Typical files: *_convergence.csv, *_summary.csv, basis_comparison_*.png

Usage
- From repository root:
  - python research/06_h3_basis_comparison/06_h3_basis_comparison.py
  - python research/06_h3_basis_comparison/06_h3_basis_comparison_test.py

Notes
- See also BASIS_COMPARISON_README.md in this folder for a more detailed description.