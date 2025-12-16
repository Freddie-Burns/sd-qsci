01_h4_chain — H4 chain energies

Purpose
- Compare energies from RHF, UHF, FCI, and QSCI for a linear H4 chain across bond lengths.
- Validate that the RHF→UHF orbital-rotation circuit preserves the UHF energy.

<figure>
  <img src="data/h4_chain_energies.png" alt="Energy comparison across bond lengths" width="65%" />
  <figcaption>
    Energy vs bond length for the H4 chain: RHF, UHF, FCI, and QSCI curves; QSCI closely follows FCI near equilibrium while UHF lowers energy as symmetry breaks.
  </figcaption>
</figure>

Main script
- 01_h4_chain.py — runs scans over bond length and produces a comparison plot.

Outputs
- Figure: figures/h4_chain_energies.png
- Console logs of energies by method for each bond length.

Usage
- From repository root:
  - python research/01_h4_chain/01_h4_chain.py
