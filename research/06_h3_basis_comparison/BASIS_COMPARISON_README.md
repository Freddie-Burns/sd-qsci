# H3+ Basis Set Comparison Scripts

This directory contains scripts for comparing QSCI convergence across multiple basis sets for the H3+ triangular ion.

## Scripts

### 1. `06_h3_basis_comparison.py` (Full Analysis)

Runs FCI and QSCI calculations for multiple basis sets:
- **STO-3G** (minimal)
- **STO-6G** (minimal)
- **3-21G** (split-valence)
- **6-31G** (split-valence)
- **6-31G*** (polarization on heavy atoms)
- **6-31G**** (polarization on all atoms)
- **6-311++G**** (diffuse + polarization)
- **cc-pVDZ** (correlation-consistent double-zeta)
- **cc-pVTZ** (correlation-consistent triple-zeta)
- **aug-cc-pVTZ** (augmented triple-zeta)
- **aug-cc-pVQZ** (augmented quadruple-zeta)

**Features:**
- Calculates RHF, UHF, and FCI energies for each basis set
- Computes QSCI convergence with increasing subspace size
- Saves individual and combined CSV files with all data
- Creates comparison plots showing convergence for all basis sets
- Generates summary statistics table

**Usage:**
```bash
python research/06_h3_basis_comparison.py
```

**⚠️ Computational Cost Warning:**
- Larger basis sets (cc-pVTZ, aug-cc-pVTZ, aug-cc-pVQZ) have **significantly** more orbitals
- Hilbert space scales as 2^(2*n_orbitals), making FCI calculations expensive
- aug-cc-pVQZ may take considerable time and memory
- Consider running the test script first, or commenting out larger basis sets if needed

**Output:**
- `data/06_h3_basis_comparison/all_basis_summary.csv` - Summary table of all basis sets
- `data/06_h3_basis_comparison/{basis}_convergence.csv` - Individual basis set data
- `data/06_h3_basis_comparison/{basis}_summary.csv` - Individual basis set summary
- `data/06_h3_basis_comparison/basis_comparison_convergence.png` - Linear scale plot
- `data/06_h3_basis_comparison/basis_comparison_convergence_log.png` - Log scale plot
- `data/06_h3_basis_comparison/basis_energy_levels.png` - Energy level comparison

Note: Special characters in basis names (*, +, -) are replaced in filenames (star, plus, _)

### 2. `06_h3_basis_comparison_test.py` (Quick Test)

A lighter version that tests only 4 basis sets (STO-3G, 3-21G, 6-31G*, cc-pVDZ) for quick validation.

**Usage:**
```bash
python research/06_h3_basis_comparison_test.py
```

**Output:**
- `data/06_h3_basis_comparison_test/all_basis_summary.csv`
- `data/06_h3_basis_comparison_test/all_basis_convergence.csv`
- `data/06_h3_basis_comparison_test/basis_comparison.png`

## Configuration

Both scripts can be easily customized by editing the constants at the top:

```python
BASIS_SETS = [
    "sto-3g", "sto-6g", "3-21g", "6-31g", "6-31g*",
    "6-31g**", "6-311++g**", "cc-pvdz", "cc-pvtz",
    "aug-cc-pvtz", "aug-cc-pvqz"
]
BOND_LENGTH = 2.0
SV_TOL = 1e-12  # Statevector tolerance
FCI_TOL = 1e-6  # FCI convergence tolerance
```

**Tip:** If some basis sets are too large, you can comment them out:
```python
BASIS_SETS = [
    "sto-3g", "3-21g", "6-31g*", "cc-pvdz",
    # "aug-cc-pvqz",  # Too large, skip this one
]
```

## Expected Computational Requirements

Approximate number of orbitals and Hilbert space size for H3:

| Basis Set      | Orbitals | Hilbert Size | Est. Time |
|----------------|----------|--------------|-----------|
| STO-3G         | 3        | 64          | < 1 sec   |
| STO-6G         | 3        | 64          | < 1 sec   |
| 3-21G          | 6        | 4,096       | < 1 sec   |
| 6-31G          | 6        | 4,096       | < 1 sec   |
| 6-31G*         | 6        | 4,096       | < 1 sec   |
| 6-31G**        | 9        | 262,144     | ~5 sec    |
| 6-311++G**     | 12       | 16,777,216  | ~30 sec   |
| cc-pVDZ        | 15       | ~1e9        | ~1-2 min  |
| cc-pVTZ        | 30       | ~1e18       | **HUGE**  |
| aug-cc-pVTZ    | 42       | ~1e25       | **HUGE**  |
| aug-cc-pVQZ    | 69       | ~1e41       | **HUGE**  |

**⚠️ Warning:** cc-pVTZ and larger basis sets may be **intractable** for FCI on H3+. The Hilbert space becomes astronomically large. You may need to:
1. Use FCI solvers that work in the reduced space only
2. Skip FCI and focus on QSCI convergence
3. Reduce to smaller molecules
4. Use only smaller basis sets

## Output Explanation

### CSV Files

**all_basis_summary.csv** contains:
- `basis`: Basis set name
- `n_orbitals`: Number of spatial orbitals
- `rhf_energy`: RHF energy (Hartree)
- `uhf_energy`: UHF energy (Hartree)
- `fci_energy`: Full CI energy (Hartree)
- `n_fci_configs`: Number of FCI configurations
- `max_subspace_size`: Maximum QSCI subspace size
- `min_qsci_energy`: Lowest QSCI energy achieved
- `energy_diff_to_fci`: Difference from FCI energy
- `n_configs_below_uhf`: Configs needed to go below UHF
- `n_configs_reach_fci`: Configs needed to reach FCI (within tolerance)

**{basis}_convergence.csv** contains:
- `subspace_size`: Number of configurations in subspace
- `qsci_energy`: QSCI energy for this subspace
- `fci_subspace_energy`: FCI subspace energy (for comparison)
- `mean_sample_number`: Mean number of samples needed

### Plots

**basis_comparison_convergence.png**: Shows energy difference from FCI vs subspace size for all basis sets on a linear scale. This helps visualize how quickly each basis set converges to its FCI limit.

**basis_comparison_convergence_log.png**: Same as above but with log scale for the energy difference, making it easier to see convergence when differences become very small.

**basis_energy_levels.png**: Bar chart comparing RHF, UHF, and FCI energies across all basis sets.

## Expected Behavior

- **Larger basis sets** (e.g., cc-pVTZ) typically give lower (more negative) absolute energies
- **More complete basis sets** (augmented, correlation-consistent) provide better descriptions of electron correlation
- **QSCI convergence speed** may vary by basis set - larger basis sets might need more configurations
- The number of configurations needed to reach FCI accuracy depends on:
  - Basis set size (more orbitals = more configurations)
  - Degree of electron correlation
  - System geometry
  - Basis set type (minimal vs correlation-consistent)

## Notes

- The H3+ ion is chosen because it's small enough to run quickly with smaller basis sets
- Bond length is set to 2.0 Å for an equilateral triangle
- All calculations use charge=1 and spin=0 (closed-shell singlet)
- The scripts use the `sd_qsci.analysis` module for all quantum chemistry calculations
- Scripts handle basis set failures gracefully - if one fails, others continue

## Basis Set Details

### Pople-style (6-31G family)
- Economical for routine calculations
- 6-31G**: adds polarization to H atoms
- 6-311++G**: adds diffuse functions (important for anions, excited states)

### Correlation-consistent (cc-pV*Z)
- Designed for systematic improvement
- More complete description of electron correlation
- pVDZ < pVTZ < pVQZ (double < triple < quadruple)
- aug- prefix adds diffuse functions

### Minimal (STO)
- STO-3G, STO-6G: fast but qualitative only
- Good for testing and educational purposes

## Troubleshooting

**If a basis set fails:**
- Check that PySCF supports the basis set for hydrogen
- May need to increase Python memory limit
- Some basis sets produce very large Hilbert spaces that are intractable

**For very large basis sets:**
- Consider using DMRG or other methods instead of FCI
- Focus on smaller basis sets for full analysis
- Use reduced molecule (H2 instead of H3+)

**Memory issues:**
- Monitor RAM usage during FCI calculations
- Close other applications
- Consider running on HPC cluster for large basis sets

For the basis sets with special characters (*, +), the filenames automatically replace them:
- `6-31g*` → `6_31gstar`
- `6-311++g**` → `6_311plusplusgstarstar`

