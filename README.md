[![docs](https://img.shields.io/badge/docs-online-blue)](https://freddie-burns.github.io/sd-qsci/)

# Single Determinant QSCI

**Investigating the effectiveness of single Slater determinant trial wavefunctions for quantum simulation and QSCI methods.**

This project explores how well simple mean-field states—such as unrestricted or restricted Hartree–Fock (UHF/RHF) single determinants—serve as **trial wavefunctions** in **quantum simulation chemistry (QSCI)** workflows.
The goal is to benchmark their performance and limitations when used in hybrid classical–quantum methods and related post-HF quantum algorithms.

---

## Motivation

Many quantum chemistry and quantum simulation algorithms rely on a trial wavefunction to initialize or constrain a variational search.
While multi-determinant or correlated references can improve accuracy, **single Slater determinants** (from UHF or RHF) remain the simplest and most computationally efficient choice.

This repository aims to:

* Quantify the accuracy of single-determinant trials across small molecules.
* Compare UHF vs RHF determinants as initial states.
* Interface **PySCF** (for reference and integral generation) with **Qiskit** (for quantum simulation).

---

## Project structure

```
sd-qsci/
├─ pyproject.toml         # Project metadata and dependencies (managed by uv)
├─ src/
│  └─ sd_qsci/            # Python package (imports as sd_qsci)
│     ├─ __init__.py
│     ├─ qc.py            # Quantum circuit creation and execution
│     ├─ spin.py          # Spin analysis utilities
│     ├─ utils.py         # General utilities
│     ├─ hamiltonian/     # Hamiltonian construction from PySCF
│     └─ __pycache__/
├─ research/              # Research experiments and workflows
│  ├── __init__.py
│  ├── experiments/       # Standalone research scripts
│  └── config/            # Shared configurations
├─ tests/                 # Unit and integration tests
├─ notebooks/
│  └─ dev/                # Development notebooks
├─ docs/                  # Sphinx documentation
├─ data/                  # Molecular geometries and results
└─ README.md
```

---
git clone https://github.com/Freddie-Burns/sd-qsci.git
cd sd-qsci

This project uses [**uv**](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone repository
git clone https://github.com/Freddie-Burns/sd_qsci.git
cd single_determinant_qsci

# Create virtual environment
uv venv
* [PySCF](https://pyscf.org) — ab initio electronic structure calculations
* [Qiskit](https://qiskit.org) — quantum circuit construction and simulation
* [ffsim](https://github.com/qiskit-community/ffsim) — fermion simulation
* [NumPy](https://numpy.org) — numerical computing
* [SciPy](https://scipy.org) — scientific computing
```
Main dependencies:

* [PySCF](https://pyscf.org) — reference mean-field and integrals
* [Qiskit](https://qiskit.org) — quantum simulation
* [NumPy](https://numpy.org)
* [pytest](https://docs.pytest.org) — testing framework
* [JupyterLab](https://jupyter.org) — visualising molecular orbitals

uv run python -m sd_qsci

## Quick start

Run the demo script:

```bash
uv run python -m single_determinant_qsci.main
```
Example notebooks:
or launch notebooks:

notebooks/dev/00_hamiltonian_tutorial.ipynb
uv run jupyter lab
```
Demonstrates Hamiltonian construction from PySCF molecular systems.
Example notebook:

notebooks/dev/01_verify_unitary_qc.ipynb
notebooks/01_h2_uhf_trial.ipynb

Verifies unitarity of quantum circuits and orbital rotation operations.
```

Demonstrates generating an unrestricted Hartree–Fock determinant for H₂ and evaluating its effectiveness in a small QSCI circuit.

```
notebooks/hamiltonian_tutorial.ipynb
```

---

## Testing

Run tests via:

```bash
uv run pytest
```

---

## Possible extensions

* Benchmark larger systems (LiH, BeH₂, H₂O).
* Compare RHF vs UHF vs multi-determinant expansions.
* Compare with Qiskit Nature Hamiltonian generation.
* Add visualization of orbital correlation and overlap metrics.

---

## References

* *Qiskit Nature Documentation*
* *PySCF: The Python-based Simulations of Chemistry Framework*
* Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory* (2000)

---

## License

MIT License © [Freddie Burns] 2025

