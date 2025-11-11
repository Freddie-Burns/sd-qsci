sd-qsci documentation
=====================

**Single Determinant Quantum Simulation Chemistry Investigation**

``sd-qsci`` is a Python package for investigating the effectiveness of single Slater
determinant trial wavefunctions in quantum simulation chemistry (QSCI) methods.

This package provides tools to:

* Construct fermionic Hamiltonians from PySCF mean-field calculations
* Generate quantum circuits for orbital rotations (RHF → UHF transitions)
* Analyze spin properties of quantum states
* Interface between classical quantum chemistry (PySCF) and quantum simulation (Qiskit)

Installation
------------

Install using `uv <https://github.com/astral-sh/uv>`_:

.. code-block:: bash

   git clone https://github.com/Freddie-Burns/sd-qsci.git
   cd sd-qsci
   uv sync

Quick Start
-----------

.. code-block:: python

   from pyscf import gto, scf
   from sd_qsci.hamiltonian import hamiltonian_from_pyscf
   from sd_qsci.qc import rhf_uhf_orbital_rotation_circuit, run_statevector

   # Define molecule
   mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
   rhf = scf.RHF(mol).run()

   # Generate quantum circuit for orbital rotation
   circuit, uhf, (Ua, Ub) = rhf_uhf_orbital_rotation_circuit(mol, rhf)

   # Run simulation
   statevector = run_statevector(circuit)

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

