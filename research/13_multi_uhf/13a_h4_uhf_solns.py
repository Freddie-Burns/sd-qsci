import numpy as np
from pyscf import gto, scf

# Define H4 in a line
mol = gto.M(
    atom='''
    H 0 0 0
    H 0 0 2
    H 0 0 4
    H 0 0 6
    ''',
    basis='sto-3g',
    spin=0,  # singlet
    charge=0
)


# Method 1: Different initial guesses using stability analysis
def find_uhf_solution(mol, init_guess='minao'):
    mf = scf.UHF(mol)
    mf.kernel(init_guess=init_guess)

    # Check stability and follow to another solution if unstable
    mo1 = mf.stability()[0]
    if mo1 is not mf.mo_coeff:
        print(f"Initial solution unstable, following to new solution...")
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = scf.UHF(mol)
        mf.kernel(dm1)

    return mf


# Method 2: Random perturbations to initial guess
def find_multiple_solutions(mol, n_trials=5):
    solutions = []
    energies = []

    for i in range(n_trials):
        mf = scf.UHF(mol)

        if i == 0:
            # First solution: standard guess
            mf.kernel()
        else:
            # Add random noise to break symmetry
            np.random.seed(i)
            dm = mf.get_init_guess()
            noise = 0.1 * np.random.random(dm.shape)
            dm = dm + noise
            mf.kernel(dm)

        # Check if this is a new solution
        is_new = True
        for prev_e in energies:
            if abs(mf.e_tot - prev_e) < 1e-6:
                is_new = False
                break

        if is_new:
            solutions.append(mf)
            energies.append(mf.e_tot)
            print(f"Solution {len(solutions)}: E = {mf.e_tot:.8f} Ha")
            print(f"  <S^2> = {mf.spin_square()[0]:.4f}")

    return solutions, energies


# Method 3: Use stability analysis iteratively
def explore_with_stability(mol):
    mf = scf.UHF(mol)
    mf.kernel()

    print(f"Initial solution: E = {mf.e_tot:.8f} Ha")
    print(f"  <S^2> = {mf.spin_square()[0]:.4f}")

    # Check for internal and external instabilities
    mo1, stable_i, stable_e = mf.stability(return_status=True)

    if not stable_i:
        print("Internal instability found!")
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf2 = scf.UHF(mol)
        mf2.kernel(dm1)
        print(f"New solution: E = {mf2.e_tot:.8f} Ha")
        print(f"  <S^2> = {mf2.spin_square()[0]:.4f}")
        return mf2

    return mf


# Run the methods
print("=" * 60)
print("Finding multiple UHF solutions for linear H4")
print("=" * 60)

solutions, energies = find_multiple_solutions(mol, n_trials=10)

print("\n" + "=" * 60)
print(f"Found {len(solutions)} distinct solutions")
for i, (sol, e) in enumerate(zip(solutions, energies)):
    print(f"\nSolution {i + 1}:")
    print(f"  Energy: {e:.8f} Ha")
    print(f"  <S^2>: {sol.spin_square()[0]:.4f}")
    print(f"  Converged: {sol.converged}")