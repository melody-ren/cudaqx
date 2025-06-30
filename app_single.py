import cudaq
from cudaq import spin
from algo.gqe import gqe  

qubit_count = 2

# Generate an operator pool for the GQE
def ops_pool(n):
    pool = []
    for i in range(n):
        pool.append(cudaq.SpinOperator(spin.x(i)))
        pool.append(cudaq.SpinOperator(spin.y(i)))
        pool.append(cudaq.SpinOperator(spin.z(i)))
    for i in range(n - 1):
        pool.append(cudaq.SpinOperator(spin.z(i) * spin.z(i + 1)))  # ZZ entangling
    return pool

pool = ops_pool(qubit_count)

# Define a simple Hamiltonian: Z₀ + Z₁
ham = spin.z(0) + spin.z(1)

# Helper functions to extract coeffs and Pauli words
def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    return [term.evaluate_coefficient() for term in op]

def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
    return [term.get_pauli_word(qubit_count) for term in op]

# Kernel that applies the selected operators
@cudaq.kernel
def kernel(qcount: int, coeffs: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(qcount)
    h(q)
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])

# cost function for GQE
def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):
    full_coeffs = []
    full_words = []
    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)

    return cudaq.observe(kernel, ham, qubit_count, full_coeffs, full_words).expectation()

# Run GQE
minE, best_ops = gqe(cost, pool, max_iters=25, ngates=4)

print(f'Ground Energy = {minE}')
print('Ansatz Ops')

