Generative Quantum Eigensolver (GQE)
-------------------------------------

The Generative Quantum Eigensolver (GQE) is a novel hybrid quantum-classical algorithm that leverages classical generative models to produce quantum circuits for finding ground state energies. 
GQE employs a generative model to dynamically construct quantum circuits with desired properties.

A notable implementation of GQE is the GPT Quantum Eigensolver (GPT-QE), which uses a transformer-based architecture. This approach combines:

- Pre-training capabilities using existing quantum circuit datasets
- Ability to learn without prior knowledge (from scratch)
- Particular effectiveness in electronic structure problems

Key features of GQE:

- Generative approach: Uses classical generative models to produce quantum circuits
- Flexible learning: Can leverage pre-training or learn from scratch
- Adaptive optimization: Dynamically constructs and improves circuits during optimization
- Broad applicability: Extends beyond Hamiltonian simulation to other quantum computing applications
- Scalable: Can leverage multiple QPUs for parallel evaluation

GQE Algorithm Overview:

1. Initialize or load a pre-trained generative model
2. Generate candidate quantum circuits
3. Evaluate circuit performance on target Hamiltonian
4. Update the generative model based on results
5. Repeat generation and optimization until convergence

The GQE implementation in CUDA-Q Solvers is based on the `GQE paper <https://arxiv.org/abs/2401.09253>`_.

CUDA-Q Solvers Implementation
+++++++++++++++++++++++++++++

CUDA-Q Solvers provides a high-level interface for running GQE simulations. Here's how to use it:

.. tab:: Python

   .. literalinclude:: ../../examples/solvers/python/gqe_mpi.py
      :language: python
      :start-after: [Begin Documentation]

The CUDA-Q Solvers implementation of GQE provides a flexible framework for adaptive circuit construction and optimization. 
The algorithm can efficiently utilize multiple QPUs through MPI for parallel operator evaluation, making it suitable for larger quantum systems. 


   
