Generative Quantum Eigensolver (GQE)
-------------------------------------

The GQE algorithm samples and generates circuits from Boltzmann distribution according to the cost function which is defined using the logit matching method.

A notable implementation of GQE is the GPT Quantum Eigensolver (GPT-QE), which uses a transformer-based architecture. 
This approach combines:

- Pre-training capabilities using existing quantum circuit datasets
- Ability to learn without prior knowledge 

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

   .. literalinclude:: ../../examples/solvers/python/gqe_h2.py
      :language: python
      :start-after: [Begin Documentation]

The CUDA-Q Solvers implementation of GQE provides a flexible framework for adaptive circuit construction and optimization. 
The algorithm can efficiently utilize multiple QPUs through MPI for parallel operator evaluation, making it suitable for larger quantum systems. 

.. note::

   The GQE implementation is a Python-only implementation.


   
