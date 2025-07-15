"""
Example usage of TensorNetworkDecoder from cudaq_qec.

This script demonstrates how to instantiate and use the tensor network decoder
for a simple parity check code and logical observable.

Requirements:
- cudaq_qec
- quimb
- autoray
- cupy (optional, for GPU acceleration)

These requirements can be installed via pip:

pip install cudaq-qec[tn_decoder]

"""

import numpy as np
import cudaq_qec as qec

# Example parity check matrix (H) for a [3,1] repetition code
H = np.array([
    [1, 1, 0],
    [0, 1, 1]
], dtype=np.uint8)

# Logical observable: parity of all bits
logical_obs = np.array([[1, 1, 1]], dtype=np.uint8)

# Simple noise model: independent bit-flip probability p for each qubit
p = 0.1
noise_model = [p, p, p]

# Instantiate the decoder
decoder = qec.get_decoder(
    "tensor_network_decoder",
    H=H,
    logical_obs=logical_obs,
    noise_model=noise_model,
    dtype="float32",
    device="cpu"  # Use "cuda" for GPU if available
)

# Example syndrome: no error detected
syndrome = [0.0, 0.0]  # probabilities for each check

result = decoder.decode(syndrome)
print("Decoded logical observable probability (no error):", result.result)

# Example syndrome: error detected on second check
syndrome = [0.0, 1.0]
result = decoder.decode(syndrome)
print("Decoded logical observable probability (error on second check):", result.result)

# Batch decoding example
syndrome_batch = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0]
], dtype=np.float32)

batch_results = decoder.decode_batch(syndrome_batch)
print("Batch decoded logical observable probabilities:")
for i, res in enumerate(batch_results):
    print(f"  Syndrome {i}: {res.result}")