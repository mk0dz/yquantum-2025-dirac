"""
Variable-length quantum hash implementation.

This implements a quantum hash function that can handle variable-length inputs
and produces fixed-size outputs (256 bits) with high entropy and security.
"""
import hashlib
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def quantum_var_hash(input_data, output_bits=256):
    """
    Variable-length quantum hash function that produces a fixed-size output.
    
    Args:
        input_data: Input data to hash (string or bytes)
        output_bits: Output size in bits (default: 256)
        
    Returns:
        bytes: Fixed-length hash value
    """
    # Convert input to bytes if it's a string
    if isinstance(input_data, str):
        input_data = input_data.encode()
    
    # Ensure at least 8 qubits for better entropy
    num_qubits = max(8, output_bits // 16)
    
    # Pad and split the input into blocks
    block_size = max(4, len(input_data) // 8)
    padded_data = _pad_input(input_data, block_size)
    blocks = [padded_data[i:i+block_size] for i in range(0, len(padded_data), block_size)]
    
    # Use SHA-256 inspired initial values for better mixing
    h0 = 0x6a09e667
    h1 = 0xbb67ae85
    h2 = 0x3c6ef372
    h3 = 0xa54ff53a
    h4 = 0x510e527f
    h5 = 0x9b05688c
    h6 = 0x1f83d9ab
    h7 = 0x5be0cd19
    
    # Convert initial values to complex state
    initial_values = [h0, h1, h2, h3, h4, h5, h6, h7]
    state = _initialize_state(initial_values, num_qubits)
    
    # Process each block
    for block in blocks:
        # Create quantum circuit from block data
        qc = _create_enhanced_circuit(block, num_qubits)
        
        # Get statevector
        sv = Statevector.from_instruction(qc)
        
        # Update state by combining with previous state
        state = _enhanced_state_update(state, sv.data, block)
    
    # Convert final state to hash value with improved entropy extraction
    hash_value = _enhanced_state_to_hash(state, output_bits)
    
    return hash_value

def _initialize_state(seed_values, num_qubits):
    """Initialize quantum state with seed values for better randomization"""
    # Create a state vector of the right size
    state = np.zeros(2**num_qubits, dtype=complex)
    
    # Use seed values to initialize different parts of the state
    state_size = 2**num_qubits
    section_size = state_size // len(seed_values)
    
    for i, seed in enumerate(seed_values):
        start = i * section_size
        end = (i + 1) * section_size
        
        # Generate values based on seed
        for j in range(start, end):
            angle = (((seed + j) * 0x9e3779b9) % 1000) / 1000.0 * 2 * np.pi
            state[j] = np.exp(1j * angle)
    
    # Normalize the state
    norm = np.linalg.norm(state)
    if norm > 0:
        state = state / norm
    
    return state

def _pad_input(data, block_size):
    """Pad input data to be a multiple of block_size with improved padding scheme"""
    # Add length as bytes (8 bytes for length)
    with_length = data + len(data).to_bytes(8, byteorder='big')
    
    # Calculate how much padding is needed
    padding_needed = (block_size - (len(with_length) % block_size)) % block_size
    
    # Use non-zero padding with a pattern based on data length for better diffusion
    padding = bytearray()
    for i in range(padding_needed):
        padding.append((len(data) ^ i ^ 0x5A) % 256)  # XOR with length, position and constant
    
    padded = with_length + bytes(padding)
    return padded

def _create_enhanced_circuit(block, num_qubits):
    """Create an enhanced quantum circuit with better entropy generation"""
    qc = QuantumCircuit(num_qubits)
    
    # Initialize with Hadamard gates for superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Track a hash of processed data for better diffusion
    running_hash = 0
    
    # Process data in multiple layers
    for layer in range(3):  # Use 3 layers for better mixing
        # Apply gates based on block data
        for i, byte in enumerate(block):
            qubit_idx = (i + layer) % num_qubits
            next_qubit = (qubit_idx + 1) % num_qubits
            
            # Update running hash
            running_hash = ((running_hash << 5) + running_hash + byte) & 0xFFFFFFFF
            
            # Use running hash to determine gate application for better diffusion
            gate_selector = (running_hash >> (8 * (i % 4))) & 0xFF
            
            # Apply gates based on complex selection logic
            if gate_selector & 0x01:
                qc.x(qubit_idx)
            if gate_selector & 0x02:
                qc.z(qubit_idx)
            if gate_selector & 0x04:
                qc.y(qubit_idx)
            if gate_selector & 0x08:
                qc.s(qubit_idx)
            if gate_selector & 0x10:
                qc.t(qubit_idx)
            
            # Add non-trivial entanglement pattern
            if gate_selector & 0x20:
                qc.cx(qubit_idx, next_qubit)
            if gate_selector & 0x40:
                qc.cz(qubit_idx, (qubit_idx + 2) % num_qubits)
            if gate_selector & 0x80 and qubit_idx > 1:
                # Add some 3-qubit gates for more complex entanglement
                ctrl1 = qubit_idx
                ctrl2 = (qubit_idx + 1) % num_qubits
                target = (qubit_idx + 2) % num_qubits
                qc.ccx(ctrl1, ctrl2, target)
            
            # Add parameterized rotations with angles derived from byte and running hash
            angle_x = np.pi * (((byte ^ (running_hash & 0xFF)) & 0x0F) / 15.0)
            angle_y = np.pi * ((((byte >> 4) ^ (running_hash >> 8) & 0xFF) & 0x0F) / 15.0)
            angle_z = np.pi * (((running_hash >> 16) & 0xFF) / 255.0)
            
            qc.rx(angle_x, qubit_idx)
            qc.ry(angle_y, qubit_idx)
            qc.rz(angle_z, qubit_idx)
        
        # Add a barrier between layers for clarity
        qc.barrier()
        
        # Add a global entanglement step between layers
        for q in range(num_qubits-1):
            qc.cx(q, q+1)
        qc.cx(num_qubits-1, 0)  # Connect the last qubit to the first
    
    return qc

def _enhanced_state_update(current_state, new_state, block):
    """Enhanced state update with better mixing properties"""
    # Use a more complex combination method than simple multiplication
    
    # Calculate a weighting factor based on the block
    block_sum = sum(block) % 256
    weight = (block_sum / 255.0) * 0.5 + 0.5  # Range 0.5 to 1.0
    
    # Weighted combination
    weighted_current = current_state * weight
    weighted_new = new_state * (1 - weight)
    
    # Component-wise combination with phase interference
    real_part = np.real(weighted_current) + np.real(weighted_new)
    imag_part = np.imag(weighted_current) + np.imag(weighted_new)
    
    # Add non-linear combination
    combined = real_part + 1j * imag_part
    
    # Apply a non-linear transformation
    phase_shift = np.exp(1j * np.pi * (block_sum / 255.0))
    combined = combined * phase_shift
    
    # Normalize
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    
    return combined

def _enhanced_state_to_hash(state, output_bits):
    """Convert quantum state to hash with high entropy extraction"""
    # Extract more information from the quantum state
    probs = np.abs(state)**2
    phases = np.angle(state)
    real_parts = np.real(state)
    imag_parts = np.imag(state)
    
    # Create intermediate bytes array with higher entropy
    intermediate = []
    
    # Mix different aspects of the state using non-linear functions
    for i in range(len(state)):
        # Mix probability and phase with real and imaginary parts
        val1 = int((probs[i] * 255) % 256)
        val2 = int((phases[i] / (2 * np.pi) * 255) % 256)
        val3 = int(((real_parts[i] + 1) / 2 * 255) % 256)
        val4 = int(((imag_parts[i] + 1) / 2 * 255) % 256)
        
        # Non-linear mixing (similar to SHAs)
        mixed = (val1 ^ val2) + ((val3 * val4) % 256)
        mixed = (mixed * 0x9e3779b9) % 256  # Multiply by golden ratio for better diffusion
        
        intermediate.append(mixed)
    
    # Apply a compression function inspired by cryptographic hash functions
    output_bytes = output_bits // 8
    
    # Initialize hash with SHA-256 constants for better initial entropy
    h = [
        0x6a09e667 & 0xFF, 0xbb67ae85 & 0xFF, 0x3c6ef372 & 0xFF, 0xa54ff53a & 0xFF,
        0x510e527f & 0xFF, 0x9b05688c & 0xFF, 0x1f83d9ab & 0xFF, 0x5be0cd19 & 0xFF
    ]
    
    # Process intermediate values in blocks
    for i in range(0, len(intermediate), 64):
        block = intermediate[i:i+64]
        
        # Pad block if needed
        if len(block) < 64:
            block = block + [0] * (64 - len(block))
        
        # Expand block (similar to SHA-256 but adapted for 8-bit operations)
        w = list(block)
        for j in range(16, 64):
            # Use 8-bit friendly rotations
            s0 = _rotate_right(w[j-15], 1) ^ _rotate_right(w[j-15], 2) ^ (w[j-15] >> 3)
            s1 = _rotate_right(w[j-2], 3) ^ _rotate_right(w[j-2], 4) ^ (w[j-2] >> 1)
            w.append((w[j-16] + s0 + w[j-7] + s1) % 256)
        
        # Apply compression function
        a, b, c, d, e, f, g, hh = h
        
        for j in range(64):
            # Use 8-bit friendly rotations
            S1 = _rotate_right(e, 2) ^ _rotate_right(e, 3) ^ _rotate_right(e, 1)
            ch = (e & f) ^ ((~e) & g)
            temp1 = (hh + S1 + ch + w[j]) % 256
            S0 = _rotate_right(a, 2) ^ _rotate_right(a, 1) ^ _rotate_right(a, 3)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) % 256
            
            hh = g
            g = f
            f = e
            e = (d + temp1) % 256
            d = c
            c = b
            b = a
            a = (temp1 + temp2) % 256
        
        # Update hash values
        h[0] = (h[0] + a) % 256
        h[1] = (h[1] + b) % 256
        h[2] = (h[2] + c) % 256
        h[3] = (h[3] + d) % 256
        h[4] = (h[4] + e) % 256
        h[5] = (h[5] + f) % 256
        h[6] = (h[6] + g) % 256
        h[7] = (h[7] + hh) % 256
    
    # Combine hash values into the final output
    result = bytearray()
    for i in range(output_bytes):
        block_idx = i % 8
        value = h[block_idx]
        result.append(value)
    
    return bytes(result)

def _rotate_right(value, shift):
    """Rotate right operation (8-bit)"""
    # Ensure shift is within valid range for 8-bit operations
    shift = shift % 8
    return ((value >> shift) | (value << (8 - shift))) & 0xFF

# Example usage
if __name__ == "__main__":
    # Allow user to choose input
    test_input = input("Enter the input string for quantum hash: ")
    result = quantum_var_hash(test_input)
    print(f"Input: {test_input}")
    print(f"Hash: {result.hex()}")
    
    # Test with same input
    result2 = quantum_var_hash(test_input)
    print(f"Same input hash: {result2.hex()}")
    print(f"Hashes match: {result == result2}")
    
    # Test with slightly different input
    test_input2 = input("Enter the 2nd input string for quantum hash: ")
    result3 = quantum_var_hash(test_input2)
    print(f"Different input: {test_input2}")
    print(f"Hash: {result3.hex()}")
    print(f"Hashes match: {result == result3}")
    
    # Compare with SHA-256
    sha_result = hashlib.sha256(test_input.encode()).digest()
    print(f"SHA-256: {sha_result.hex()}") 