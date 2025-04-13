import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Pauli
import math

class QuantumHash:
    def __init__(self, num_qubits=16, num_layers=4):
        """
        Initialize a quantum hash function with specified parameters.
        
        Args:
            num_qubits (int): Number of qubits in the circuit (max 20)
            num_layers (int): Number of repeating circuit layers (depth)
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.TOTAL_BITS = 16
        self.FRACTION_BITS = 15
    
    def _bytes_to_parameters(self, input_bytes):
        """
        Convert input bytes to circuit parameters.
        Uses a complex mixing function to ensure each input bit affects multiple parameters.
        
        Args:
            input_bytes (bytes): Input data to hash
            
        Returns:
            list: Angles for quantum gates
        """
        # Ensure we have at least 32 bytes (for proper mixing)
        if len(input_bytes) < 32:
            padded = bytearray(input_bytes)
            padded.extend([0] * (32 - len(input_bytes)))
            input_bytes = bytes(padded)
        
        # Create initial parameter set
        params = []
        
        # Process input bytes in chunks and apply a mixing function
        chunks = [input_bytes[i:i+4] for i in range(0, len(input_bytes), 4)]
        
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                # Create a unique index based on layer and qubit
                idx = (layer * self.num_qubits + qubit) % len(chunks)
                
                # Mix bytes from this chunk
                chunk = chunks[idx]
                chunk_val = int.from_bytes(chunk, byteorder='big')
                
                # Apply non-linear mixing function and scale to [0, 2π]
                mix_val = (chunk_val ^ (chunk_val >> 11) ^ (layer * 0x5a5a5a5a + qubit * 0x3c3c3c3c))
                angle = (mix_val % 1000) / 1000.0 * 2 * np.pi
                
                params.append(angle)
                
                # For RZ gates, use a different mixing function
                mix_val2 = (chunk_val ^ (chunk_val << 7) ^ (layer * 0x33333333 + qubit * 0x77777777))
                angle2 = (mix_val2 % 1000) / 1000.0 * 2 * np.pi
                
                params.append(angle2)
        
        return params
    
    def _build_circuit(self, params):
        """
        Build the parameterized quantum circuit.
        
        Args:
            params (list): Parameter values for the circuit
            
        Returns:
            QuantumCircuit: The constructed quantum circuit
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Apply Hadamard gates to create superposition
        for i in range(self.num_qubits):
            qc.h(i)
        
        param_idx = 0
        
        # Add layered structure with rotations and entanglement
        for l in range(self.num_layers):
            # RY rotations
            for i in range(self.num_qubits):
                qc.ry(params[param_idx], i)
                param_idx += 1
            
            # RZ rotations
            for i in range(self.num_qubits):
                qc.rz(params[param_idx], i)
                param_idx += 1
            
            # Add entanglement with CNOT gates
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
            
            # Add cross-entanglement (connect distant qubits)
            for i in range(self.num_qubits // 2):
                qc.cx(i, self.num_qubits - i - 1)
                
            # Add a non-linear operation with Toffoli gates (if enough qubits)
            if self.num_qubits >= 3:
                for i in range(self.num_qubits - 2):
                    qc.ccx(i, i + 1, i + 2)
            
        return qc
    
    def _to_fixed(self, x):
        """
        Convert a float expectation to fixed-point.
        
        Args:
            x (float): Value to convert
            
        Returns:
            int: Fixed-point representation
        """
        fraction_mult = 1 << self.FRACTION_BITS
        return int(x * fraction_mult + (0.5 if x >= 0 else -0.5))
    
    def _pack_expectations(self, expectations):
        """
        Pack expectation values into output bytes.
        
        Args:
            expectations (list): Expectation values from quantum circuit
            
        Returns:
            bytes: Output hash value
        """
        # Convert expectations to fixed-point
        fixed_exps = [self._to_fixed(exp) for exp in expectations]
        
        # Pack into bytes
        data = []
        for fixed in fixed_exps:
            for i in range(self.TOTAL_BITS // 8):
                data.append((fixed >> (8 * i)) & 0xFF)
        
        return bytes(data)
    
    def _stretch_output(self, output, target_length):
        """
        Stretch or compress output to match target length.
        
        Args:
            output (bytes): Initial output bytes
            target_length (int): Desired length in bytes
            
        Returns:
            bytes: Adjusted output
        """
        if len(output) == target_length:
            return output
        
        # If output is too short, repeat it
        if len(output) < target_length:
            repeats = target_length // len(output) + 1
            extended = output * repeats
            return extended[:target_length]
        
        # If output is too long, compress it
        step = len(output) / target_length
        result = bytearray()
        
        for i in range(target_length):
            idx = min(int(i * step), len(output) - 1)
            result.append(output[idx])
            
        return bytes(result)
    
    def hash(self, input_data):
        """
        Hash the input data using quantum circuit simulation.
        
        Args:
            input_data (bytes): Data to hash
            
        Returns:
            bytes: Hash value of same length as input
        """
        target_length = len(input_data)
        
        # Convert input to circuit parameters
        params = self._bytes_to_parameters(input_data)
        
        # Build and run the quantum circuit
        qc = self._build_circuit(params)
        
        # Prepare the state vector from the circuit
        sv = Statevector.from_instruction(qc)
        
        # Calculate expectation values on X, Y, and Z axes to maximize entropy
        x_exps = [sv.expectation_value(Pauli("X"), [i]).real for i in range(self.num_qubits)]
        y_exps = [sv.expectation_value(Pauli("Y"), [i]).real for i in range(self.num_qubits)]
        z_exps = [sv.expectation_value(Pauli("Z"), [i]).real for i in range(self.num_qubits)]
        
        # Combine expectations for more entropy
        all_exps = x_exps + y_exps + z_exps
        
        # Pack the expectations into bytes
        output = self._pack_expectations(all_exps)
        
        # Resize to match input length
        return self._stretch_output(output, target_length)


# Function to be used in main.py
def quantum_hash(input_bytes):
    """
    Hash function that takes bytes as input and returns bytes as output.
    
    Args:
        input_bytes (bytes): Input data to hash
        
    Returns:
        bytes: Hash value of same length as input
    """
    hasher = QuantumHash(num_qubits=16, num_layers=4)
    return hasher.hash(input_bytes)


# Example usage
if __name__ == "__main__":
    # Test with a 32-byte input
    test_input = b"This is a test of the quantum hash."
    result = quantum_hash(test_input)
    print(f"Input length: {len(test_input)} bytes")
    print(f"Output length: {len(result)} bytes")
    print(f"First few bytes: {result[:8].hex()}")