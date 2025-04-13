# Quantum Hash Implementation Challenge

This repository contains implementations of quantum hash functions developed for the quantum cryptography challenge. Our goal was to create efficient and secure quantum hash algorithms that outperform existing implementations like QubitCoin.

*read the [paper:](/paper/pdf.pdf)*

## Overview

We have developed several quantum hash implementations with different performance and security characteristics:

1. **Variable-length Quantum Hash** (`variable_hash.py`): A quantum hash implementation that can handle inputs of any length with high entropy and security properties.
2. **Optimized Quantum Hash** (`optimized_hash.py`): A highly optimized quantum hash implementation that outperforms QubitCoin while maintaining good security properties.
3. **QubitCoin Hash** (`qubitcoin_hash.py`): A reference implementation of the QubitCoin approach for comparison purposes.

## Key Features of the Optimized Implementation

Our optimized quantum hash implementation includes several advanced techniques:

- **Reduced Circuit Complexity**: Using carefully selected qubits and layers for optimal performance
- **Enhanced Caching**: Aggressive caching of quantum states, circuits, and parameters
- **Efficient Parallelization**: Advanced parallel block processing with tree-reduction for combining results
- **SHA-256 Inspired Mixing**: Parameter mixing techniques inspired by SHA-256 for improved diffusion
- **Hardware-Efficient Circuit Design**: Optimized gate selection and entanglement patterns

## Enhanced Variable-length Implementation

Our variable-length quantum hash implementation provides:

- **Input Flexibility**: Processes inputs of any size with consistent output length
- **Advanced Quantum Circuit**: Uses enhanced entanglement patterns and multi-layer processing
- **Improved Entropy Extraction**: Better extraction of quantum state features for higher security
- **SHA-256 Inspired Compression**: Post-processing inspired by classical cryptographic functions
- **Excellent Avalanche Effect**: Achieves approximately 50% bit changes with small input changes

## Performance Results

Our quantum hash implementations show significant improvements over the QubitCoin approach:

| Hash Function               | Avg Time (s) | Avalanche % | Entropy | Quality  |
|-----------------------------|--------------|-------------|---------|----------|
| Variable-length Quantum Hash| 1.994557     | 50.62%      | 2.98/8.0| Excellent|
| Optimized Quantum Hash      | 0.008912     | 49.09%      | 4.91/8.0| Excellent|
| QubitCoin Hash              | 0.051647     | 35.20%      | 4.88/8.0| Poor     |
| SHA-256 (classical)         | 0.000001     | 50.07%      | 4.89/8.0| Excellent|

The optimized implementation is **5.5x faster** than QubitCoin while providing much better avalanche effect (49.09% vs 35.20%, closer to the ideal 50%).



## Installation

### Requirements

```
numpy
qiskit
qiskit-aer
matplotlib
tabulate
scipy
seaborn
```

Install the requirements:

```bash
pip install -r requirements.txt
```

## Benchmarking & Visualization

We've implemented a comprehensive benchmarking framework to analyze and visualize the performance and security properties of Our hash functions:

### Key Metrics Measured

- **Performance**: Execution time across different input sizes
- **Avalanche Effect**: How many output bits change when a single input bit is flipped
- **Entropy**: Statistical randomness of the output hash values
- **Byte Distribution**: Uniformity of byte values in hash outputs
- **Bit Change Patterns**: How individual bits respond to input changes

### Generated Visualizations

Running the benchmark script (`benchmark.py`) generates various visualizations in the `visualizations` directory:

- **Comparative Analysis**: Summary charts comparing all hash functions
- **Performance by Input Size**: How execution time scales with input length
- **Timing Distribution**: Box plots showing execution time variance
- **Byte Distribution Heatmaps**: 16x16 grids showing byte value frequencies
- **Bit Change Patterns**: Heatmaps of which bits change with input modifications
- **Avalanche Histograms**: Distribution of bit changes across test cases

To run the benchmarks and generate visualizations:

```bash
python benchmark.py
```


The results will be saved in the `visualizations` directory, including:
- Performance comparison charts
- Security analysis visualizations
- Byte distribution heatmaps
- Detailed bit change pattern analysis



## Future Improvements

Potential future improvements include:

1. Further optimizing the circuit design for specific quantum hardware architectures
2. Improving the entropy of the variable-length hash implementation
3. Implementing runtime adaptive behavior to optimize for different input sizes
4. Exploring alternative entanglement patterns for better security properties 
