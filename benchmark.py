"""
Benchmark script for comparing quantum hash implementations.
"""
import time
import random
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from collections import defaultdict
from scipy import stats
import math

# Import hash functions
from solution.variable_hash import quantum_var_hash
from solution.qubitcoin_hash import qubitcoin_hash
from solution.optimized_hash import optimized_quantum_hash

def generate_test_data(sizes, num_samples=10):
    """Generate test data with various sizes"""
    data = []
    size_map = {}  # To track which size each data belongs to
    
    for size in sizes:
        for _ in range(num_samples):
            sample = bytes(random.randint(0, 255) for _ in range(size))
            data.append(sample)
            size_map[sample] = size
            
    return data, size_map

def calculate_bit_difference(hash1, hash2):
    """Calculate percentage of bits that differ between two hashes"""
    bit_diff = 0
    for b1, b2 in zip(hash1, hash2):
        xor = b1 ^ b2
        bit_diff += bin(xor).count('1')
    return (bit_diff / (8 * len(hash1))) * 100

def calculate_entropy(data):
    """Calculate Shannon entropy of a byte sequence"""
    if not data:
        return 0
    
    # Count occurrences of each byte value
    counts = defaultdict(int)
    for byte in data:
        counts[byte] += 1
    
    # Calculate entropy
    length = len(data)
    entropy = 0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    
    # Normalize to 0-8 range (8 bits per byte)
    return entropy

def measure_avalanche_effect(hash_func, num_samples=50, input_size=64, return_detailed=False):
    """Measure how much the output hash changes when input is slightly modified"""
    results = []
    detailed_results = []
    
    for _ in range(num_samples):
        # Generate random input
        data = bytes(random.randint(0, 255) for _ in range(input_size))
        original_hash = hash_func(data)
        
        # Modify a single bit
        modified = bytearray(data)
        byte_pos = random.randint(0, len(modified) - 1)
        bit_pos = random.randint(0, 7)
        modified[byte_pos] ^= (1 << bit_pos)
        
        # Hash the modified input
        modified_hash = hash_func(bytes(modified))
        
        # Calculate percentage of bits changed
        result = calculate_bit_difference(original_hash, modified_hash)
        results.append(result)
        
        if return_detailed:
            bit_changes = []
            for b1, b2 in zip(original_hash, modified_hash):
                xor = b1 ^ b2
                for i in range(8):
                    bit_changes.append(1 if (xor & (1 << i)) else 0)
            detailed_results.append(bit_changes)
    
    if return_detailed:
        return np.mean(results), detailed_results
    return np.mean(results)

def analyze_hash_distribution(hash_func, num_samples=100, input_size=64):
    """Analyze distribution properties of the hash function"""
    hashes = []
    byte_frequencies = defaultdict(int)
    entropies = []
    
    for _ in range(num_samples):
        # Generate random input
        data = bytes(random.randint(0, 255) for _ in range(input_size))
        hash_value = hash_func(data)
        hashes.append(hash_value)
        
        # Collect byte frequencies
        for byte in hash_value:
            byte_frequencies[byte] += 1
        
        # Calculate entropy of each hash
        entropies.append(calculate_entropy(hash_value))
    
    # Analyze byte distribution
    byte_values = list(range(256))
    byte_counts = [byte_frequencies.get(b, 0) for b in byte_values]
    
    # Chi-square test for uniformity - properly adjusted to ensure sums match
    total_bytes = num_samples * len(hashes[0])
    expected_freq = total_bytes / 256  # Expected frequency per byte value
    observed = np.array([byte_frequencies.get(b, 0) for b in byte_values])
    
    # Calculate Chi-square statistic manually instead of using stats.chisquare
    # to avoid issues with sum mismatch
    expected = np.ones(256) * expected_freq
    chi2_stat = np.sum(((observed - expected) ** 2) / expected)
    # Approximate p-value based on chi-square distribution with 255 degrees of freedom
    p_value = 1.0 - stats.chi2.cdf(chi2_stat, 255)
    
    return {
        'hashes': hashes,
        'byte_frequencies': dict(byte_frequencies),
        'byte_distribution': (byte_values, byte_counts),
        'entropy_mean': np.mean(entropies),
        'entropy_std': np.std(entropies),
        'chi2_stat': chi2_stat,
        'p_value': p_value
    }

def measure_performance(hash_func, test_data, size_map=None):
    """Measure performance of a hash function"""
    times = []
    sizes_times = defaultdict(list) if size_map else None
    
    for data in test_data:
        start_time = time.time()
        hash_func(data)
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        if size_map:
            size = size_map[data]
            sizes_times[size].append(elapsed)
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    result = {
        'mean': mean_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'times': times
    }
    
    if size_map:
        size_stats = {}
        for size, size_times in sizes_times.items():
            size_stats[size] = {
                'mean': np.mean(size_times),
                'std': np.std(size_times),
                'times': size_times
            }
        result['by_size'] = size_stats
    
    return result

def benchmark_all():
    """Run benchmarks on all hash functions"""
    print("Running hash function benchmarks...")
    
    # Configure test parameters
    sizes = [32, 64, 128, 256, 512]
    test_data, size_map = generate_test_data(sizes)
    
    # Hash functions to test
    hash_functions = [
        ("Variable-length Quantum Hash", quantum_var_hash),
        ("QubitCoin Hash", qubitcoin_hash),
        ("Optimized Quantum Hash", optimized_quantum_hash),
        ("SHA-256 (Classical)", lambda x: hashlib.sha256(x).digest())
    ]
    
    # Run benchmarks
    results = []
    detailed_data = {}
    
    for name, func in hash_functions:
        print(f"Testing {name}...")
        
        # Measure performance
        perf_stats = measure_performance(func, test_data, size_map)
        
        # Measure avalanche effect
        avg_avalanche, bit_changes = measure_avalanche_effect(func, return_detailed=True)
        
        # Analyze hash distribution
        distribution = analyze_hash_distribution(func)
        
        # Add to results
        results.append({
            'name': name,
            'time': perf_stats['mean'],
            'time_std': perf_stats['std'],
            'avalanche': avg_avalanche,
            'entropy': distribution['entropy_mean']
        })
        
        # Save detailed data for visualization
        detailed_data[name] = {
            'performance': perf_stats,
            'bit_changes': bit_changes,
            'distribution': distribution
        }
    
    # Display results
    print("\nResults:")
    table_data = []
    for r in results:
        table_data.append([
            r['name'], 
            f"{r['time']:.6f} sec (±{r['time_std']:.6f})", 
            f"{r['avalanche']:.2f}%",
            f"{r['entropy']:.2f}/8.0",
            "Excellent" if 45 <= r['avalanche'] <= 55 else 
            "Good" if 40 <= r['avalanche'] <= 60 else "Poor"
        ])
    
    headers = ["Hash Function", "Average Time", "Avalanche Effect", "Entropy", "Quality"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Create visualizations directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(results, detailed_data, sizes)
    
    print("\nResults and visualizations saved to 'visualizations' directory")

def generate_visualizations(results, detailed_data, sizes):
    """Generate various visualizations from the benchmark data"""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Extract data for plotting
    names = [r['name'] for r in results]
    times = [r['time'] for r in results]
    avalanche = [r['avalanche'] for r in results]
    entropy = [r['entropy'] for r in results]
    
    # 1. Basic comparison chart (enhanced version)
    plt.figure(figsize=(15, 10))
    
    # Plot time (log scale)
    plt.subplot(2, 2, 1)
    bars = plt.bar(names, times)
    plt.yscale('log')
    plt.ylabel('Time (seconds, log scale)')
    plt.title('Performance Comparison')
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}',
                ha='center', va='bottom', rotation=0)
    
    # Plot avalanche effect
    plt.subplot(2, 2, 2)
    bars = plt.bar(names, avalanche)
    plt.axhline(y=50, color='r', linestyle='--', label='Ideal (50%)')
    plt.ylabel('Avalanche Effect (%)')
    plt.title('Security Comparison - Avalanche Effect')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Plot entropy
    plt.subplot(2, 2, 3)
    bars = plt.bar(names, entropy)
    plt.axhline(y=8, color='r', linestyle='--', label='Perfect (8 bits)')
    plt.ylabel('Entropy (bits)')
    plt.ylim(0, 8.5)
    plt.title('Security Comparison - Entropy')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Radar chart
    plt.subplot(2, 2, 4, polar=True)
    
    # Normalize values for radar chart
    max_time = max(times)
    norm_times = [1 - (t / max_time) for t in times]  # Invert so faster is better
    norm_avalanche = [a / 50 if a <= 50 else (100 - a) / 50 for a in avalanche]  # Closer to 50% is better
    norm_entropy = [e / 8 for e in entropy]  # Higher entropy is better
    
    # Plot radar chart
    categories = ['Speed', 'Avalanche', 'Entropy']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    ax = plt.subplot(2, 2, 4, polar=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    for i, name in enumerate(names):
        values = [norm_times[i], norm_avalanche[i], norm_entropy[i]]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    plt.xticks(angles[:-1], categories)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.title('Overall Quality Comparison')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('visualizations/hash_comparison.png', dpi=300)
    plt.close()
    
    # 2. Performance by input size
    plt.figure(figsize=(12, 8))
    
    for name in names:
        if 'by_size' in detailed_data[name]['performance']:
            size_data = detailed_data[name]['performance']['by_size']
            x = sorted(size_data.keys())
            y = [size_data[size]['mean'] for size in x]
            
            plt.plot(x, y, marker='o', label=name)
    
    plt.xlabel('Input Size (bytes)')
    plt.ylabel('Time (seconds)')
    plt.title('Performance by Input Size')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('visualizations/performance_by_size.png', dpi=300)
    plt.close()
    
    # 3. Timing distribution box plots
    plt.figure(figsize=(12, 8))
    
    timing_data = [detailed_data[name]['performance']['times'] for name in names]
    plt.boxplot(timing_data, labels=names)
    plt.yscale('log')
    plt.ylabel('Time (seconds, log scale)')
    plt.title('Distribution of Execution Times')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/timing_distribution.png', dpi=300)
    plt.close()
    
    # 4. Byte distribution heatmaps (one per hash function)
    for name in names:
        plt.figure(figsize=(12, 8))
        
        # Get byte distribution data
        byte_values, byte_counts = detailed_data[name]['distribution']['byte_distribution']
        
        # Reshape for heatmap (16x16 grid for 256 byte values)
        heatmap_data = np.zeros((16, 16))
        for i in range(256):
            row = i // 16
            col = i % 16
            heatmap_data[row, col] = byte_counts[i]
        
        # Plot heatmap
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title(f'Byte Distribution - {name}')
        plt.xlabel('Lower 4 bits (hex)')
        plt.ylabel('Upper 4 bits (hex)')
        plt.tight_layout()
        plt.savefig(f'visualizations/byte_distribution_{name.replace(" ", "_")}.png', dpi=300)
        plt.close()
    
    # 5. Bit change visualization
    for name in names:
        bit_changes = detailed_data[name]['bit_changes']
        
        if bit_changes:
            # Create a 2D array for visualization
            # Each row represents a test case, each column a bit position
            bit_matrix = np.array(bit_changes)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(bit_matrix[:50, :64], cmap='binary', cbar=False)
            plt.title(f'Bit Change Pattern - {name}')
            plt.xlabel('Bit Position')
            plt.ylabel('Test Case')
            plt.tight_layout()
            plt.savefig(f'visualizations/bit_changes_{name.replace(" ", "_")}.png', dpi=300)
            plt.close()
            
            # Also plot histogram of bit changes per test
            plt.figure(figsize=(10, 6))
            bit_change_counts = np.sum(bit_matrix, axis=1)
            sns.histplot(bit_change_counts, kde=True)
            plt.axvline(np.mean(bit_change_counts), color='r', linestyle='--', 
                        label=f'Mean: {np.mean(bit_change_counts):.1f} bits')
            ideal = len(bit_matrix[0]) / 2
            plt.axvline(ideal, color='g', linestyle='--', 
                        label=f'Ideal: {ideal:.1f} bits')
            plt.title(f'Distribution of Bit Changes - {name}')
            plt.xlabel('Number of Bits Changed')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'visualizations/bit_change_histogram_{name.replace(" ", "_")}.png', dpi=300)
            plt.close()

if __name__ == "__main__":
    benchmark_all() 