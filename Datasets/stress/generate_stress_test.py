import numpy as np
import time


def generate_synthetic_data():
    # Configuration
    N = 10**7       # Total number of points (10,000,000)
    D = 32          # Number of dimensions
    K = 32          # Number of clusters

    # Range used to sample cluster centers
    center_box = (0, 100)
    # Cluster spread (standard deviation)
    cluster_std = 9.0

    print("==========================================")
    print(f"Generating dataset: N={N}, D={D}, K={K}")
    print("Estimated memory usage: ~1.5 GB")
    print("Estimated output size: ~3.0 GB")
    print("==========================================\n")

    start_time = time.time()

    # Allocate points evenly across clusters, including the remainder.
    points_per_cluster = N // K
    remainder = N % K

    # Preallocate arrays using compact data types.
    X = np.zeros((N, D), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int32)

    print("1. Generating cluster centers...")
    centers = np.random.uniform(center_box[0], center_box[1], (K, D)).astype(np.float32)

    print("2. Generating Gaussian-distributed points...")
    current_idx = 0

    for k in range(K):
        # Assign remainder points to the first clusters.
        n_samples = points_per_cluster + (1 if k < remainder else 0)

        # Generate Gaussian noise and translate it to the cluster center.
        noise = np.random.randn(n_samples, D).astype(np.float32) * (cluster_std / 3.0)
        X[current_idx : current_idx + n_samples] = centers[k] + noise
        labels[current_idx : current_idx + n_samples] = k

        current_idx += n_samples

    gen_time = time.time() - start_time
    print(f"   -> Data generation completed in {gen_time:.2f} seconds.\n")

    print("3. Writing stress_test_n1e7_d32_k32.csv...")
    print("   This is usually the slowest step.")

    # Header: dim1, dim2, ..., dim32 separated by commas.
    feature_header = ",".join([f"dim{i+1}" for i in range(D)])

    # Write the feature matrix without NumPy's default '#' header prefix.
    np.savetxt("stress_test_n1e7_d32_k32.csv", X, fmt='%.4f', delimiter=',',
               header=feature_header, comments='')

    print("4. Writing stress_test_labels.txt...")
    np.savetxt("stress_test_labels.txt", labels, fmt='%d',
               header="label", comments='')

    total_time = time.time() - start_time
    print(f"\nGeneration completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    generate_synthetic_data()
