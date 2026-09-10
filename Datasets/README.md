# Datasets

## Synthetic datasets and workloads

### Scalability

The scripts used to generate the synthetic inputs for the sample-size
scalability study are available in [`scalability/`](scalability/README.md).

### Stress test

The manuscript's stress-test and hotspot analysis uses a synthetic workload
with `N = 10^7`, `D = 32`, and `K = 32`. Its NumPy generator is available at
[`stress/generate_stress_test.py`](stress/generate_stress_test.py).

### Local-memory boundary

The local-memory boundary experiments use a deterministic synthetic workload
with `N = 40,960` and a 393-dimensional master dataset. The generator is
available in [`local_memory_boundary/`](local_memory_boundary/).

### Single-device capacity

The single-device capacity experiment uses programmatically constructed
synthetic workloads with variable `N`; no standalone dataset file is required.

See [`SYNTHETIC_DATASETS.md`](SYNTHETIC_DATASETS.md) for the archived
experimental inputs, generation procedures, and parameters of the synthetic
workloads.

## Generator requirements

- Scalability generators: Python 3, NumPy, and Matplotlib.
- Stress generator: Python 3 and NumPy.
- Local-memory boundary generator: Python 3 standard library.

## Generator output contract

The scalability, stress, and local-memory boundary generators each produce a
comma-separated feature CSV with one header row. These feature CSV files can
be passed directly to oneFCM with the `-i` option using the matching `-n` and
`-d` values.

## Real-world datasets

### US Census Data (1990) / Census90

The manuscript uses the U.S. Census 1990 dataset for scalability experiments.
The original data are available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/116/us+census+data+1990).

### GID / Five-Billion-Pixels

The remote-sensing workload in the end-to-end evaluation uses Gaofen-2 imagery
from [Five-Billion-Pixels](https://x-ytong.github.io/project/Five-Billion-Pixels.html)
and land-cover annotations from the [Gaofen Image Dataset (GID)](https://x-ytong.github.io/project/GID.html).

### KDD Cup 1999

The manuscript uses the KDD Cup 1999 network intrusion detection dataset in the
real-world end-to-end evaluation. The original data are available from the
[official ACM SIGKDD KDD Cup 1999 page](https://kdd.org/kdd-cup/view/kdd-cup-1999/Data).

### ElectricityLoadDiagrams20112014

This electricity load dataset contains quarter-hourly consumption measurements
for multiple clients and is used in the real-world end-to-end evaluation. The
original data are available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams2011).
