# oneFCM

## Overview

This repository contains the official implementation accompanying the paper
**oneFCM: An Efficient and Portable SYCL/oneAPI Implementation of Fuzzy c-Means Clustering**.
oneFCM implements classical fuzzy c-means clustering for CPUs and GPUs using
SYCL and oneAPI. The contribution is the adaptation and integration of
established parallel techniques for FCM, not a new clustering algorithm.
The build produces the `cmeansOA` command-line executable. The code version
associated with the paper is
[v1.0.0](https://github.com/TianchunGu/oneFCM/releases/tag/v1.0.0).

## Authors

Tianchun Gu and Guanlin He.

## Citation

Please cite this paper if you use oneFCM in your research:

```bibtex
@article{guOneFCM,
  author  = {Gu, Tianchun and He, Guanlin},
  title   = {{oneFCM}: An Efficient and Portable {SYCL}/{oneAPI} Implementation of Fuzzy c-Means Clustering},
  journal = {Future Generation Computer Systems},
  year    = {in press},
  note    = {Accepted for publication}
}
```

Corresponding author: Guanlin He.

For software citation metadata, see [`CITATION.cff`](CITATION.cff).

## Features

- Device-resident FCM iteration with data, memberships, centers, and labels
  kept on the selected SYCL device during the iterative phase.
- Hierarchical center reduction with resource-aware local-memory tiling.
- Two membership-update paths:
  - Scheme A for `K <= 32`, using thread-private distances.
  - Scheme B for `K > 32`, using resource-aware two-dimensional `(T_k, T_d)`
    tiling and two passes. Its current selection contract reserves 1024 bytes
    of local memory; this is an implementation boundary, not a universal
    device-capacity claim.
- Configurable work-group sizes, including a global request and per-kernel
  center, membership, and center-change overrides.
- Fuzzifier values `m > 1` that remain valid in the production
  single-precision representation and yield a finite membership exponent,
  with an automatic `m = 2` specialization and a generic `pow` path.
- Exact 20-iteration `fixed20` execution and convergence-driven
  `native-dynamic` execution.
- Project-scoped fast-math selection, enabled or disabled at configure time.

## Requirements

- The documented release workflow assumes CMake 3.16 or newer.
- A C++17 SYCL compiler and runtime for the selected backend.
- oneDPL headers providing `oneapi/dpl/random`.

Intel oneAPI normally supplies the compiler, SYCL runtime, and oneDPL for CPU,
Intel GPU, and FPGA-emulator configurations. NVIDIA and AMD configurations
require compatible SYCL backend toolchains in addition to the vendor runtime.
oneMKL is not a project dependency.

The containerized Platform A reference environment is documented in
[`containers/README.md`](containers/README.md).


## Build

Run the commands from the repository root. Make sure the SYCL compiler and runtime for the target device are available in the current environment.

Configure and build separately for CPU and GPU targets.

### CPU

```bash
cmake -S . -B build-cpu -DHARDWARE_TARGET=CPU -DBUILD_TESTING=ON
cmake --build build-cpu -j
```

### GPU

```bash
cmake -S . -B build-gpu -DHARDWARE_TARGET=<gpu-target>
cmake --build build-gpu -j
```

Select the GPU target and toolchain appropriate for the SYCL backend available on the system.
The accepted GPU target tokens are `GPU`, `INTEL_ARC_B580`, and `AMD_GPU`;
backend-specific SYCL toolchain settings may still be required by the selected
CMake configuration.

### Backend configuration

`HARDWARE_TARGET=GPU` selects the SYCL CUDA backend for NVIDIA GPUs. Set
`CUDA_GPU_ARCH` to match the target GPU; the current default is `sm_89`, but it
is not a universal NVIDIA setting:

```bash
cmake -S . -B build-nvidia \
  -DHARDWARE_TARGET=GPU \
  -DCUDA_GPU_ARCH=sm_89
```

`HARDWARE_TARGET=AMD_GPU` uses an out-of-tree SYCL compiler with ROCm/HIP.
Provide `SYCL_LLVM_ROOT`, `SYCL_BUILD_ROOT`, and `ROCM_PATH` in the environment.
`FCM_AMD_SYCL_COMPILER` may select a specific compiler; otherwise the build
uses `SYCL_BUILD_ROOT/bin/clang++`. Set `AMD_GPU_ARCH` to match the target AMD
GPU; the current default is `gfx1200`:

```bash
SYCL_LLVM_ROOT=path/to/intel-llvm-source \
SYCL_BUILD_ROOT=path/to/intel-llvm-build \
FCM_AMD_SYCL_COMPILER=path/to/clang++ \
ROCM_PATH=path/to/rocm \
cmake -S . -B build-amd \
  -DHARDWARE_TARGET=AMD_GPU \
  -DAMD_GPU_ARCH=gfx1200
```

`HARDWARE_TARGET=INTEL_ARC_B580` uses the oneAPI Level Zero/JIT configuration
defined by this project. `HARDWARE_TARGET=FPGA` builds the current Intel FPGA
emulator path; it does not represent a verified FPGA hardware deployment.


### Fast-math selection

Fast math is enabled by default. Select the mode explicitly when the distinction
matters:

```bash
cmake -S . -B build-fast -DHARDWARE_TARGET=CPU -DFCM_ENABLE_FAST_MATH=ON
cmake -S . -B build-no-fast -DHARDWARE_TARGET=CPU -DFCM_ENABLE_FAST_MATH=OFF
```

Do not mix the project option with conflicting global `-ffast-math` or
`-fno-fast-math` flags.

## Run

The following self-contained CPU example uses the tracked
[`tests/fixtures/tiny.csv`](tests/fixtures/tiny.csv):

```bash
mkdir -p build-cpu/example-fixed20
./build-cpu/cmeansOA \
  -i tests/fixtures/tiny.csv \
  -o build-cpu/example-fixed20/labels.csv \
  -c 2 -e 0.001 -n 4 -d 2
```

`fixed20`, `m = 2`, seed 550, automatic m-power selection, and automatic
work-group selection are the defaults.

### Command-line interface

| Option | Meaning |
| --- | --- |
| `-i` | Input CSV path |
| `-o` | Output label CSV path; its parent directory must already exist |
| `-c` | Positive number of clusters `K` |
| `-e` | Positive finite threshold; validated in both protocols and used for stopping only by `native-dynamic` |
| `-n` | Positive number of data rows to read after the header |
| `-d` | Positive number of numeric dimensions to read from each row |
| `--iteration-protocol` | `fixed20` or `native-dynamic`; default `fixed20` |
| `--max-iter` | Positive iteration cap; `fixed20` accepts only 20, while `native-dynamic` requires this option explicitly |
| `--seed` | Non-negative membership-initialization seed; default 550 |
| `--fuzzifier` | Value `m > 1` that remains valid in the production single-precision representation and yields a finite membership exponent; default 2.0 |
| `--m-power-policy` | `auto` or `generic`; default `auto` |
| `--force-generic-m` | Compatibility alias for `--m-power-policy generic` |
| `--work-group-size` | Positive strict request shared by center/membership and center-change kernels |
| `--center-wg-size` | Positive center-kernel override |
| `--membership-wg-size` | Positive membership-kernel override |
| `--change-wg-size` | Positive center-change-kernel override |

All six short options are required. Unknown and duplicate options are rejected.

## Input and output

### Input CSV

The reader requires the first line to be a header with at least `-d` columns.
It then reads the first `-n` data rows and the first `-d` comma-separated values
from each row as finite `float` values. Missing or narrow headers, too few data
rows or columns, malformed values, and non-finite values fail with an error.
Extra columns and rows are allowed and ignored beyond the requested prefix.
The host reader transposes accepted values into dimension-major `[D, N]`
storage before device execution.

The tracked fixture begins as follows:

```csv
x,y
0.0,0.0
0.1,0.2
```

### Output files

A successful run writes three CSV files in the directory containing the path specified by `-o`:

- The requested label file: one headerless row per point, containing the `D` input values followed by a 1-based cluster label.
- `centers.csv`: `K` headerless rows, each containing `D` cluster-center values.
- `time.csv`: one header row and one result row containing timing and iteration information.

The output label is a one-based cluster identifier in the range `1..K`, not a ground-truth class label.

#### `time.csv` fields

| Field | Meaning |
| --- | --- |
| `read_data_time_ms` | Wall-clock time for reading and transposing the input CSV, in milliseconds. |
| `write_data_time_ms` | Wall-clock time for writing the label and center outputs, in milliseconds. |
| `fcm_time_ms_total` | Total FCM wall-clock time, including device allocation and data transfer, membership initialization, iterative computation, classification, and result copies. |
| `fcm_time_ms_avg_per_iter` | `fcm_time_ms_total / iteration_count`; a coarse per-iteration normalization of the total FCM time, not the iterative-loop time. |
| `iter_wall_time_ms_avg` | Average wall-clock time of the iterative FCM loop per executed iteration. |
| `first_iter_wall_time_ms` | Wall-clock time of the first iteration. |
| `other_iter_wall_time_ms_avg` | Average wall-clock time of the iterations after the first. |
| `init_membership_kernel_time_ms` | Membership-initialization kernel profiling time, in milliseconds. |
| `classify_kernel_time_ms` | Classification kernel profiling time, in milliseconds. |
| `update_membership_kernel_time_ms_avg` | Average membership-update kernel profiling time per executed iteration. |
| `update_cluster_centers_kernel_time_ms_avg` | Average cluster-center-update kernel profiling time per executed iteration. |
| `change_matrix_kernel_time_ms_avg` | Average center-change kernel profiling time per convergence check. |
| `iteration_count` | Number of FCM iterations actually executed. |
| `change_calls` | Number of center-change checks performed. |

`iter_wall_time_ms_avg` measures only the iterative loop: membership initialization occurs before this interval, while classification and final result copies occur after it.

Kernel profiling fields are meaningful only when the project is built with `FCM_ENABLE_PROFILING=ON`. When profiling is disabled, these fields are reported as `0`.

## Execution modes

### Fixed20

`fixed20` completes exactly 20 iterations. The first iteration has no center
change check, so a completed run performs 19 checks. A positive finite `-e`
value remains a required validated input but never causes early stopping.
Supplying `--max-iter` is optional; if supplied, its value must be 20.


### Native Dynamic

`native-dynamic` enables convergence-based execution. A positive maximum-iteration cap must be supplied explicitly.

For example:

```bash
mkdir -p build-cpu/example-dynamic

./build-cpu/cmeansOA \
  -i tests/fixtures/tiny.csv \
  -o build-cpu/example-dynamic/labels.csv \
  -c 2 -e <threshold> -n 4 -d 2 \
  --iteration-protocol native-dynamic \
  --max-iter 100 \
  --seed <seed>
```

For oneFCM, the convergence metric is the maximum absolute element-wise change between consecutive cluster-center matrices. Convergence is detected using the strict comparison

```text
change < threshold
```

and the first convergence check occurs after the second completed iteration. The reported iteration count includes the iteration that satisfies the stopping condition.

The stopping threshold is workload- and implementation-dependent. oneFCM therefore does not define a universal default epsilon for `native-dynamic`; users should select a threshold appropriate for the dataset, numerical scale, precision, and intended convergence behavior.

#### Dynamic thresholds used in the manuscript experiments

The end-to-end Dynamic experiments in the accompanying manuscript used the following calibrated stopping tolerances:

| Workload         | Seed |       oneFCM GPU |       oneFCM CPU |       Native-OMP |          scikit-fuzzy |      Native-CUDA |
| ---------------- | ---: | ---------------: | ---------------: | ---------------: | --------------------: | ---------------: |
| GID              |  700 | `2.91168708e-05` | `2.54213828e-05` |  `0.00288921618` |  `0.7965019822354785` |  `9.3340881e-05` |
| KDD Cup 1999     |  700 | `0.000238835826` | `0.000116765506` |  `0.00046005848` |  `1.4935728761491072` | `0.000746667443` |
| Electricity Load |  701 | `1.04382643e-05` | `1.04755172e-05` | `7.71135183e-06` | `0.10741036858604799` | `1.06915841e-05` |

All Dynamic manuscript runs used `max_iter = 100`. The seed was fixed per workload and shared across the compared backends: GID and KDD Cup 1999 used seed `700`, while Electricity Load used seed `701`.

These tolerances are calibrated for the stopping criterion used by each implementation. Their numerical magnitudes should therefore not be compared directly across implementations with different native convergence metrics. They are reported here to document the experimental settings used in the manuscript, not as generally recommended thresholds for other datasets or applications.


## Seed and fuzzifier

The default seed is 550. Membership initialization uses
`oneapi::dpl::minstd_rand`, draws a uniform value for each point/cluster entry,
and normalizes memberships per point. A fixed seed supports repeatable setup
within a controlled software/hardware configuration, but does not promise
bitwise-identical output across different compilers, runtimes, or backends.

The fuzzifier must be greater than 1, remain finite and greater than 1 after
conversion to the production single-precision representation, and yield a
finite membership exponent `q = 1 / (m - 1)`. With the default
`--m-power-policy auto`, exact public `m = 2` uses squaring for center weights
and the identity ratio for membership powers. Other accepted values use the
generic `pow` implementation. `--m-power-policy generic` forces the generic
path, including for `m = 2`; `--force-generic-m` is retained only as a
compatibility alias. The supported examples `m = 1.5`, `m = 2.5`, and `m = 3`
use the generic path.

## Work-group sizes

Without explicit options, the legacy selector starts from 512 work-items for a
CPU and 256 for a GPU for the center/membership kernels, and 256 for the
center-change kernel. The effective size remains subject to the selected
device's maximum work-group size, supported subgroup sizes, local-memory
capacity, and kernel resource needs.

`--work-group-size` is the convenient global request. The three per-kernel
options override it for their respective kernels and are intended for advanced
configuration. Explicit requests are strict: unsupported resource or launch
constraints fail instead of silently substituting an unrelated size.

## Testing

Configure with `BUILD_TESTING=ON`, build the project, and run the regression suite:

```bash
ctest --test-dir build-cpu --output-on-failure
```

See [tests/README.md](tests/README.md) for the test suite and additional test commands.

## Datasets used in the manuscript

### Synthetic scalability datasets

The Section 5.2 sample-size scalability study uses four synthetic inputs with
`N` from 100,000 to 100,000,000, `D = 4`, and runtime `K = 4`. The corresponding
generator scripts and their source attribution are available under
[`Datasets/scalability/`](Datasets/scalability/README.md). The large generated
CSV files are not included in this repository.

See [`Datasets/README.md`](Datasets/README.md) for an overview of the
real-world datasets used in the manuscript and their official sources.

## Reproducibility scope

This repository contains the production oneFCM implementation, public
regression tests, synthetic workload generators, and source links for the
real-world datasets used in the manuscript. Large third-party datasets are not
redistributed, and real-world preprocessing pipelines are not packaged here.

## License

Except for separately licensed third-party components, oneFCM is licensed under
the BSD 3-Clause License. See [LICENSE](LICENSE) for details. The historical
synthetic-data generators under [`Datasets/scalability/`](Datasets/scalability/README.md)
retain their applicable upstream CeCILL-C v1.0 license.
