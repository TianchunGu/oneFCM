# Platform A Docker environment

## Purpose and reproducibility boundary

This container provides a validated Platform A reference environment for
building and running oneFCM.
The container captures the principal user-space software environment, while
the physical GPU, host operating-system kernel, NVIDIA driver, and other
hardware-dependent execution characteristics remain provided by the host
system.

At the time this release was prepared, the official Codeplay 2025.0 installer
was no longer available from its original distribution endpoint. The image
therefore builds the CUDA adapter from pinned official
[Unified Runtime](https://github.com/oneapi-src/unified-runtime) v0.10.0 and
[Unified Memory Framework](https://github.com/oneapi-src/unified-memory-framework)
source revisions validated with the oneAPI 2025.0 toolchain. The resulting
adapter has the expected `libur_adapter_cuda.so.0` SONAME and was functionally
validated on Platform A.

## Prerequisites

- Docker with BuildKit and named build-context support.
- For GPU execution, NVIDIA Container Toolkit and a compatible NVIDIA driver on
  the host.
- The official Intel offline installer named
  `intel-oneapi-base-toolkit-2025.0.0.885_offline.sh`.
- Local clean checkouts of the official Unified Runtime and Unified Memory
  Framework repositories at the exact commits below.

The `.885` installer filename is correct; the installer's internal Base Toolkit
product version is `2025.0.0+884`. Download it from Intel's official archive and
verify it before use:

```bash
sha256sum intel-oneapi-base-toolkit-2025.0.0.885_offline.sh
```

Expected SHA-256:

```text
b379986e96a70d4d8c64b46eaecec73f14916ba39438ff167cb5f4f5fffc66de
```

The Dockerfile installs only the `intel.oneapi.lin.dpcpp-cpp-compiler`
component required by oneFCM. The installer still records the exact Base
Toolkit product identity, while unrelated tools such as VTune and Advisor are
not added to the image.

## Prepare the official adapter sources

```bash
git clone https://github.com/oneapi-src/unified-runtime.git
git -C unified-runtime checkout --detach \
  d040a8face402e5ecc7c99cdb4e976832592ce99

git clone https://github.com/oneapi-src/unified-memory-framework.git
git -C unified-memory-framework checkout --detach \
  59c4150b7120a7af5b3c8eb2d9b8bbb5d2e96aa3

test "$(git -C unified-runtime rev-parse HEAD)" = \
  d040a8face402e5ecc7c99cdb4e976832592ce99
test -z "$(git -C unified-runtime status --short)"
test "$(git -C unified-memory-framework rev-parse HEAD)" = \
  59c4150b7120a7af5b3c8eb2d9b8bbb5d2e96aa3
test -z "$(git -C unified-memory-framework status --short)"
```

No source repository is cloned during `docker build`; both source trees are
provided as named local contexts.

The `UR_COMMIT` and `UMF_COMMIT` manifest entries record the pinned revisions
above. Their source identity is enforced by the exact-commit and clean-checkout
prerequisites before the build; the Dockerfile does not inspect Git metadata in
the named contexts.

## Build the image

Run from the oneFCM repository root. The Intel context is the directory that
contains the offline installer, not the installer file itself.

```bash
test -z "$(git status --short)"
ONEFCM_SOURCE_COMMIT="$(git rev-parse HEAD)"

DOCKER_BUILDKIT=1 docker build \
  --file containers/Dockerfile.platform-a \
  --build-arg ONEFCM_SOURCE_COMMIT="${ONEFCM_SOURCE_COMMIT}" \
  --build-context intel_basekit_installer_dir=/path/to/intel-installer-directory \
  --build-context ur_source=/path/to/unified-runtime \
  --build-context umf_source=/path/to/unified-memory-framework \
  --tag onefcm:platform-a \
  .
```

`ONEFCM_SOURCE_COMMIT` records the exact source commit supplied at build time
in both the OCI image revision label and the container environment manifest.
The command above derives it from the current repository checkout; no local
filesystem path is embedded in the image metadata.

The final image contains neither the 2.2 GiB Intel installer nor the adapter
source/build tree. Record both immutable image identifiers after the build:

```bash
docker image inspect onefcm:platform-a \
  --format 'image_id={{.Id}} repo_digests={{json .RepoDigests}} size={{.Size}}'
```

Before the image is pushed, `RepoDigests` may be empty; the local image ID is
still immutable. A registry digest must be recorded after any publication.

## Verify the environment

```bash
docker run --rm onefcm:platform-a bash -lc '
  source /opt/intel/oneapi/setvars.sh >/dev/null
  cat /etc/os-release
  cat /opt/onefcm-environment/manifest.txt
  cat /opt/onefcm-environment/cuda-adapter.sha256
  /opt/intel/oneapi/installer/installer --list-products
  icpx --version
  nvcc --version
  cmake --version
  sycl-ls
'
```

GPU discovery requires a GPU-enabled run:

```bash
docker run --rm --gpus all onefcm:platform-a bash -lc '
  source /opt/intel/oneapi/setvars.sh >/dev/null
  nvidia-smi --query-gpu=name,driver_version,compute_cap \
    --format=csv,noheader
  sycl-ls
'
```

## CPU build and functional smoke

```bash
docker run --rm onefcm:platform-a bash -lc '
  source /opt/intel/oneapi/setvars.sh >/dev/null
  cmake -S . -B build-cpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DHARDWARE_TARGET=CPU \
    -DFCM_ENABLE_FAST_MATH=ON \
    -DFCM_ENABLE_PROFILING=OFF \
    -DBUILD_TESTING=ON
  cmake --build build-cpu --parallel
  ctest --test-dir build-cpu --output-on-failure
'
```

## NVIDIA GPU build and functional smoke

```bash
docker run --rm --gpus all onefcm:platform-a bash -lc '
  source /opt/intel/oneapi/setvars.sh >/dev/null
  cmake -S . -B build-gpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DHARDWARE_TARGET=GPU \
    -DCUDA_GPU_ARCH=sm_89 \
    -DFCM_ENABLE_FAST_MATH=ON \
    -DFCM_ENABLE_PROFILING=OFF \
    -DBUILD_TESTING=ON
  cmake --build build-gpu --parallel
  ctest --test-dir build-gpu -L static --output-on-failure
  mkdir -p build-gpu/functional-smoke
  ONEAPI_DEVICE_SELECTOR=cuda:gpu ./build-gpu/cmeansOA \
    -i tests/fixtures/tiny.csv \
    -o build-gpu/functional-smoke/labels.csv \
    -c 2 -e 0.001 -n 4 -d 2
'
```

The commands above are build, test, and tiny-fixture functional checks. They are
not performance benchmarks and must not be used to infer container/native
performance equivalence.

## Performance validation

The container environment was separately evaluated for performance consistency
against the previously validated non-containerized oneFCM Section 5.2 results.
The validation covered 14 configurations on both CPU and GPU, with three
independent processes per configuration and device class, for 84 container
observations in total. The overall median absolute difference was approximately
0.31%, all GPU configuration differences were below 0.87%, and no systematic
slowdown was observed across the sample-size, dimensionality, or cluster-count
sweeps.

This is a performance-consistency validation against the previously validated
non-containerized Section 5.2 results, not a paired measurement of strict
containerization overhead.
