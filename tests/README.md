# Public regression suite

The default suite is intentionally small, self-contained, and suitable for a
fresh CPU-oriented SYCL build. It uses only the standard Python library and the
tracked `fixtures/tiny.csv`; no accelerator, network access, paper dataset, or
private repository is required.

Run all default tests:

```bash
ctest --test-dir <build-dir> --output-on-failure
```

Run one layer:

```bash
ctest --test-dir <build-dir> -L static --output-on-failure
ctest --test-dir <build-dir> -L cpu --output-on-failure
ctest --test-dir <build-dir> -L smoke --output-on-failure
```

## Test manifest

| Test | Layer | Contract | Executable/fixture |
| --- | --- | --- | --- |
| `public_cmeans_header_compile_test` | static | `cmeans.h` compiles independently | none |
| `public_execution_mode_header_compile_test` | static | `execution_mode.h` compiles independently | none |
| `public_m_power_header_compile_test` | static | `m_power_config.h` compiles independently | none |
| `public_work_group_header_compile_test` | static | `work_group_config.h` compiles independently | none |
| `public_surface_contract_test` | static | retained public API and removed internal surface | source files only |
| `public_tree_contract_test` | static | exact public source tree, required files, and private-path hygiene | source tree only |
| `public_comment_contract_test` | static | English-only C/C++, CMake, and Python comments/docstrings | source tree only |
| `container_contract_test` | static | Platform A source identity gates and bounded performance-validation wording | Dockerfile and container guide |
| `dataset_generator_contract_test` | static | synthetic dataset documentation and generator source parameters | source files and public dataset guide |
| `work_group_source_contract_test` | static | WGS ownership, K=32/33 split, Scheme-B 2D/1024-B/two-pass structure | source files only |
| `membership_init_timing_boundary_contract_test` | static | initialization timing event ordering | source file only |
| `m_power_source_contract_test` | static | one host dispatch and no private fuzzifier controls | source files only |
| `execution_mode_config_test` | cpu | Fixed20/Dynamic configuration and rejection rules | none |
| `dynamic_control_test` | cpu | strict Dynamic stop and Fixed20 iteration semantics | none |
| `normalization_layout_test` | cpu | cluster-major normalization and tolerance boundary | none |
| `normalization_nonfinite_test` | cpu | nonfinite and invalid-input fail-closed behavior | none |
| `work_group_selection_test` | cpu | CPU/GPU defaults, explicit 128/256/512, device/local-memory boundaries, tile metadata | synthetic capabilities only |
| `m_power_config_test` | cpu | fuzzifier parsing and specialized/generic dispatch | none |
| `execution_mode_cli_contract_test` | cpu | Fixed20/Dynamic CLI execution and rejection | `cmeansOA`, `tiny.csv` |
| `m_power_public_regression_test` | cpu | public fuzzifier matrix and m=2 equivalence | `cmeansOA`, `tiny.csv` |
| `production_api_smoke_test` | cpu, smoke | direct production API Fixed20 behavior | SYCL CPU selector |
| `production_release_boundary_test` | cpu, smoke | ordinary outputs, 14-column timing schema, public build boundary | `cmeansOA`, `tiny.csv` |
| `public_cli_contract_test` | cpu, smoke | exact public CLI vocabulary, defaults, alias, and tiny execution | `cmeansOA`, `tiny.csv` |
| `readme_contract_test` | static | public sections, exact CLI vocabulary, links, path hygiene, and bounded release claims | `README.md`, source file |

Device-specific backend checks are intentionally not registered in default
CTest. Release qualification may run the same public suite in an explicitly
configured Intel Arc, NVIDIA, AMD, or FPGA build; lack of such a device does not
change the default suite result.
