# Synthetic scalability generators

These scripts generate the synthetic inputs for the sample-size scalability
configurations described in Section 5.2 of the oneFCM paper.

## Source and modifications

The original source is CentraleSupélec's
[`cpu-gpu-kmeans`](https://gitlab-research.centralesupelec.fr/Stephane.Vialle/cpu-gpu-kmeans)
[`Synthetic_Data_Generator.py`](https://gitlab-research.centralesupelec.fr/Stephane.Vialle/cpu-gpu-kmeans/-/blob/6f1ed5f59fbd0f9e645c189aaa8cfabd532d9347/Synthetic_Data_Generator.py),
at commit `6f1ed5f59fbd0f9e645c189aaa8cfabd532d9347` (Git blob
`84beaa57d024ab058a122fc7191f1e044bc6ec38`). These files are modified
variants of that generator.

Modified by Tianchun Gu on 2025-04-07. The modifications adjust the number of
points per cluster to produce each total sample count, change the output file
from `.txt` to `.csv`, and change the delimiter from tab to comma. The
generation distribution and sampling algorithm are unchanged.

## Configurations

| Variant | Public script | N |
|---|---|---:|
| 105 | `generate_scalability_n1e5.py` | 100,000 |
| 106 | `generate_scalability_n1e6.py` | 1,000,000 |
| 107 | `generate_scalability_n1e7.py` | 10,000,000 |
| 108 | `generate_scalability_n1e8.py` | 100,000,000 |

All four configurations use `D = 4`. The generator creates four clusters, and
the paper experiments use runtime `K = 4`; these are distinct concepts that
happen to have the same value here.

## Generation procedure

Each cluster has radius 9 and one of the following fixed centers:

```text
cluster 0 = (40, 40, 60, 60)
cluster 1 = (40, 60, 60, 40)
cluster 2 = (60, 40, 40, 60)
cluster 3 = (60, 60, 40, 40)
```

Two independent two-dimensional uniform-disk samples form each four-dimensional
point. Coordinates are stored as `numpy.float32`. Points are written in
cluster-major order without shuffling. Each generated feature CSV includes a
header row so that it follows oneFCM's public input format.

## Reproducibility note

The scripts use Python's global `random` generator without an explicit seed, so
generated values may differ between runs. The large generated CSV files are not
included in this repository.

See [`../SYNTHETIC_DATASETS.md`](../SYNTHETIC_DATASETS.md) for the generation
procedures and parameters used by these datasets.

## License

These generator variants retain their applicable upstream
CeCILL-C v1.0 license. See [LICENSE-CeCILL-C](LICENSE-CeCILL-C).
