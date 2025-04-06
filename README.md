# K-Sil Clustering
<p align="center">
  <img src="demo/ksil_gif.gif" alt="K-Sil Demo" width="550"/>
</p>

## Installation

You can install **K-Sil** from PyPI:

```bash
pip install ksil
```

or directly from the GitHub Repository:

```bash
pip install git+https://github.com/semoglou/ksil.git
```

## Usage

## Parameters

| Parameter              | Type            | Default      | Description                                                                   |
|------------------------|-----------------|--------------|-------------------------------------------------------------------------------|
| `n_clusters`           | int             | `3`          | Number of clusters to form                                                    |
| `init_method`          | str             | `'random'`   | `'random'` or `'k-means++'` centroid initialization                           |
| `max_iter`             | int             | `100`        | Maximum number of iterations                                                  |
| `random_state`         | int             | `42`         | Random seed for reproducibility                                               |
| `silhouette_objective` | str             | `'macro'`    | Objective: `'micro'`, `'macro'`, or `'convex'`                                |
| `approximation`        | bool            | `False`      | Use silhouette approximation (faster for large datasets)                      |
| `sample_size`          | int / float     | `-1`         | Sample size: `-1` (full), fraction (0–1), or fixed count                      |
| `weighting`            | str             | `'power'`    | `'power'` or `'exponential'` weighting scheme                                 |
| `sensitivity`          | float / str     | `'auto'`     | Weight contrast: float or `'auto'` (grid search)                              |
| `alpha`                | float           | `0.5`        | Macro/micro tradeoff if using `'convex'` objective                            |
| `tol`                  | float           | `1e-4`       | Convergence threshold based on centroid movement                              |
| `n_jobs`               | int             | `-1`         | Number of parallel jobs (`-1` = all cores)                                    |
