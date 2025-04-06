# K-Sil Clustering
<p align="center">
  <img src="demo/ksil_gif.gif" alt="K-Sil Demo" width="550"/>
</p>

## Parameters

| Parameter              | Type            | Default      | Description                                                                 |
|------------------------|-----------------|--------------|-----------------------------------------------------------------------------|
| `n_clusters`           | `int`           | `3`          | Number of clusters to form                                                  |
| `init_method`          | `str`           | `'random'`   | Centroid initialization method: `'random'` or `'k-means++'`                |
| `max_iter`             | `int`           | `100`        | Maximum number of iterations                                                |
| `random_state`         | `int`           | `42`         | Random seed for reproducibility                                             |
| `silhouette_objective`| `str`           | `'macro'`    | Scoring objective: `'micro'`, `'macro'`, or `'convex'`                      |
| `approximation`        | `bool`          | `False`      | Use silhouette approximation (faster on large datasets)                     |
| `sample_size`          | `int` / `float` | `-1`         | Sampling size: `-1` = full data, fraction (0–1), or count (>1)             |
| `weighting`            | `str`           | `'power'`    | Weighting scheme: `'power'` or `'exponential'`                              |
| `sensitivity`          | `float` / `str` | `'auto'`     | Contrast of weighting: float or `'auto'` for grid search                    |
| `alpha`                | `float`         | `0.5`        | Balance of micro/macro when `silhouette_objective='convex'`                |
| `tol`                  | `float`         | `1e-4`       | Convergence threshold (average centroid movement)                          |
| `n_jobs`               | `int`           | `-1`         | Number of parallel jobs (`-1` = all cores)                                  |
