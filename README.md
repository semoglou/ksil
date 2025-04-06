# K-Sil Clustering
<p align="center">
  <img src="demo/ksil_gif.gif" alt="K-Sil Demo" width="550"/>
</p>

## Overview

## Installation

You can install **K-Sil** from PyPI:

```bash
pip install ksil
```

or directly from the GitHub repository:

```bash
pip install git+https://github.com/semoglou/ksil.git
```

## How to Use

### Parameters

The following parameters can be set when initializing a `KSil` model:

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

### Methods 

Public methods for fitting, prediction, and analysis:

| Method                     | Description                                                       |
|----------------------------|-------------------------------------------------------------------|
| `fit(X)`                   | Fit the model on dataset `X`                                      |
| `predict(Y)`               | Assign cluster labels to new data points based on the fitted model|
| `transform(X)`             | Return a distance matrix to centroids                             |
| `fit_predict(X)`           | Fit and predict in one step                                       |
| `fit_transform(X)`         | Fit and transform in one step                                     |


### Attributes

Available attributes after calling `.fit(X)`:

| Attribute           | Description                                                     |
|---------------------|-----------------------------------------------------------------|
| `labels_`           | Final cluster labels assigned to each data point                |
| `cluster_centers_`  | Learned centroids (as a pandas Series of coordinate lists)      |
| `n_iter_`           | Number of iterations until convergence                          |
