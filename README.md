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

### Configuration Options
The following **parameters** can be set when initializing a `KSil` model:

| Parameter              | Type            | Default      | Description                                                                   |
|------------------------|-----------------|--------------|-------------------------------------------------------------------------------|
| `n_clusters`           | int             | `3`          | Number of clusters to form                                                    |
| `init_method`          | str             | `"random"`   | `"random"` or `"k-means++"` centroid initialization                           |
| `max_iter`             | int             | `100`        | Maximum number of iterations                                                  |
| `random_state`         | int             | `42`         | Random seed for reproducibility                                               |
| `silhouette_objective` | str             | `"macro"`    | Objective: `"micro"`, `"macro"`, or `"convex"`                                |
| `approximation`        | bool            | `False`      | Use silhouette approximation (faster for large datasets)                      |
| `sample_size`          | int / float     | `-1`         | Sample size: `-1` (full), fraction (0–1), or fixed count                      |
| `weighting`            | str             | `"power"`    | `"power"` or `"exponential"` weighting scheme                                 |
| `sensitivity`          | float / str     | `"auto"`     | Weight contrast: float or `"auto"` (grid search)                              |
| `alpha`                | float           | `0.5`        | Macro/micro tradeoff if using `"convex"` objective                            |
| `tol`                  | float           | `1e-4`       | Convergence threshold based on centroid movement                              |
| `n_jobs`               | int             | `-1`         | Number of parallel jobs (`-1` for all available cores)                        |

### Model Functions
Public **methods** for fitting, prediction, and analysis (all expecting array-like datasets of shape `[n_samples, n_features]`):

| Method                     | Description                                                       |
|----------------------------|-------------------------------------------------------------------|
| `fit(X)`                   | Fit the model on dataset `X`                                      |
| `predict(Y)`               | Predict cluster labels for data points in `Y` based on the fitted model|
| `transform(Z)`             | Transform data points in `Z` to distances to learned centroids     |
| `fit_predict(X)`           | Fit the model on `X` and return cluster labels                    |
| `fit_transform(X)`         | Fit the model and transform `X`                                   |

### Results
After fitting the model, the following **attributes** are available:

| Attribute           | Description                                                     |
|---------------------|-----------------------------------------------------------------|
| `labels_`           | Final cluster labels assigned to each data point                |
| `cluster_centers_`  | Learned centroids (as a pandas Series of coordinate lists)      |
| `n_iter_`           | Number of iterations until convergence                          |

## Quick Start
```python
from ksil import KSil
import numpy as np

# Example dataset
X = np.array([
    [1, 2], 
    [1, 4], 
    [1, 5], 
    [2, 8],  
    [3, 6],  
    [5, 7] 
])

# Initialize the K-Sil model
ksil = KSil(n_clusters=2, silhouette_objective="macro")

# Fit the model to the dataset
ksil.fit(X)

# Retrieve the cluster centroids (learned center points of each cluster)
centroids = ksil.cluster_centers_
# Use np.array(ksil.cluster_centers_) if you prefer the centroids in NumPy array format

# Retrieve the cluster labels (labels assigned to each data point in X)
labels = ksil.labels_

# Alternatively, fit the model and get cluster labels in one step
labels = KSil(n_clusters=2).fit_predict(X)



