# K-Sil Clustering
K-Sil is a centroid-based clustering algorithm designed to improve the quality of *k*-means partitions by integrating silhouette scores directly into the centroid update process.
Unlike *k*-means, which treats all points equally, K-Sil dynamically weights data points in each iteration based on their silhouette scores through self-tuning, per cluster weighting schemes,
effectively increasing the influence of well-clustered, high-confidence regions on centroid updates, while suppressing the impact of outliers and noisy or unreliable instances. 
As a result, K-Sil reduces sensitivity to poor centroid initialization and yields clustering partitions that are both more accurate in capturing intrinsic data patterns and more resilient to noise, outliers, overlapping groups, and cluster imbalances.
It supports silhouette-based objectives, such as *Macro*-averaged Silhouette (cluster-level), *Micro*-averaged (point-level),
or their combination, allowing flexible emphasis during clustering, while maintaining scalability through objective-aware sampling and efficient silhouette approximations.

<p align="center">
  <img src="demo/ksil_g.gif" alt="K-Sil Demo" width="500"/>
</p>

## Key Components


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
| `silhouette_objective` | str             | `"macro"`    | Objective: `"macro"`, `"micro"` (or `"convex"`)                               |
| `approximation`        | bool            | `False`      | Use silhouette approximation (faster for large datasets)                      |
| `sample_size`          | int / float     | `-1`         | Sample size: `-1` (full), fraction (0–1), or fixed count                      |
| `weighting`            | str             | `"power"`    | `"power"` or `"exponential"` weighting scheme                                 |
| `sensitivity`          | float / str     | `"auto"`     | Weight contrast: float or `"auto"` (grid search)                              |
| `alpha`                | float           | `0.5`        | Macro/micro tradeoff (used if the objective=`"convex"`)                       |
| `tol`                  | float           | `1e-4`       | Convergence threshold based on centroid movement                              |
| `n_jobs`               | int             | `-1`         | Number of parallel jobs (`-1` for all available cores)                        |

### Model Functions
Public **methods** for fitting, prediction, and analysis (expecting array-like datasets `[n_samples, n_features]`):

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
#### Fit K-Sil and Retrieve Cluster Labels & Centroids
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

# Get the number of iterations the algorithm took to converge
n_iter = ksil.n_iter_

# Retrieve the cluster labels (labels assigned to each data point in X)
labels = ksil.labels_

# Alternatively, fit the model and get cluster labels in one step
# labels = KSil(n_clusters=2).fit_predict(X)

# Retrieve the cluster centroids (learned center points of each cluster)
centroids = ksil.cluster_centers_
# Use np.array(ksil.cluster_centers_) if you prefer the centroids in NumPy array format

print(f"Iterations: {n_iter}")
print(f"Cluster Labels: {labels}")
print(f"Cluster Centroids:\n {centroids}")
```
Output:
```
Iterations: 2
Cluster Centroids: [0 0 0 1 1 1]
Cluster Labels:
0    [1.5, 4.25]
1    [3.5, 7.5]
```

#### Predict Labels and Transform Data
```python
# Assuming the K-Sil model has already been fitted in the previous cell

# New data for prediction and transformation
new_data = np.array([
    [2, 3], 
    [4, 5]
])

# Predict the cluster labels for the new data points based on the fitted model
pred_labels = ksil.predict(new_data)

# Transform the new data points into distances to the previously learned centroids
transformation = ksil.transform(new_data)

print(f"Predicted Labels for New Data: {pred_labels}")
print(f"Transformed New Data (Distances to Centroids):\n {transformation}")
```
Output:
```
Predicted Labels for New Data: [0 1]
Transformed New Data (Distances to Centroids):
[[1.3462912  4.74341649]
 [2.61007663 2.54950976]]
```
For a detailed example exploring additional aspects such as sampling and approximations, check out this [notebook](./analysis/functionality.ipynb).

## Clustering Performance

K-Sil demonstrates consistent and statistically significant improvements over *k*-means in clustering quality,
as measured by silhouette scores (both *macro* and *micro*) across synthetic and real world datasets.
Higher Normalized Mutual Information (NMI) scores are also observed on synthetic datasets, where ground truth labels allow for such evaluation. 
K-Sil’s most substantial improvements occur for the *Macro*-averaged Silhouette score when also configured to prioritize this objective.
By design, its cluster-centric weighting strategy, which emphasizes well-clustered instances and de-emphasizes uncertain ones within each cluster, 
naturally aligns with the macro objective’s focus on cluster-level quality.
This makes K-Sil particularly effective for detecting subtle patterns (e.g., in biological data), preserving minority groups (e.g., in medical diagnostics or fraud detection),
and ensuring fair representation in heterogeneous data (e.g., in customer or demographic segmentation).

<p align="center">
  <img src="demo/synthtruelabels.png" alt="Ground Truth Clusters" width="300"/><br/>
  <sub><em>
    Synthetic dataset with varied cluster shapes and densities.
  </em></sub>
</p>

<p align="center">
  <img src="demo/kmvsksclusters.png" alt="KMeans vs KSil Clusters" width="600"/><br/>
  <sub><em>
    Cluster assignments by K-Means (left) and K-Sil (right), both initialized with the same centroids (final centroids: ❌).<br/>
    For K-Sil, <code>silhouette_objective="macro"</code>, <code>sensitivity=1</code> (no <code>"auto"</code> grid search applied).
  </em></sub>
</p>

For a comprehensive evaluation of K-Sil on synthetic and real-world datasets, including statistical comparisons, performance benchmarks,
and detailed results on *macro*-/*micro*-averaged silhouette and NMI, see the notebooks in the [`analysis/`](analysis/) folder, 
particularly [ksil_performance.ipynb](./analysis/ksil_performance.ipynb).

#

<p align="center"><sub>K-Sil Clustering Algorithm · v0.1.0 · Last updated: 04/2025 · MIT License</sub></p>
