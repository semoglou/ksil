# K-Sil Clustering
<p align="center">
  <img src="demo/ksil_gif.gif" alt="K-Sil Demo" width="550"/>
</p>

## Parameters

## Parameters

Parameter      |        Type       |     Default  |    Description  
---------------------- --------------- ------------ -----------------------------------------------------------------------------------------  
`n_clusters`           `int`           `3`          Number of clusters to form  
`init_method`          `str`           `'random'`   Centroid initialization: `'random'` or `'k-means++'`  
`max_iter`             `int`           `100`        Maximum number of iterations  
`random_state`         `int`           `42`         Random seed for reproducibility  
`silhouette_objective` `str`           `'macro'`    Scoring target: `'micro'`, `'macro'`, or `'convex'`  
`approximation`        `bool`          `False`      Use silhouette approximation (faster on large data)  
`sample_size`          `int/float`     `-1`         `-1` = full data, else fraction (0–1) or absolute count  
`weighting`            `str`           `'power'`    `'power'` or `'exponential'` weighting of points  
`sensitivity`          `float/str`     `'auto'`     Weight contrast: float or `'auto'` (grid search)  
`alpha`                `float`         `0.5`        Convex blend factor (if objective is `'convex'`)  
`tol`                  `float`         `1e-4`       Centroid shift threshold for convergence  
`n_jobs`               `int`           `-1`         Number of parallel jobs (`-1` = all cores)  

