import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import cdist

def point_silhouettes(X, labels, approximation=False, centers=None):
    """
    Compute silhouette scores for each point in the dataset, with approximate computation option.

    Parameters:
       - X: array-like, shape (n_samples, n_features)
             The input data.
       - labels: array-like, shape (n_samples,)
                  The cluster labels for each sample.
       - approximation: bool, default=False
                        - If False, computes exact silhouette scores using sklearn's silhouette_samples.
                        - If True, uses an approximate silhouette computation based on centroids and sum of squares.
       - centers: pd.Series, default=None (optional: used only for approximation option)
                  A dictionary mapping cluster labels to their precomputed centroids.
                  - If provided/precomputed, these centroids are used for silhouette approximation.
                  - If None, centroids are computed as the mean of points in each cluster.

    Returns: np.ndarray, shape (n_samples,)
             The silhouette score for each point in X.
    """
    labels = np.asarray(labels)

    # Exact silhouette computation (approximation=False)
    if not approximation:
        return silhouette_samples(X, labels=labels)

    # Approximation (approximation=True)
    # Ensure each cluster has a centroid

    # Compute approximate centers (average of points) if centers not provided
    if centers is None:
        unique_labels = np.unique(labels)
        centers = pd.Series({cl: np.mean(X[labels == cl], axis=0) for cl in unique_labels})

    if hasattr(centers, 'index'):
        unique_labels = np.unique(labels)
        for cl in unique_labels:
            if cl not in centers.index:
                # If centroid not present, compute it as the mean of its cluster's points
                centers.at[cl] = np.mean(X[labels == cl], axis=0)
        centers = centers.sort_index()
        clusters = centers.index.to_numpy()
        mapping = {c: i for i, c in enumerate(clusters)}
        inv = np.array([mapping[label] for label in labels])
    else:
        # If centers a simple array-like, assume labels 0-based
        clusters = np.arange(len(centers))
        inv = labels

    k = len(clusters)
    n = X.shape[0]

    # Convert centers to a NumPy array
    centers_arr = np.array(centers.tolist())

    # Compute differences and squared distances between points and their assigned centroids
    diffs = X - centers_arr[inv]         # Shape: (n, d)
    dists_sq = np.sum(diffs**2, axis=1)    # Shape: (n,)

    # Compute per-cluster counts and within-cluster sum-of-squares (SS)
    counts = np.bincount(inv, minlength=k)
    SS_arr = np.bincount(inv, weights=dists_sq, minlength=k)

    # Compute full distance matrix from every point to every centroid
    D = cdist(X, centers_arr, metric='euclidean')
    
    # Extract for each point the distance to its own centroid
    D_diag = D[np.arange(n), inv]

    # For each point, get the count and SS corresponding to its cluster
    count_vec = counts[inv]
    ss_vec = SS_arr[inv]

    # Compute approximate intra-cluster distances a(x_i).
    a_vals = np.where(
        count_vec > 1,
        np.sqrt((count_vec * (D_diag ** 2) + ss_vec) / (count_vec - 1)),
        D_diag
    )

    # Compute candidate distances for b(x_i)
    # candidate_matrix[i, j] = sqrt( D[i, j]^2 + (SS_arr[j] / counts[j]) )
    cluster_term = SS_arr / np.maximum(counts, 1e-6)  # per-cluster term
    candidate_matrix = np.sqrt(D**2 + cluster_term.reshape(1, k))
    candidate_matrix[np.arange(n), inv] = np.inf  # Exclude own cluster

    # b(x_i) is the minimum candidate value
    b_vals = np.min(candidate_matrix, axis=1)
    
    # Maximum of a(x_i), b(x_i)
    max_ab = np.maximum(np.maximum(a_vals, b_vals), 1e-6) # Avoid zero division

    # Compute approximate silhouette scores for each point.
    silhouette_scores = (b_vals - a_vals) / max_ab
    
    return silhouette_scores

def micro_silhouette(X, labels, approximation=False, centers=None, sample_size=-1, seed=42):
    """
    Compute the micro-average silhouette score, the overall mean silhouette score across all points.

    Parameters:
       - X: array-like, shape (n_samples, n_features)
            The input dataset.
       - labels: array-like, shape (n_samples,)
                 Cluster labels for each sample.
       - approximation: bool, default=False
                        - If False, computes exact silhouette scores using sklearn's silhouette_samples.
                        - If True, uses an approximate silhouette computation based on centroids and sum of squares.
       - centers: dict or pd.Series, default=None (optional)
                  A dictionary mapping cluster labels to their precomputed centroids.
                  - If provided, these centroids are used for silhouette approximation.
                  - If None, centroids are computed as the mean of points in each cluster.
       - sample_size: int or float, default=-1
                      Number of points to sample uniformly (for micro-average silhouette score)
                      - If sample_size == -1: no sampling.
                      - If 0 <= sample_size <= 1: sample that fraction of the data.
                      - If sample_size > 1: sample that exact number of points.
       - seed: int, default=42
               Random seed for reproducibility, used when sampling is enabled (sample_size!=-1).

    Returns: float
             The micro-average silhouette score.
    """
    # If sampling is not enabled (sample_size=-1)
    if sample_size == -1:
        point_scores = point_silhouettes(X, labels, approximation=approximation, centers=centers)
    else: # If sampling is enabled (sample_size != -1)
        # Perform uniform sampling, as this sampling strategy aligns with micro-average silhouette score nature
        np.random.seed(seed)
        n_samples=len(X)
        labels = np.array(labels)
        if 0 <= sample_size <= 1: # sample_size as a fraction of the dataset
            size=int(np.ceil(sample_size*n_samples))
        elif sample_size > 1: # sample_size as the exact number of points to sample
            size=int(sample_size)
        else:
            raise ValueError(
                '"sample_size" must be -1, a fraction (between 0 and 1), or an exact number of data points'
            )
        if size == 0:
            raise ValueError("Sample size computed as 0. Increase sample_size or adjust the parameters.")

        indices=np.random.choice(n_samples, size=size, replace=False)

        num_sampled=len(indices)
        num_clusters=len(np.unique(labels[indices]))
        if num_sampled <= num_clusters:
            raise ValueError(
                  f"Uniform Sampling produced only {num_sampled} samples for {num_clusters} clusters. "
                  "Increase sample_size or adjust the sampling strategy.")

        # Silhouette scores computation on the sampled dataset
        point_scores = point_silhouettes(X[indices], labels[indices], approximation=approximation, centers=centers)

    return np.mean(point_scores)  # Simple mean across all points (micro)

def macro_silhouette(X, labels, approximation=False, centers=None, sample_size=-1, seed=42):
    """
    Compute the macro-average silhouette score, the mean of per-cluster silhouette scores.

    Parameters:
       - X: array-like, shape (n_samples, n_features)
            The input dataset.
       - labels: array-like, shape (n_samples,)
                 Cluster labels for each sample.
       - approximation: bool, default=False
                        - If False, computes exact silhouette scores using sklearn's silhouette_samples.
                        - If True, uses an approximate silhouette computation based on centroids and sum of squares.
       - centers: dict or pd.Series, default=None (optional)
                  A dictionary mapping cluster labels to their precomputed centroids.
                  - If provided, these centroids are used for silhouette approximation.
                  - If None, centroids are computed as the mean of points in each cluster.
       - sample_size: int or float, default=-1
                      Number of points to sample in a balanced way (for macro-average silhouette score)
                      - If sample_size == -1: no sampling.
                      - If 0 <= sample_size <= 1: sample that fraction of the data.
                      - If sample_size > 1: sample that exact number of points.
       - seed: int, default=42
               Random seed for reproducibility, used when sampling is enabled (sample_size!=-1).

    Returns:  float
              The macro-average silhouette score.
    """
    # sampling is not enabled (sample_size=-1)
    if sample_size == -1:
        point_scores = point_silhouettes(X, labels, approximation=approximation, centers=centers)
    else: # If sampling is enabled (sample_size != -1)
        # Perform balanced sampling, as this sampling strategy aligns with macro-average silhouette score nature
        np.random.seed(seed)
        n_samples=len(X)
        labels = np.array(labels)
        if 0 <= sample_size <= 1: # sample_size as a fraction of the dataset
            size=int(np.ceil(sample_size*n_samples))
        elif sample_size > 1: # sample_size as the exact number of points to sample
            size=int(sample_size)
        else:
            raise ValueError(
                '"sample_size" must be -1, a fraction (between 0 and 1), or an exact number of data points'
            )
        if size == 0:
            raise ValueError("Sample size computed as 0. Increase sample_size or adjust the parameters.")

        unique_clusters = np.unique(labels)
        samples_per_cluster = max(1, size // len(unique_clusters))
        indices_list = []
        for cluster_label in unique_clusters:
            cluster_indices = np.where(labels == cluster_label)[0]
            if len(cluster_indices) == 0:
                continue
            chosen_size = min(len(cluster_indices), samples_per_cluster)
            if chosen_size == 0:
                continue
            chosen_indices = np.random.choice(cluster_indices, size=chosen_size, replace=False)
            indices_list.append(chosen_indices)
        if indices_list:
            indices = np.concatenate(indices_list)
        else:
            indices = np.array([], dtype=int)
        if indices.shape[0] <= len(unique_clusters):
            raise ValueError(
                f"Balanced Sampling produced only {indices.shape[0]} samples for {len(unique_clusters)} clusters. "
                "Increase sample_size or adjust the sampling strategy.")

        # Retain the sampled dataset and labels array
        X = X[indices]
        labels = labels[indices]
        point_scores = point_silhouettes(X, labels, approximation=approximation, centers=centers)

    # Compute per-cluster mean silhouette scores
    unique_clusters = np.unique(labels)
    cluster_means = []

    for cluster in unique_clusters:
        cluster_scores = point_scores[labels == cluster]  # Get silhouette scores for the cluster
        cluster_means.append(cluster_scores.mean())  # Compute mean silhouette for this cluster

    return np.mean(cluster_means)  # Mean across all clusters (macro)