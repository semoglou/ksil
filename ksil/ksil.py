import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, pairwise_distances
from scipy.spatial.distance import cdist
from collections import Counter
from joblib import Parallel, delayed

class KSil:
    """
    K-Sil Clustering Algorithm
    (Silhouette-Guided Weighted K-Means Clustering)

    Parameters
    ----------

    - n_clusters: int, default=3
          The number of clusters to form.

    - init_method: str, default='random'
          Method for initializing centroids, options are:
          - 'random': One K-Means iteration with random initialization.
          - 'k-means++': One K-Means iteration with k-means++ initialization.

    - max_iter: int, default=100
          Maximum number of iterations for the algorithm.

    - random_state: int, default=42
          Random seed for reproducibility.

    - silhouette_objective: str, default='macro'
          Silhouette aggregation to emphasize during clustering,
          by selecting appropriare sampling (if sample_size!=-1),
          weight sensitivity in grid search (if sensitivity='auto')
          and tracking the best score.
          Options:
          - 'micro': Micro-averaged silhouette score, treating each point equally,
             best for datasets where individual point clustering performance is critical.
          - 'macro': Macro-averaged silhouette score, emphasizing cluster-level performance,
             best when well-separated clusters are desired.
          - 'convex': Combination of micro and macro scores, (combined using the 'alpha' parameter)
             best when both individual point performance and overall cluster separation are important.

    - approximation: bool, default=False
          - If True, the algorithm uses silhouette approximations across iterations:
                  a(x_i) = sqrt( [n_c * ||x_i - μ_c||^2 + SS_c] / (n_c - 1) )
                  b(x_i) = min_{d!=c} { sqrt( ||x_i - μ_d||^2 + (SS_d / n_d) ) }
                  s(x_i) = [b(x_i) - a(x_i)] / max{a(x_i), b(x_i)}
                  (Faster then "exact" silhouette scores for large datasets).
          - If False, computes the exact silhouette scores using the sklearn’s silhouette_samples
                  (Faster than "approximation" strategy for small datasets).

    - sample_size: int or float, default=-1
          Number of samples to use for silhouette scores computation, weights assignment and centroid updates.
          If 'silhouette_objective'= 'macro' performs 'Balanced' sampling, else performs 'Uniform' sampling.
          - If -1, use the full dataset (no sampling) (faster for small-medium datasets).
          - If between 0 and 1, use that fraction of the dataset (faster for medium-large datasets).
          - If >1, use that exact number of data points.

    - weighting: str, default='power'
          Weighting scheme for assigning influence to data points based on their silhouette scores,
          which in turn influences how much each point contributes during centroid updates.
          Options:
          - 'power': Applies a power-law weighting.
          This approach emphasizes the absolute differences in silhouette values.
          Best suited when clusters are relatively homogeneous in shape,
          size, and density—so that the magnitude of the silhouette differences is reliable.
          - 'exponential': Applies an exponential scaling based on silhouette rank.
          This method focuses on the relative ordering of points rather than their absolute differences.
          Best used when clusters vary in shape, density, or size,
          making the relative ranking of points more informative than their exact silhouette values.

    - sensitivity: float or str ("auto"), default="auto"
          Controls the contrast in weight assignment.
          Higher sensitivity values increase the emphasis on differences in silhouette values.
          - If 'auto': performs grid search to find optimal sensitivity based on "silhouette_objective".
          - If float: uses fixed sensitivity value (>0).

    - n_jobs: int, default=-1
          Number of CPU cores to use for parallel computation.
          (Set to -1 to use all available cores, to 1 for no parallelization,
          or to use half:
          >>> import multiprocessing
          >>> num_cores = multiprocessing.cpu_count()
          >>> n_jobs = num_cores//2)

    - alpha: float, default=0.5
          Parameter for combining micro-averaged and macro-averaged silhouette scores
          ( S_convex = alpha*S_micro + (1-alpha)*S_macro )
          when `silhouette_objective` is set to 'convex'.

    - tol: float, default=1e-4
          Threshold for (centroid-movement) convergence.

    Attributes
    ----------

    - labels_: np.ndarray
          Final cluster labels of each point after fitting.

    - cluster_centers_: pd.Series
          Final cluster centroids (coordinates) after fitting.

    - n_iter_: int
          Number of iterations performed during clustering.

    Methods
    -------

    - fit(X): Fits the K-Sil algorithm to the dataset X (pd.DataFrame or np.ndarray),
              identifying cluster centroids and labels.

    - predict(X): Predicts cluster labels for new data points in vector X (pd.DataFrame or np.ndarray),
                  based on the previously fitted model.

    - transform(X): Transforms X (pd.DataFrame or np.ndarray) into a new representation
                    where each sample is represented by its distances to each cluster centroid.

    - fit_predict(X): Combines fitting and prediction by clustering X and returning its cluster labels.

    - fit_transform(X): Combines fitting and transformation by clustering X and returning the transformed data.

    Notes
    -----

    - Approximation Strategy:
    For large datasets, the approximation method is recommended (with/without sampling) because it scales efficiently
    by avoiding the overhead of computing exact pairwise distances. Although we do not compute exact silhouette scores,
    this method generally retains the relative ranking of silhouette scores across different points and clusters.
    As a result, while silhouette values may slightly differ from exact computations,
    the ordering of points based on their scores remains relatively consistent.
    This reliable ordering makes the approximation strategy particularly compatible with the exponential weighting scheme,
    which leverages relative silhouette rankings rather than precise score differences ('power' weights).
    For small to medium-sized datasets, even though exact silhouette score computation is inherently more complex,
    it can often be faster in practice due to highly optimized C code implementation of 'silhouette_samples'.

    - Weighting Scheme Choice:
    If 'sensitivity != auto' (fixed weight sensitivity value): 'exponential' (weighting='exponential') works better
    than 'power' because it uses relative ranks of silhouette scores, which remain stable even with fixed sensitivity values.
    Conversely, if 'sensitivity=auto' (grid search): 'power' (weighting='power') is preferred most of the times.

    - Random Initialization and Locan Optima:
    Due to sensitivity of the initial centroid placement, the K-Sil algorithm may converge to a local optimum.
    For improved robustness, it is recommended to run the algorithm multiple times with different random seeds
    and select the result yielding the highest silhouette (objective) score.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd 
    >>> X = np.array([[1, 2], [1, 4], [1, 5],
    ...               [2, 8], [3, 6], [5, 7]])
    >>> ksil= KSil(n_clusters=2, silhouette_objective="macro", sensitivity="auto").fit(X)
    >>> ksil.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> ksil.predict([[5, 6], [9, 0]])
    array([1, 0])
    >>> np.array(ksil.cluster_centers_.to_list()) # or ksil.cluster_centers_ for pd.Series object
    array([[1.5 , 4.25],
          [3.5 , 7.5 ]])
    >>> ksil.transform([[0,0],[1,9]])
    array([[4.50693909, 8.27647268],
           [4.77624329, 2.91547595]])
    """

    def __init__(self,
                 n_clusters=3,
                 init_method='random',
                 max_iter=100,
                 random_state=42,
                 silhouette_objective='macro',
                 approximation = False,
                 sample_size=-1,
                 weighting='power',
                 sensitivity='auto',
                 n_jobs=-1,
                 alpha=0.5,
                 tol=1e-4):

        # Parameters
        self.n_clusters = n_clusters # Number of clusters to form
        self.init_method = init_method # Centroid initialization method
        self.max_iter = max_iter # Maximum number of iterations
        self.random_state = random_state # Random seed
        self.silhouette_objective = silhouette_objective # Silhouette-aggregation approach to emphasize
        self.approximation = approximation # Option for silhouette approximation
        self.sample_size = sample_size # Sample size for silhouette computation
        self.weighting = weighting # Weighting scheme to use
        self.sensitivity = sensitivity # Weight-sensitivity (fixed value or 'auto' for grid search)
        self.n_jobs = n_jobs # n_jobs for parallelization
        self.alpha = alpha # Parameter for combining micro-macro when silhouette_objective='convex'
        self.tol = tol # Centroid movement convergence threshold

        # Attributes
        self.labels_ = None # Final cluster labels for each data point
        self.cluster_centers_ = None # Final cluster centroids
        self.n_iter_ = None # Number of iterations performed during clustering

    def _initialize_centroids_kmeans(self, X, n_clusters):
        """
        Initialize centroids using a KMeans iteration with 'random' init.

        Parameters:
           - X (np.ndarray): Data array of shape (n_samples, n_features).
           - n_clusters (int): The number of clusters.

        Returns: A pandas Series of centroids and an array of initial labels.
        """
        kmeans = KMeans(n_clusters=n_clusters,
                        init='random',
                        random_state=self.random_state,
                        n_init=1,
                        max_iter=1).fit(X)
        centers = pd.Series(list(kmeans.cluster_centers_), index=range(n_clusters))
        return centers, kmeans.labels_

    def _initialize_centroids_kmeansplus(self, X, n_clusters):
        """
        Initialize centroids using a KMeans iteration with 'k-means++' init.

        Parameters:
           - X (np.ndarray): Data array of shape (n_samples, n_features).
           - n_clusters (int): The number of clusters.

        Returns: A pandas Series of centroids and an array of initial labels.
        """
        kmeans = KMeans(n_clusters=n_clusters,
                        init='k-means++',
                        random_state=self.random_state,
                        n_init=1,
                        max_iter=1).fit(X)
        centers = pd.Series(list(kmeans.cluster_centers_), index=range(n_clusters))
        return centers, kmeans.labels_

    def _initialize_centroids(self, X, n_clusters):
        """
        Initialize centroids based on the chosen method ('init_method').

        Parameters:
           - X (np.ndarray): Data array of shape (n_samples, n_features).
           - n_clusters (int): The number of clusters.

        Returns: (centers: pd.Series, labels: np.array)
        """
        if self.init_method == 'random':
            return self._initialize_centroids_kmeans(X, n_clusters)
        elif self.init_method == 'k-means++':
            return self._initialize_centroids_kmeansplus(X, n_clusters)
        else:
            raise ValueError('"init_method" must be either "random" or "k-means++".')

    def _uniform_sampling(self, X, labels, size):
        """
        Uniformly sample 'size' points from X.

        Parameters:
          - X (np.ndarray): Data array of shape (n_samples, n_features).
          - labels (np.ndarray): Cluster labels for each data point.
          - size (int): Total number of points to sample from X.

        Returns: (np.ndarray) The indices of the sampled points.

        Raises: ValueError if number of sampled indices < number of unique clusters.
        """
        indices = np.random.choice(len(X), size=size, replace=False)
        num_sampled = len(indices)
        num_clusters = len(np.unique(labels[indices]))
        if num_sampled <= num_clusters:
            raise ValueError(
                f"Uniform Sampling produced only {num_sampled} samples for {num_clusters} clusters. "
                "Increase sample_size."
            )
        return indices

    def _balanced_sampling(self, X, labels, size):
        """
        Sample roughly equal amounts from each cluster.

        Parameters:
          - X (np.ndarray): Data array of shape (n_samples, n_features).
          - labels (np.ndarray): Cluster labels for each data point.
          - size (int): Total number of points to sample from X.

        Returns: (np.ndarray) The indices of the sampled points.

        Raises: ValueError if number of sampled indices < number of unique clusters.
        """
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
                "Increase sample_size."
            )
        return indices

    def _sample_data(self, X, labels):
        """
        "Balanced" sampling for macro-averaged silhouette_objective.
        "Uniform" sampling for micro/convex silhouette_objective.

          - If sample_size == -1: use the entire dataset (return all indices).
          - If 0 <= sample_size <= 1: sample that fraction of the data.
          - If sample_size > 1: sample that exact number of points.

        Parameters:
          - X (np.ndarray): Data array of shape (n_samples, n_features).
          - labels (array-like): Cluster labels for each data point.

        Returns: (np.ndarray) The indices of the sampled points.

        Raises: ValueError if the computed sample size is 0 or invalid.
        """
        n_samples = len(X)
        labels = np.array(labels)

        if self.sample_size == -1:
            return np.arange(n_samples)
        elif 0 <= self.sample_size <= 1:
            size = int(np.ceil(self.sample_size * n_samples))
        elif self.sample_size > 1:
            size = int(self.sample_size)
        else:
            raise ValueError(
                "sample_size must be -1, a fraction (between 0 and 1), or an exact number of data points."
            )

        if size == 0:
            raise ValueError("Sample size computed as 0. Increase sample_size or adjust the parameters.")

        if self.silhouette_objective == 'macro':
            return self._balanced_sampling(X, labels, size)
        elif self.silhouette_objective in ['micro', 'convex']:
            return self._uniform_sampling(X, labels, size)
        else:
            raise ValueError(
                "Invalid 'silhouette_objective'. Expected 'macro', 'micro', or 'convex'."
            )

    def _compute_silhouette_scores(self, X, labels, centers):
        """
        Compute point-level silhouette scores for the dataset.

       - If self.approximation is True, we use a silhouette approximation.
       - If self.approximation is False we use sklearn’s silhouette_samples.

        Parameters:
          - X (np.ndarray): Data array of shape (n_samples, n_features).
          - labels (array-like): Cluster labels corresponding to each data point.
          - centers (pd.Series): Series mapping cluster labels to centroids
            (used to compute within-cluster SS and counts).

        Returns: A DataFrame with columns: 'labels', 'points', and 'silhouette'.
        """
        if len(X) == 0:
            return pd.DataFrame({'labels': [], 'points': [], 'silhouette': []})

        labels = np.asarray(labels)

        # (approximation=False) Exact silhouette computation:
        if not self.approximation:
            point_silhouette_scores = silhouette_samples(X, labels=labels)
            return pd.DataFrame({
                'labels': labels,
                'points': X.tolist(),
                'silhouette': point_silhouette_scores
            })

        # (approximation=True) Approximate silhouette computation:
        # Map labels to 0-based indices
        if hasattr(centers, 'index'):
            # Ensure every cluster present in labels has a centroid in centers
            unique_labels = np.unique(labels)
            for cl in unique_labels:
                if cl not in centers.index:
                    # If centroid not present, compute it as the mean of its cluster's points
                    centers.at[cl] = np.mean(X[labels == cl], axis=0).tolist()
            # Sort the centers by index for consistency
            centers = centers.sort_index()
            clusters = centers.index.to_numpy()
            mapping = {c: i for i, c in enumerate(clusters)}
            inv = np.array([mapping[label] for label in labels])
        else:
            clusters = np.arange(len(centers))
            inv = labels  # assuming labels are already 0-based integers

        k = len(clusters)
        n = X.shape[0]

        # Convert stored centers to a np.array
        centers_arr = np.array(centers.tolist())

        # Compute differences and squared distances between each point and its assigned centroid
        diffs = X - centers_arr[inv]       # shape: (n, d)
        dists_sq = np.sum(diffs**2, axis=1)  # shape: (n,)

        # Compute per-cluster counts and within-cluster sum-of-squares (SS)
        counts = np.bincount(inv, minlength=k)
        SS_arr = np.bincount(inv, weights=dists_sq, minlength=k)

        # Compute the full distance matrix from every point to every stored centroid
        D = cdist(X, centers_arr, metric='euclidean')

        # Extract for each point the distance to its own centroid
        D_diag = D[np.arange(n), inv]

        # For each point, get the count and SS corresponding to its cluster
        count_vec = counts[inv]
        ss_vec = SS_arr[inv]

        # Compute intra-cluster distance a(x_i)
        a_vals = np.where(
            count_vec > 1,
            np.sqrt((count_vec * (D_diag ** 2) + ss_vec) / (count_vec - 1)),
            D_diag
        )

        # Compute candidate distances for b(x_i):
        # candidate_matrix[i, j] = sqrt( D[i, j]^2 + (SS_arr[j] / counts[j]) )
        cluster_term = SS_arr / np.maximum(counts, 1e-6)
        candidate_matrix = np.sqrt(D ** 2 + cluster_term.reshape(1, k))

        # Exclude the candidate corresponding to the point's own cluster
        candidate_matrix[np.arange(n), inv] = np.inf

        # For each point, b(x_i) is the minimum candidate value.
        b_vals = np.min(candidate_matrix, axis=1)

        # Maximum of a(x_i), b(x_i)
        max_ab = np.maximum(np.maximum(a_vals, b_vals), 1e-6) # Avoid division by zero

        # Compute the silhouette scores
        s_vals = (b_vals - a_vals) / max_ab

        return pd.DataFrame({
            'labels': labels,
            'points': X.tolist(),
            'silhouette': s_vals
        })

    def _compute_silhouette_micro(self, silhouette_data):
        """
        Compute the micro-averaged silhouette score for the dataset.

        Parameters:
           - silhouette_data (pd.DataFrame): Containing a 'silhouette' column.

        Returns: The micro-averaged silhouette score (float).
        """
        if silhouette_data.empty:
            return 0
        return silhouette_data['silhouette'].mean()

    def _compute_silhouette_macro(self, silhouette_data):
        """
        Compute the macro-averaged silhouette score for the dataset.

        Parameters:
           - silhouette_data (pd.DataFrame): Containing a 'silhouette' column and a 'labels' column.

        Returns: The macro-averaged silhouette score (float).
        """
        silhouette_per_cluster = silhouette_data.groupby('labels')['silhouette']
        if silhouette_per_cluster.size == 0:
            return 0
        cluster_silhouette = silhouette_per_cluster.mean()
        return cluster_silhouette.mean()

    def _compute_silhouette_convex(self, S_micro, S_macro):
        """
        Compute the convex combination of micro-averaged and macro-averaged silhouette scores
        (self.alpha combines S_micro and S_macro).

        Parameters:
          - S_micro (float): Micro-averaged silhouette score.
          - S_macro (float): Macro-averaged silhouette score.

        Returns: Convex combination of S_micro and S_macro.
        """
        return self.alpha * S_micro + (1 - self.alpha) * S_macro

    def _power_weights(self, silhouette_data, sensitivity):
        """
        Assign weights to data points based on their silhouette scores in each cluster using a power-law scheme.

        For each cluster, this method performs the following:
        - Shift the silhouette scores: subtract the minimum silhouette score in the cluster
          (and add a small constant so that all scores are positive and division by zero is avoided afterwards).
        - Compute the median of these shifted scores within the cluster.
        - Assign each point a weight equal to the ratio of its shifted silhouette score to the cluster median,
          raised to the power 'sensitivity'. This means:
              - Points with shifted scores above the median will receive a weight > 1,
              - Points with shifted scores below the median will receive a non-negative weight < 1,
           with the exponent 'sensitivity' controlling the degree of contrast.

        These weights are then used to give more influence to points that are relatively better clustered.
        """
        # Compute shifted silhouette scores in each cluster by subtracting the minimum silhouette score
        silhouette_data['shifted_s'] = silhouette_data.groupby('labels')['silhouette'].transform(lambda s: s-s.min()+1e-6)

        # Compute the median of shifted silhouette scores
        silhouette_data['median_s'] = silhouette_data.groupby('labels')['shifted_s'].transform('median')

        # Assign weights to each data point by computing the ratio of its shifted silhouette score to the median silhouette
        # Raising this ratio to the power of 'sensitivity' emphasizes points that are above the median (or de-emphasizes those below)
        silhouette_data['weight'] = (silhouette_data['shifted_s']/silhouette_data['median_s'])**sensitivity

        return silhouette_data.drop(columns=['shifted_s', 'median_s'])

    def _exponential_weights(self, silhouette_data, sensitivity):
        """
        Assign weights to data points based on their silhouette score ranks in each cluster using an exponential-scaling scheme.

        For each cluster, this method performs the following steps:
        - Rank the points in descending order by their silhouette scores.
          The highest score receives rank 1, with dense ranking ensuring that tied scores receive the same rank.
        - Compute the maximum rank within the cluster.
        - Approximate the median rank as (max_rank + 1) / 2 and define a normalization factor (scaler) as max_rank / 2.
          These values normalize the rank differences across clusters.
        - Calculate the normalized difference from the median rank for each point.
        - Assign each point a weight using the formula:
              weight = exp(-sensitivity * normalized_distance)
          This means:
              - Points with ranks better than the median (negative normalized distance) receive weights > 1,
              - Points with ranks worse than the median receive weights < 1.
          The parameter 'sensitivity' controls the degree of contrast in these weights.

        Overall, these weights amplify the influence of data points that have relatively superior silhouette ranks
        (indicating they are better clustered) while reducing the impact of those with inferior ranks.
        """
        # Rank points within each cluster by their silhouette scores in descending order
        silhouette_data['silhouette_rank'] = silhouette_data.groupby('labels')['silhouette'] \
                                                          .rank(method='dense', ascending=False) \
                                                          .astype(int) # Dense to get "tie" ranks

        # Compute the maximum rank within each cluster
        max_rank = silhouette_data.groupby('labels')['silhouette_rank'].transform('max')

        # Approximate the median rank and define a normalization factor
        median_rank = (max_rank + 1) / 2.0

        # Scale to ensure that the exponential function behaves consistently
        scaler = max_rank / 2.0
        scaler = scaler.replace(0, 1)  # Avoid zero division

        # Compute normalized distance from the median rank
        normalized_dist = (silhouette_data['silhouette_rank'] - median_rank) / scaler

        # Compute weights using the exponential scaling formula
        silhouette_data['weight'] = np.exp(-normalized_dist * sensitivity)

        return silhouette_data.drop(columns=['silhouette_rank'])

    def _assign_weights(self, silhouette_data, sensitivity=2.0):
        """
        Assign weights to data points based on their silhouette scores based on "weighting" parameter.

        - 'power' weighting scheme:
          Use the power-law weighting scheme when clusters are relatively homogeneous in terms of size, density,
          and shape. In such cases, the absolute differences in silhouette scores are reliable. The power-law
          scheme scales each point's shifted silhouette score relative to the cluster median, so that points
          with higher-than-median scores receive weights > 1, and those with lower scores receive weights < 1.

        - 'exponential' weighting scheme:
          Use the exponential weighting scheme when clusters vary significantly in size, density, or shape. When
          clusters are heterogeneous, the relative ranking of points within each cluster is more informative than
          the absolute silhouette values. This scheme ranks the points, normalizes the rank differences relative to
          the cluster's median rank, and then applies an exponential scaling. The result is that points with ranks
          better than the median are emphasized (weight > 1), while points with worse ranks are de-emphasized.

        Parameters:
            - silhouette_data (pd.DataFrame): A DataFrame containing columns: 'silhouette' and 'labels'.
            - sensitivity (float): Parameter controlling the contrast in the weighting.

        Returns: The input DataFrame with an added 'weight' column.
        """
        if self.weighting == 'power':
            return self._power_weights(silhouette_data, sensitivity)
        elif self.weighting == 'exponential':
            return self._exponential_weights(silhouette_data, sensitivity)
        else:
            raise ValueError(
                'Invalid "weighting" parameter, must be either "power" or "exponential".'
            )

    def _update_centroids_weighted(self, n_clusters, weighted_data, previous_centers):
        """
        Update cluster centroids using weighted averages.

        For each cluster, computes the weighted mean of all data points assigned
        to that cluster, using the weights provided in the 'weight' column of weighted_data.

        Parameters:
           - n_clusters (int): Total number of clusters.
           - weighted_data (pd.DataFrame): DataFrame with columns:
                  - 'labels': cluster label for each point.
                  - 'points': a list of coordinates for each point.
                  - 'weight': the weight assigned to each point.
           - previous_centers (pd.Series): The centroids from the previous iteration.

        Returns: A pandas Series containing the updated centroids for all clusters.
        """
        def compute_weighted_mean(cluster):
            # Filter data points belonging to the current cluster
            cluster_points = weighted_data[weighted_data['labels'] == cluster]
            if not cluster_points.empty:
                # Points column to np.array
                points = np.array(cluster_points['points'].tolist())
                # Weights into (column) vector
                weights = cluster_points['weight'].values.reshape(-1, 1)
                weighted_sum = np.sum(points * weights, axis=0)
                total_weight = np.sum(weights)
                if total_weight != 0:
                    weighted_mean = weighted_sum / total_weight
                else:
                    weighted_mean = np.mean(points, axis=0)
                return weighted_mean.tolist()
            else:
                return None  # Indicate no points assigned

        jobs = self.n_jobs if n_clusters>=10 else 1 # No parallelization when clusters<10
        results = Parallel(n_jobs=jobs)(
            delayed(compute_weighted_mean)(cluster) for cluster in range(n_clusters)
        )

        # Compile the new centroids, using the previous centroid if a cluster has no points
        # Empty cluster centroids will be re-initialized afterwards
        # but we retain previous centroid to avoid inconsistencies
        new_centers = []
        for cluster, centroid in enumerate(results):
            if centroid is not None:
                new_centers.append(centroid)
            else:
                # Retain the previous centroid if empty
                if cluster < len(previous_centers):
                    new_centers.append(previous_centers[cluster])
                else:
                    # If previous_centers does not have an entry for this cluster
                    raise KeyError(f"Previous centers do not contain cluster {cluster}.")

        return pd.Series(new_centers)

    def _assign_clusters(self, X, centers):
        """
        Assign each data point in X to its nearest cluster based on Euclidean distance.

        Parameters:
           - X (array-like): The data points to be clustered.
           - centers (pd.Series): The current cluster centroids.

        Returns: An array of cluster labels.
        """
        # Centers to a np.array for efficient computation.
        centers_array = np.array(centers.tolist())
        dist_matrix = pairwise_distances(X, centers_array, metric='euclidean', n_jobs=self.n_jobs)
        labels = np.argmin(dist_matrix, axis=1)
        return labels

    def _reinitialize_empty_clusters(self, n_clusters, centers, labels, X):
        """
        Reinitialize empty clusters by reassigning them to data points from the largest cluster.

        If after point assignment, one or more clusters have no data points,
        this function reinitializes those clusters by selecting,
        from the largest cluster, the data point that is furthest from its centroid
        and assigning that point to the empty cluster (new centroid).

        Parameters:
           - n_clusters (int): Total number of clusters.
           - centers (pd.Series): Current cluster centers, indexed by cluster labels.
           - labels (np.ndarray): Current cluster assignments for each data point.
           - X (np.ndarray or pd.DataFrame): The dataset.

        Returns: (centers, labels) with updated cluster centers and assignments.
        """
        # Identify empty clusters
        empty_clusters = list(set(range(n_clusters)) - set(labels))

        if not empty_clusters:
            # If no empty clusters: No reinitialization needed
            return centers, labels

        # Determine the largest cluster (the cluster with the most data points)
        largest_cluster_label = Counter(labels).most_common(1)[0][0]

        # Ensure that the largest_cluster_label exists in centers (Optional)
        # Unnecessary because of the 'previous_centers' in centroids update
        # But we include it as a safeguard
        if largest_cluster_label not in centers.index:
            # Initialize the missing cluster center by computing the mean of its points
            points_in_largest = X[labels == largest_cluster_label]
            if points_in_largest.size == 0:
                raise ValueError(f"Largest cluster label {largest_cluster_label} has no points assigned.")
            new_centroid = points_in_largest.mean(axis=0)
            centers.at[largest_cluster_label] = list(new_centroid)

        # Retrieve the centroid of the largest cluster
        largest_centroid = np.array(centers[largest_cluster_label])

        # Get indices of points in the largest cluster
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        cluster_arr = X[largest_cluster_indices]

        # For each empty cluster, reinitialize it using a point from the largest cluster
        for empty_cluster in empty_clusters:
            if cluster_arr.size == 0:
                # No more points to assign
                break

            # Compute distances from all points in the largest cluster to its centroid
            dist_vector = pairwise_distances(cluster_arr, [largest_centroid], metric='euclidean', n_jobs=self.n_jobs).ravel()

            # Identify the index of the furthest point
            max_idx = dist_vector.argmax()

            # Get the actual index of this furthest point in the dataset
            furthest_point_idx = largest_cluster_indices[max_idx]

            # Retrieve the coordinates of the furthest point
            furthest_point_arr = X[furthest_point_idx]

            # Update the centroid of the empty cluster to be this furthest point
            if empty_cluster in centers.index:
                centers.at[empty_cluster] = list(furthest_point_arr)  # Ensure it's a list to match structure
            else:
                # If the empty cluster label is not in centers, add it
                centers.at[empty_cluster] = list(furthest_point_arr)

            # Reassign the label of this point to the newly initialized cluster label
            labels[furthest_point_idx] = empty_cluster

            # Remove the reassigned point from cluster_arr and largest_cluster_indices to avoid re-selection
            cluster_arr = np.delete(cluster_arr, max_idx, axis=0)
            largest_cluster_indices = np.delete(largest_cluster_indices, max_idx)

        # Validate that all clusters now have at least one assigned point
        unique_labels = set(labels)
        if len(unique_labels) != n_clusters:
            missing_clusters = set(range(n_clusters)) - unique_labels
            raise ValueError(f"Expected {n_clusters} clusters, but got {len(unique_labels)}. Missing clusters: {missing_clusters}.")

        return centers, labels

    def _check_centroid_stability(self, previous_centers, centers, n_clusters, X_shape):
        """
        Calculate the average movement of centroids between consecutive iterations.

        Parameters:
           - previous_centers (pd.Series): The cluster centroids from the previous iteration.
           - centers (pd.Series): The current cluster centroids.
           - n_clusters (int): The total number of clusters.
           - X_shape (tuple): The shape of the dataset X (used here to determine the number of features).

        Returns: True if the average movement is below tolerance, indicating convergence.
        """
        total_movement = 0.0

        for cluster in range(n_clusters):
            # Previous centroid
            prev_centroid = np.array(previous_centers.get(cluster, [0] * X_shape[1]))

            # Current centroid
            curr_centroid = np.array(centers.get(cluster, [0] * X_shape[1]))

            # Euclidean norm of movement
            total_movement += np.linalg.norm(curr_centroid - prev_centroid)

        return (total_movement / n_clusters) < self.tol

    def _KSil(self, X, n_clusters, max_iter, sensitivity):
        """
        Main K-Sil Clustering Algorithm.

        It initializes cluster centroids and (optionally) samples a fixed subset of the full dataset
        (if sample_size != -1) to speed up the expensive silhouette score computations,
        then iteratively:
          - Computes silhouette scores (using optional approximation).
          - Calculates the silhouette_objective score (micro, macro, or convex combination).
          - Computes weights for data points based on their silhouette scores.
          - Updates centroids using weighted averages.
          - Checks for convergence via centroid stability.
          - Reassigns each data point in the full dataset to its nearest updated centroid.
          - Reinitializes empty clusters using isolated points from the largest cluster.

        Throughout the iterations, the algorithm tracks the clustering solution (centroids and labels)
        that achieved the highest silhouette_objective score. The process stops when the centroids' movement
        falls below a small threshold or when the maximum number of iterations is reached.

        Parameters:
           - X (array-like): The input dataset of shape (n_samples, n_features).
           - n_clusters (int): The number of clusters.
           - max_iter (int): The maximum number of iterations.
           - sensitivity (float): The weight-sensitivity parameter used in the weighted centroid update.

        Returns: (best_centers, best_labels, best_score)
                - best_centers (pd.Series): The cluster centroids corresponding to the best silhouette score found.
                - best_labels (np.ndarray): The cluster labels for each data point for that best solution.
                - best_score (float): The best 'silhouette_objective' score found.
        """
        # Set random seed
        np.random.seed(self.random_state)

        # Retain a copy of the full dataset
        X_full = X.copy()

        # Initialize centroids and labels
        centers, labels = self._initialize_centroids(X_full, n_clusters)

        # If sampling is enabled, obtain fixed sample indices from the full dataset
        if self.sample_size != -1:
            sample_indices = self._sample_data(X_full, labels)
            X = X_full[sample_indices] # X_full sampled
        else:
            sample_indices = np.arange(len(X_full)) # All indices (no sampling)
            X = X_full

        # Initialize variables to track the best clustering solution
        best_score = -1
        best_centers, best_labels = centers.copy(), labels.copy()

        for n_iter in range(1, max_iter + 1):
            # Update the sampled labels from full dataset using the fixed sample indices
            labels_sampled = labels[sample_indices] # If sampling was enabled

            # Compute point-silhouette scores
            silhouette_data = self._compute_silhouette_scores(X, labels_sampled, centers)

            # Calculate the silhouette objective score
            if self.silhouette_objective == 'convex':
                # For convex combination of scores we compute both
                S_micro = self._compute_silhouette_micro(silhouette_data)
                S_macro = self._compute_silhouette_macro(silhouette_data)
                primary_score = self._compute_silhouette_convex(S_micro, S_macro)
            elif self.silhouette_objective == 'micro':
                primary_score = self._compute_silhouette_micro(silhouette_data)
            else:
                primary_score = self._compute_silhouette_macro(silhouette_data)

            # Update the best solution if the current score is better
            if primary_score > best_score:
                best_score = primary_score
                best_centers = centers.copy()
                best_labels = labels.copy()

            # Retain previous centroids for convergence checking
            previous_centers = centers.copy()

            # Compute weights based on silhouette scores
            weighted_data = self._assign_weights(silhouette_data, sensitivity)

            # Update centroids by computing the weighted average of points in each cluster
            centers = self._update_centroids_weighted(n_clusters, weighted_data, previous_centers)

            # Check centroid stability
            has_converged = self._check_centroid_stability(
                previous_centers,
                centers,
                n_clusters,
                X_full.shape
            )
            if has_converged:
                break

            # Reassign each data point in the full dataset to the nearest updated centroid
            labels = self._assign_clusters(X_full, centers)

            # Reinitialize any empty clusters using isolated data points from the largest cluster
            centers, labels = self._reinitialize_empty_clusters(n_clusters, centers, labels, X_full)

        return best_centers, np.array(best_labels), best_score, n_iter

    def _gridsearch_KSil(self, X, n_clusters):
        """
        Weight-sensitivity grid search based on 'silhouette_objective'.
        - Coarse Grid Search over a large range (0.5 to 10 with 10 values)
        - Fine Grid Search near the best coarse candidate

        Parameters:
           - X (array-like): The input dataset of shape (n_samples, n_features).
           - n_clusters (int): The number of clusters.

        Returns: (centers, labels, n_iter)
                - centers (pd.Series): The cluster centroids corresponding to the best silhouette score solution.
                - labels (np.ndarray): The cluster assignments for the best solution.
                - n_iter (int): The number of iterations performed.
        """
        original_n_jobs=self.n_jobs
        self.n_jobs=1 # To avoid nested parallelism

        def evaluate_sensitivity(sensitivity):
            best_centers, best_labels, best_score, n_iter = self._KSil(
                X, n_clusters, self.max_iter, sensitivity=sensitivity
            )
            return {
                'sensitivity': sensitivity,
                'score': best_score,
                'centers': best_centers,
                'labels': best_labels,
                'iterations': n_iter
            }

        # Coarse Grid Search: 10 values from 0.5 to 10.0
        coarse_candidates = np.linspace(0.5, 10.0, num=10) # 'sensitivity' values
        coarse_results = Parallel(n_jobs=original_n_jobs)(
            delayed(evaluate_sensitivity)(s) for s in coarse_candidates
        )

        # Sort results by descending score
        coarse_results.sort(key=lambda x: -x['score'])
        best_coarse_sensitivity = coarse_results[0]['sensitivity']

        # Fine Grid Search: 0.5 neighborhood around best coarse value, with a step size of 0.25
        step_size = 0.25
        fine_candidates = np.arange(
            max(0.5, best_coarse_sensitivity - step_size * 2),
            min(10.0, best_coarse_sensitivity + step_size * 2),
            step_size
        )
        fine_results = Parallel(n_jobs=original_n_jobs)(
            delayed(evaluate_sensitivity)(s) for s in fine_candidates
        )

        # Combine and select the best overall candidate
        final_results = coarse_results + fine_results
        final_results.sort(key=lambda x: -x['score'])
        best_result = final_results[0]

        self.n_jobs = original_n_jobs # re-setting n_jobs to the original value

        return best_result['centers'], best_result['labels'], best_result['iterations']

    def fit(self, X):
        """
        Fit the K-Sil clustering algorithm on dataset X.

        Parameters:
          - X: array-like of shape (n_samples, n_features)
               The input dataset to be clustered (pd.DataFrame or np.ndarray).

        Returns: self
                 The fitted instance of the clustering algorithm.

        Raises: ValueError
                If n_clusters is less than 2.
        """
        if self.n_clusters < 2:
            raise ValueError("Silhouette is not defined for a single cluster.")

        # Convert X to a NumPy array for consistent processing if needed
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)

        if self.sensitivity == "auto":
            self.cluster_centers_, self.labels_, self.n_iter_ = self._gridsearch_KSil(X_arr, self.n_clusters)
        else:
            self.cluster_centers_, self.labels_, _, self.n_iter_ = self._KSil(X_arr,
                                                                              self.n_clusters,
                                                                              self.max_iter,
                                                                              self.sensitivity)
        return self

    def predict(self, X):
        """
        Assign each sample in X to the nearest cluster center determined during fitting.

        Parameters:
          - X: array-like of shape (n_samples, n_features)
               New data points to be assigned to clusters (pd.DataFrame or np.ndarray).

        Returns: ndarray of shape (n_samples,)
                 Cluster labels for each sample in X.

        Raises: ValueError
                If the model has not been fitted yet.
        """
        if self.cluster_centers_ is None:
            raise ValueError('KSil model is not fitted yet. Call ".fit(...)" first.')

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)

        labels = self._assign_clusters(X_arr, self.cluster_centers_)

        return np.array(labels)

    def transform(self, X):
        """
        Transform the data X into a new representation based on the learned centroids.

        Parameters:
          - X: array-like of shape (n_samples, n_features)
               The input data to be transformed (pd.DataFrame or np.ndarray).

        Returns: ndarray of shape (n_samples, n_clusters)
                 A distance matrix where each element [i, j] is the Euclidean distance
                 from sample i to the j-th cluster centroid.

        Raises: ValueError
                If the model has not been fitted yet.
        """
        if self.cluster_centers_ is None:
            raise ValueError('KSil model is not fitted yet. Call ".fit(...)" first.')

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)

        centers_arr = np.array(self.cluster_centers_.tolist())
        distances = pairwise_distances(X_arr, centers_arr, metric='euclidean', n_jobs=self.n_jobs)

        return distances

    def fit_predict(self, X):
        """
        Fit the K-Sil clustering algorithm on dataset X and return the cluster labels.
        """
        self.fit(X)

        return self.labels_

    def fit_transform(self, X):
        """
        Fit the K-Sil clustering algorithm on dataset X and transform the data into the new representation.
        """
        self.fit(X)

        return self.transform(X)

class SphericalKSil:
    """
    Spherical K-Sil:
    This is the spherical variant of the K-Sil clustering algorithm, that operates on the unit hypersphere.
    Key differences from the standard K-Sil:
      - Unit-normalized data: Expects input X to be row-wise normalized to unit length.
      - Spherical centroids: At each iteration, centroids are computed and then
        re-normalized to unit vectors to stay on the hypersphere.
      - Euclidean on sphere: Uses Euclidean distance on normalized vectors, which
        is mathematically equivalent to cosine distance but more efficient in practice.
    This variant is especially suitable for high-dimensional embeddings such text embeddings
    or any data where direction matters more than magnitude.
    """
    def __init__(self,
                 n_clusters=3,
                 init_method='random',
                 max_iter=100,
                 random_state=42,
                 silhouette_objective='macro',
                 approximation = False,
                 sample_size=-1,
                 weighting='exponential',
                 sensitivity='auto',
                 n_jobs=-1,
                 alpha=0.5,
                 tol=1e-4):

        # Parameters
        self.n_clusters = n_clusters # Number of clusters to form
        self.init_method = init_method # Centroid initialization method
        self.max_iter = max_iter # Maximum number of iterations
        self.random_state = random_state # Random seed
        self.silhouette_objective = silhouette_objective # Silhouette-aggregation approach to emphasize
        self.approximation = approximation # Option for silhouette approximation
        self.sample_size = sample_size # Sample size for silhouette computation
        self.weighting = weighting # Weighting scheme to use
        self.sensitivity = sensitivity # Weight-sensitivity (fixed value or 'auto' for grid search)
        self.n_jobs = n_jobs # n_jobs for parallelization
        self.alpha = alpha # Parameter for combining micro-macro when silhouette_objective='convex'
        self.tol = tol # Centroid movement convergence threshold

        # Attributes
        self.labels_ = None # Final cluster labels for each data point
        self.cluster_centers_ = None # Final cluster centroids
        self.n_iter_ = None # Number of iterations performed during clustering

    def _initialize_centroids_kmeans(self, X, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters,
                        init='random',
                        random_state=self.random_state,
                        n_init=1,
                        max_iter=1).fit(X)
        centers = pd.Series(list(kmeans.cluster_centers_), index=range(n_clusters))
        return centers, kmeans.labels_

    def _initialize_centroids_kmeansplus(self, X, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters,
                        init='k-means++',
                        random_state=self.random_state,
                        n_init=1,
                        max_iter=1).fit(X)
        centers = pd.Series(list(kmeans.cluster_centers_), index=range(n_clusters))
        return centers, kmeans.labels_

    def _initialize_centroids(self, X, n_clusters):
        if self.init_method == 'random':
            return self._initialize_centroids_kmeans(X, n_clusters)
        elif self.init_method == 'k-means++':
            return self._initialize_centroids_kmeansplus(X, n_clusters)
        else:
            raise ValueError('"init_method" must be either "random" or "k-means++".')

    def _uniform_sampling(self, X, labels, size):
        indices = np.random.choice(len(X), size=size, replace=False)
        num_sampled = len(indices)
        num_clusters = len(np.unique(labels[indices]))
        if num_sampled <= num_clusters:
            raise ValueError(
                f"Uniform Sampling produced only {num_sampled} samples for {num_clusters} clusters. "
                "Increase sample_size."
            )
        return indices

    def _balanced_sampling(self, X, labels, size):
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
                "Increase sample_size."
            )
        return indices

    def _sample_data(self, X, labels):
        n_samples = len(X)
        labels = np.array(labels)

        if self.sample_size == -1:
            return np.arange(n_samples)
        elif 0 <= self.sample_size <= 1:
            size = int(np.ceil(self.sample_size * n_samples))
        elif self.sample_size > 1:
            size = int(self.sample_size)
        else:
            raise ValueError(
                "sample_size must be -1, a fraction (between 0 and 1), or an exact number of data points."
            )
        if size == 0:
            raise ValueError("Sample size computed as 0. Increase sample_size or adjust the parameters.")
        if self.silhouette_objective == 'macro':
            return self._balanced_sampling(X, labels, size)
        elif self.silhouette_objective in ['micro', 'convex']:
            return self._uniform_sampling(X, labels, size)
        else:
            raise ValueError(
                "Invalid 'silhouette_objective'. Expected 'macro', 'micro', or 'convex'."
            )

    def _compute_silhouette_scores(self, X, labels, centers):
        if len(X) == 0:
            return pd.DataFrame({'labels': [], 'points': [], 'silhouette': []})
        labels = np.asarray(labels)
        # (approximation=False) Exact silhouette computation:
        if not self.approximation:
            point_silhouette_scores = silhouette_samples(X, labels=labels)
            return pd.DataFrame({
                'labels': labels,
                'points': X.tolist(),
                'silhouette': point_silhouette_scores
            })
        # (approximation=True) Approximate silhouette computation:
        # Map labels to 0-based indices
        if hasattr(centers, 'index'):
            # Ensure every cluster present in labels has a centroid in centers
            unique_labels = np.unique(labels)
            for cl in unique_labels:
                if cl not in centers.index:
                    # If centroid not present, compute it as the mean of its cluster's points
                    centers.at[cl] = np.mean(X[labels == cl], axis=0).tolist()
            # Sort the centers by index for consistency
            centers = centers.sort_index()
            clusters = centers.index.to_numpy()
            mapping = {c: i for i, c in enumerate(clusters)}
            inv = np.array([mapping[label] for label in labels])
        else:
            clusters = np.arange(len(centers))
            inv = labels  # assuming labels are already 0-based integers

        k = len(clusters)
        n = X.shape[0]
        # Convert stored centers to a np.array
        centers_arr = np.array(centers.tolist())
        # Compute differences and squared distances between each point and its assigned centroid
        diffs = X - centers_arr[inv]       # shape: (n, d)
        dists_sq = np.sum(diffs**2, axis=1)  # shape: (n,)
        # Compute per-cluster counts and within-cluster sum-of-squares (SS)
        counts = np.bincount(inv, minlength=k)
        SS_arr = np.bincount(inv, weights=dists_sq, minlength=k)
        # Compute the full distance matrix from every point to every stored centroid
        D = cdist(X, centers_arr, metric='euclidean')
        # Extract for each point the distance to its own centroid
        D_diag = D[np.arange(n), inv]
        # For each point, get the count and SS corresponding to its cluster
        count_vec = counts[inv]
        ss_vec = SS_arr[inv]
        # Compute intra-cluster distance a(x_i)
        a_vals = np.where(
            count_vec > 1,
            np.sqrt((count_vec * (D_diag ** 2) + ss_vec) / (count_vec - 1)),
            D_diag
        )
        # Compute candidate distances for b(x_i):
        # candidate_matrix[i, j] = sqrt( D[i, j]^2 + (SS_arr[j] / counts[j]) )
        cluster_term = SS_arr / np.maximum(counts, 1e-6)
        candidate_matrix = np.sqrt(D ** 2 + cluster_term.reshape(1, k))
        # Exclude the candidate corresponding to the point's own cluster
        candidate_matrix[np.arange(n), inv] = np.inf
        # For each point, b(x_i) is the minimum candidate value.
        b_vals = np.min(candidate_matrix, axis=1)
        # Maximum of a(x_i), b(x_i)
        max_ab = np.maximum(np.maximum(a_vals, b_vals), 1e-6) # Avoid division by zero
        # Compute the silhouette scores
        s_vals = (b_vals - a_vals) / max_ab
        return pd.DataFrame({
            'labels': labels,
            'points': X.tolist(),
            'silhouette': s_vals
        })

    def _compute_silhouette_micro(self, silhouette_data):
        if silhouette_data.empty:
            return 0
        return silhouette_data['silhouette'].mean()

    def _compute_silhouette_macro(self, silhouette_data):
        silhouette_per_cluster = silhouette_data.groupby('labels')['silhouette']
        if silhouette_per_cluster.size == 0:
            return 0
        cluster_silhouette = silhouette_per_cluster.mean()
        return cluster_silhouette.mean()

    def _compute_silhouette_convex(self, S_micro, S_macro):
        return self.alpha * S_micro + (1 - self.alpha) * S_macro

    def _power_weights(self, silhouette_data, sensitivity):
        # Compute shifted silhouette scores in each cluster by subtracting the minimum silhouette score
        silhouette_data['shifted_s'] = silhouette_data.groupby('labels')['silhouette'].transform(lambda s: s-s.min()+1e-6)
        # Compute the median of shifted silhouette scores
        silhouette_data['median_s'] = silhouette_data.groupby('labels')['shifted_s'].transform('median')
        # Assign weights to each data point by computing the ratio of its shifted silhouette score to the median silhouette
        # Raising this ratio to the power of 'sensitivity' emphasizes points that are above the median (or de-emphasizes those below)
        silhouette_data['weight'] = (silhouette_data['shifted_s']/silhouette_data['median_s'])**sensitivity
        return silhouette_data.drop(columns=['shifted_s', 'median_s'])

    def _exponential_weights(self, silhouette_data, sensitivity):
        # Rank points within each cluster by their silhouette scores in descending order
        silhouette_data['silhouette_rank'] = silhouette_data.groupby('labels')['silhouette'] \
                                                          .rank(method='dense', ascending=False) \
                                                          .astype(int) # Dense to get "tie" ranks

        # Compute the maximum rank within each cluster
        max_rank = silhouette_data.groupby('labels')['silhouette_rank'].transform('max')
        # Approximate the median rank and define a normalization factor
        median_rank = (max_rank + 1) / 2.0
        # Scale to ensure that the exponential function behaves consistently
        scaler = max_rank / 2.0
        scaler = scaler.replace(0, 1)  # Avoid zero division
        # Compute normalized distance from the median rank
        normalized_dist = (silhouette_data['silhouette_rank'] - median_rank) / scaler
        # Compute weights using the exponential scaling formula
        silhouette_data['weight'] = np.exp(-normalized_dist * sensitivity)
        return silhouette_data.drop(columns=['silhouette_rank'])

    def _assign_weights(self, silhouette_data, sensitivity=2.0):
        if self.weighting == 'power':
            return self._power_weights(silhouette_data, sensitivity)
        elif self.weighting == 'exponential':
            return self._exponential_weights(silhouette_data, sensitivity)
        else:
            raise ValueError(
                'Invalid "weighting" parameter, must be either "power" or "exponential".'
            )

    def _update_centroids_weighted(self, n_clusters, weighted_data, previous_centers):
        def compute_weighted_mean(cluster):
            # Filter data points belonging to the current cluster
            cluster_points = weighted_data[weighted_data['labels'] == cluster]
            if not cluster_points.empty:
                # Points column to np.array
                points = np.array(cluster_points['points'].tolist())
                # Weights into (column) vector
                weights = cluster_points['weight'].values.reshape(-1, 1)
                weighted_sum = np.sum(points * weights, axis=0)
                total_weight = np.sum(weights)
                if total_weight != 0:
                    weighted_mean = weighted_sum / total_weight
                else:
                    weighted_mean = np.mean(points, axis=0)
                return weighted_mean.tolist()
            else:
                return None  # Indicate no points assigned

        jobs = self.n_jobs if n_clusters>=10 else 1 # No parallelization when clusters<10
        results = Parallel(n_jobs=jobs)(
            delayed(compute_weighted_mean)(cluster) for cluster in range(n_clusters)
        )
        # Compile the new centroids, using the previous centroid if a cluster has no points
        # Empty cluster centroids will be re-initialized afterwards
        # but we retain previous centroid to avoid inconsistencies
        new_centers = []
        for cluster, centroid in enumerate(results):
            if centroid is not None:
                new_centers.append(centroid)
            else:
                # Retain the previous centroid if empty
                if cluster < len(previous_centers):
                    new_centers.append(previous_centers[cluster])
                else:
                    # If previous_centers does not have an entry for this cluster
                    raise KeyError(f"Previous centers do not contain cluster {cluster}.")

        return pd.Series(new_centers)

    def _assign_clusters(self, X, centers):
        # Centers to a np.array for efficient computation.
        centers_array = np.array(centers.tolist())
        dist_matrix = pairwise_distances(X, centers_array, metric='euclidean', n_jobs=self.n_jobs)
        labels = np.argmin(dist_matrix, axis=1)
        return labels

    def _reinitialize_empty_clusters(self, n_clusters, centers, labels, X):
        # Identify empty clusters
        empty_clusters = list(set(range(n_clusters)) - set(labels))

        if not empty_clusters:
            # If no empty clusters: No reinitialization needed
            return centers, labels

        # Determine the largest cluster (the cluster with the most data points)
        largest_cluster_label = Counter(labels).most_common(1)[0][0]

        # Ensure that the largest_cluster_label exists in centers (Optional)
        # Unnecessary because of the 'previous_centers' in centroids update
        # But we include it as a safeguard
        if largest_cluster_label not in centers.index:
            # Initialize the missing cluster center by computing the mean of its points
            points_in_largest = X[labels == largest_cluster_label]
            if points_in_largest.size == 0:
                raise ValueError(f"Largest cluster label {largest_cluster_label} has no points assigned.")
            new_centroid = points_in_largest.mean(axis=0)
            centers.at[largest_cluster_label] = list(new_centroid)
        # Retrieve the centroid of the largest cluster
        largest_centroid = np.array(centers[largest_cluster_label])
        # Get indices of points in the largest cluster
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        cluster_arr = X[largest_cluster_indices]
        # For each empty cluster, reinitialize it using a point from the largest cluster
        for empty_cluster in empty_clusters:
            if cluster_arr.size == 0:
                # No more points to assign
                break
            # Compute distances from all points in the largest cluster to its centroid
            dist_vector = pairwise_distances(cluster_arr, [largest_centroid], metric='euclidean', n_jobs=self.n_jobs).ravel()
            # Identify the index of the furthest point
            max_idx = dist_vector.argmax()
            # Get the actual index of this furthest point in the dataset
            furthest_point_idx = largest_cluster_indices[max_idx]
            # Retrieve the coordinates of the furthest point
            furthest_point_arr = X[furthest_point_idx]
            # Update the centroid of the empty cluster to be this furthest point
            if empty_cluster in centers.index:
                centers.at[empty_cluster] = list(furthest_point_arr)  # Ensure it's a list to match structure
            else:
                # If the empty cluster label is not in centers, add it
                centers.at[empty_cluster] = list(furthest_point_arr)
            # Reassign the label of this point to the newly initialized cluster label
            labels[furthest_point_idx] = empty_cluster
            # Remove the reassigned point from cluster_arr and largest_cluster_indices to avoid re-selection
            cluster_arr = np.delete(cluster_arr, max_idx, axis=0)
            largest_cluster_indices = np.delete(largest_cluster_indices, max_idx)
        # Validate that all clusters now have at least one assigned point
        unique_labels = set(labels)
        if len(unique_labels) != n_clusters:
            missing_clusters = set(range(n_clusters)) - unique_labels
            raise ValueError(f"Expected {n_clusters} clusters, but got {len(unique_labels)}. Missing clusters: {missing_clusters}.")
        return centers, labels

    def _check_centroid_stability(self, previous_centers, centers, n_clusters, X_shape):
        total_movement = 0.0
        for cluster in range(n_clusters):
            # Previous centroid
            prev_centroid = np.array(previous_centers.get(cluster, [0] * X_shape[1]))
            # Current centroid
            curr_centroid = np.array(centers.get(cluster, [0] * X_shape[1]))
            # Euclidean norm of movement
            total_movement += np.linalg.norm(curr_centroid - prev_centroid)
        return (total_movement / n_clusters) < self.tol

    def _KSil(self, X, n_clusters, max_iter, sensitivity):
        # Set random seed
        np.random.seed(self.random_state)
        # Retain a copy of the full dataset
        X_full = X.copy()
        # Initialize centroids and labels
        centers, labels = self._initialize_centroids(X_full, n_clusters)
        centers = centers.apply(
        lambda r: (np.array(r, dtype=float) / np.linalg.norm(r)).tolist()
        if np.linalg.norm(r) > 0 else r)
        # If sampling is enabled, obtain fixed sample indices from the full dataset
        if self.sample_size != -1:
            sample_indices = self._sample_data(X_full, labels)
            X = X_full[sample_indices] # X_full sampled
        else:
            sample_indices = np.arange(len(X_full)) # All indices (no sampling)
            X = X_full
        # Initialize variables to track the best clustering solution
        best_score = -1
        best_centers, best_labels = centers.copy(), labels.copy()
        for n_iter in range(1, max_iter + 1):
            # Update the sampled labels from full dataset using the fixed sample indices
            labels_sampled = labels[sample_indices] # If sampling was enabled
            # Compute point-silhouette scores
            silhouette_data = self._compute_silhouette_scores(X, labels_sampled, centers)
            # Calculate the silhouette objective score
            if self.silhouette_objective == 'convex':
                # For convex combination of scores we compute both
                S_micro = self._compute_silhouette_micro(silhouette_data)
                S_macro = self._compute_silhouette_macro(silhouette_data)
                primary_score = self._compute_silhouette_convex(S_micro, S_macro)
            elif self.silhouette_objective == 'micro':
                primary_score = self._compute_silhouette_micro(silhouette_data)
            else:
                primary_score = self._compute_silhouette_macro(silhouette_data)
            # Update the best solution if the current score is better
            if primary_score > best_score:
                best_score = primary_score
                best_centers = centers.copy()
                best_labels = labels.copy()
            # Retain previous centroids for convergence checking
            previous_centers = centers.copy()
            # Compute weights based on silhouette scores
            weighted_data = self._assign_weights(silhouette_data, sensitivity)
            # Update centroids by computing the weighted average of points in each cluster
            centers = self._update_centroids_weighted(n_clusters, weighted_data, previous_centers)
            centers = centers.apply(
            lambda r: (np.array(r, dtype=float) / np.linalg.norm(r)).tolist()
            if np.linalg.norm(r) > 0 else r)
            # Check centroid stability
            has_converged = self._check_centroid_stability(
                previous_centers,
                centers,
                n_clusters,
                X_full.shape
            )
            if has_converged:
                break
            # Reassign each data point in the full dataset to the nearest updated centroid
            labels = self._assign_clusters(X_full, centers)
            # Reinitialize any empty clusters using isolated data points from the largest cluster
            centers, labels = self._reinitialize_empty_clusters(n_clusters, centers, labels, X_full)
        return best_centers, np.array(best_labels), best_score, n_iter

    def _gridsearch_KSil(self, X, n_clusters):
        original_n_jobs=self.n_jobs
        self.n_jobs=1 # To avoid nested parallelism

        def evaluate_sensitivity(sensitivity):
            best_centers, best_labels, best_score, n_iter = self._KSil(
                X, n_clusters, self.max_iter, sensitivity=sensitivity
            )
            return {
                'sensitivity': sensitivity,
                'score': best_score,
                'centers': best_centers,
                'labels': best_labels,
                'iterations': n_iter
            }
        # Coarse Grid Search: 10 values from 0.5 to 10.0
        coarse_candidates = np.linspace(0.5, 10.0, num=10) # 'sensitivity' values
        coarse_results = Parallel(n_jobs=original_n_jobs)(
            delayed(evaluate_sensitivity)(s) for s in coarse_candidates
        )
        # Sort results by descending score
        coarse_results.sort(key=lambda x: -x['score'])
        best_coarse_sensitivity = coarse_results[0]['sensitivity']
        # Fine Grid Search: 0.5 neighborhood around best coarse value, with a step size of 0.25
        step_size = 0.25
        fine_candidates = np.arange(
            max(0.5, best_coarse_sensitivity - step_size * 2),
            min(10.0, best_coarse_sensitivity + step_size * 2),
            step_size
        )
        fine_results = Parallel(n_jobs=original_n_jobs)(
            delayed(evaluate_sensitivity)(s) for s in fine_candidates
        )
        # Combine and select the best overall candidate
        final_results = coarse_results + fine_results
        final_results.sort(key=lambda x: -x['score'])
        best_result = final_results[0]
        self.n_jobs = original_n_jobs # re-setting n_jobs to the original value
        return best_result['centers'], best_result['labels'], best_result['iterations']

    def fit(self, X):
        if self.n_clusters < 2:
            raise ValueError("Silhouette is not defined for a single cluster.")
        # Convert X to a NumPy array for consistent processing if needed
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        if self.sensitivity == "auto":
            self.cluster_centers_, self.labels_, self.n_iter_ = self._gridsearch_KSil(X_arr, self.n_clusters)
        else:
            self.cluster_centers_, self.labels_, _, self.n_iter_ = self._KSil(X_arr,
                                                                              self.n_clusters,
                                                                              self.max_iter,
                                                                              self.sensitivity)
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError('KSil model is not fitted yet. Call ".fit(...)" first.')
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        labels = self._assign_clusters(X_arr, self.cluster_centers_)
        return np.array(labels)

    def transform(self, X):
        if self.cluster_centers_ is None:
            raise ValueError('KSil model is not fitted yet. Call ".fit(...)" first.')
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        centers_arr = np.array(self.cluster_centers_.tolist())
        distances = pairwise_distances(X_arr, centers_arr, metric='euclidean', n_jobs=self.n_jobs)
        return distances

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
