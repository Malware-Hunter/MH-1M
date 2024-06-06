import numpy as np
import dask.array as da
import dask.dataframe as dd
from sklearn.feature_selection import chi2
from dask import delayed, compute

class EfficientChi2FeatureSelector:
    def __init__(self, chunksize=10000, n_features=None):
        self.chunksize = chunksize
        self.n_features = n_features

# Function to compute chi2 for each chunk
    def __compute_chi2(self, X_chunk, y_chunk):
        chi2_scores, p_values = chi2(X_chunk, y_chunk)
        return chi2_scores

    def compute(self, X, y):
        # Convert to Dask arrays
        X_dask = da.from_array(X, chunks=(self.chunksize, X.shape[1]))
        y_dask = da.from_array(y, chunks=(self.chunksize,))

        # Convert Dask array to Dask DataFrame
        X_df = dd.from_dask_array(X_dask)

        
        # Apply chi2 computation in parallel using dask.delayed
        X_chunks = X_df.to_delayed()
        y_chunks = y_dask.to_delayed()
        chi2_results = [delayed(self.__compute_chi2)(X_chunk, y_chunk) for X_chunk, y_chunk in zip(X_chunks, y_chunks)]

        # Compute the results
        chi2_scores_pvalues = compute(*chi2_results)

        # Separate chi2 scores and p-values
        chi2_scores_list, p_values_list = zip(*chi2_scores_pvalues)

        # Aggregate chi2 scores
        aggregated_chi2_scores = np.mean(chi2_scores_list, axis=0)

        if self.n_features is None:
            return aggregated_chi2_scores, chi2_scores_list, p_values_list


        # # Select top n_features based on aggregated chi2 scores
        # self.top_indices = np.argsort(aggregated_chi2_scores)[-self.n_features:]
        # X_reduced = X[:, self.top_indices]

        # return X_reduced, aggregated_chi2_scores

    def transform(self, X):
        # Select the previously determined top features
        X_reduced = X[:, self.top_indices]
        return X_reduced

# Example usage
if __name__ == "__main__":
    X = np.random.rand(1340515, 22810)  # Example data
    y = np.random.randint(0, 2, size=1340515)  # Example labels

    selector = EfficientChi2FeatureSelector(chunksize=10000, n_features=100)
    X_reduced, chi2_scores = selector.fit_transform(X, y)

    print("Aggregated Chi2 Scores:", chi2_scores)
    print("Reduced Feature Set Shape:", X_reduced.shape)
