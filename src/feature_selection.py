import numpy as np
import dask.array as da
import dask.dataframe as dd
from sklearn.feature_selection import chi2
from dask import delayed, compute
import pandas as pd


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

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import chi2
from scipy.stats import ttest_ind
import joblib

class RandomChi2:
    @staticmethod
    def random_sampling_chi2_test(X, y, k_folds=10, random_state=0):
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        chi2_scores_all = []
        p_values_all = []

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Fold {i}:")
            print(X[test_index].shape, y[test_index].shape)
            chi2_scores, p_values = chi2(X[test_index], y[test_index])
            chi2_scores_all.append(chi2_scores)
            p_values_all.append(p_values)

        return np.array(chi2_scores_all), np.array(p_values_all)
    
    @staticmethod
    def test_significance(chi2_scores_all):
        # Perform statistical significance test (e.g., t-test) between folds
        num_features = chi2_scores_all.shape[1]
        p_values = []
        
        for i in range(num_features):
            scores = chi2_scores_all[:, i]
            t_stat, p_val = ttest_ind(scores[:len(scores)//2], scores[len(scores)//2:])
            p_values.append(p_val)
        
        return np.array(p_values)

    def chi2_sampling(data, y, col_names,  k=10):
        np.random.seed(0)        
        fold_size = int(data.shape[0]/k)
        sample_indices = np.random.choice(data.shape[0], fold_size, replace=False)

        chi2_stats, p_values = chi2(data[sample_indices], y[sample_indices])  # Virus total scanners detections >= 4

        print(chi2_stats.shape, p_values.shape)

        df_chi2 = pd.DataFrame({
            'names': col_names,
            'stats': chi2_stats,
            'p_values': p_values
        })

        df_chi2.head()


        chi2_sorted = df_chi2.sort_values(by='stats', ascending=False).dropna()

        # chi2_features = df_chi2[df_chi2['p_values'] < 0.05]  ## significance level (e.g. α = .05), and .head for TOP K
        chi2_features = chi2_sorted[chi2_sorted['p_values'] < 0.05]  ## significance level (e.g. α = .05), and .head for TOP K

        # chi2_features.to_csv(join(paths['data'], 'amex', 'amex-1M-chi2-features.csv'), index=False)
        return 


# Example usage
if __name__ == "__main__":
    X = np.random.rand(1340515, 22810)  # Example data
    y = np.random.randint(0, 2, size=1340515)  # Example labels

    selector = EfficientChi2FeatureSelector(chunksize=10000, n_features=100)
    X_reduced, chi2_scores = selector.fit_transform(X, y)

    print("Aggregated Chi2 Scores:", chi2_scores)
    print("Reduced Feature Set Shape:", X_reduced.shape)
