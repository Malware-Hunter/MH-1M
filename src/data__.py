from collections import Counter
import os
import argparse
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from pandas import DataFrame as dataframe
from os.path import join
# from tqdm import tqdm
import random

from sklearn.feature_selection import SelectKBest, chi2

class Balance:

    @staticmethod
    def find_least_common_class(y_labels):
        # Count the frequency of each class in y_labels
        counter = Counter(y_labels)
        
        # Find the class with the minimum count
        least_common_class, least_common_count = counter.most_common()[-1]
        
        return (least_common_class, least_common_count)

    @staticmethod
    def balance_classes(data, MAX_INSTANCES):
        """
        Balances classes by under-sampling each class to a specified maximum number of instances.
        
        Parameters:
        - X: Features, numpy array of shape (n_samples, n_features)
        - y_ohe: One-hot encoded labels, numpy array of shape (n_samples, n_classes)
        - label_list: List of actual labels corresponding to each sample
        - MAX_INSTANCES: The maximum number of instances allowed per class.
        
        Returns:
        - A dictionary with balanced features, labels, one-hot encoded labels, actual label list, and indices.
        """
        # y = data_dict['labels']['labels']
        unique_classes = np.unique(data['class'])
    
        min_class, min_value = Balance.find_least_common_class(y_labels=data['class'])
    
        if min_value < MAX_INSTANCES:
            print(f'Class {min_class} with only {min_value} samples. Updating MAX_INSTANCES to {min_value}.')
            MAX_INSTANCES = min_value
        
        under_sample = pd.DataFrame()
        # np.random.seed(0)  # For reproducibility
        for class_id in unique_classes:
            # Find indices of the current class
            group = data[data['class']==class_id]
            sampled = group.sample(MAX_INSTANCES, random_state=0, replace=False)
        
        # Concatenate all sampled indices from each class
            under_sample = pd.concat([under_sample, sampled])
    
        return under_sample.reset_index(drop=True)

    @staticmethod
    def randomUndersampling(path_datasets, MAX_INSTANCES=5000, MAX_FEATURES=1000, path_save=None):
    
        for dataset_name in os.listdir(path_datasets):
            print(dataset_name)
            data = pd.read_csv(join(path_datasets, dataset_name))
            print(f'\tdata shape: {data.shape}')
            
            labels = data['class']
            data.drop(columns=['class'], inplace=True)
            # column_names = process_column(column_names=data.columns.values, delimiter='.')
            column_names = data.columns.values
    
            if MAX_FEATURES is not None and len(column_names) > MAX_FEATURES:
                print(f'\tFeature selection')
                chi2_features = Balance.feature_selection(data.values, labels.values, column_names, MAX_FEATURES=MAX_FEATURES)
                data = data[chi2_features['names'].values]
    
            data['class'] = labels.values
            
            if MAX_INSTANCES is not None:
                data = Balance.balance_classes(data=data, MAX_INSTANCES=MAX_INSTANCES)
                print(f'\tBalanced shape: {data.shape}')
        
            has_nan = data.isnull().values.any()
            if has_nan:
                print(f'\nHas NaN: {has_nan}')
                # checker = DataFrameChecker(balanced_data)
                # print(checker.summary())
            
            if path_save:
                print(f'\tSaving data with shape: {data.shape}')
                data.to_csv(
                    join(path_save, f'{dataset_name.replace('.csv', '')}-balanced.csv'),
                    index=False
                )
            print('========================================================')
            
        return
        
    @staticmethod
    def feature_selection(data, labels, col_names, MAX_FEATURES=0):
        chi2_stats, p_values = chi2(data, labels)  # Virus total scanners detections >= 4
        df_chi2 = dataframe({
            'names': col_names,         
            'stats': chi2_stats,
            'p_values': p_values
        })
    
        chi2_sorted = df_chi2.sort_values(by='stats', ascending=False).dropna()
    
        # chi2_features = df_chi2[df_chi2['p_values'] < 0.05]  ## significance level (e.g. α = .05), and .head for TOP K
        # chi2_features = chi2_sorted[chi2_sorted['p_values'] < 0.05]  ## significance level (e.g. α = .05), and .head for TOP K
    
        if MAX_FEATURES>0:
            return chi2_sorted.head(MAX_FEATURES)
        
        return  chi2_sorted


    