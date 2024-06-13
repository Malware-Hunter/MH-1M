# import tensorflow as tf
class DataManager:


    # def __init__(self, data, labels) -> None:
    #     # Convert the numpy arrays to tf.data.Dataset
    #     self.dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    #     pass

    # Define your preprocessing function
    def preprocess(self, features, num_classes):
        # features['numeric_feature'] = tf.cast(features['numeric_feature'], tf.float32) / 100.0
        features['categorical_feature'] = tf.one_hot(features['categorical_feature'], depth=num_classes)
        return features
    
    def create_dataset(self, data, labels, buffer_size=10000, batch_size= 32, train_test=True):
        
        # Convert the numpy arrays to tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))

        if train_test:
            # Split the dataset
            train_dataset, test_dataset = self.train_test_split(
                dataset, 
                test_size=0.2, 
                shuffle=True, 
                buffer_size=buffer_size
                )

            # Shuffle, batch, and prefetch the datasets
            
            train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return train_dataset, test_dataset
        else:
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            return dataset

    # Define the train-test split function
    def train_test_split(self, dataset, test_size=0.2, shuffle=True, buffer_size=1000):
        random_state=0
        tf.random.set_seed(random_state)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size, seed=random_state)
        # Calculate the sizes
        dataset_size = dataset.cardinality().numpy()
        test_size = int(test_size * dataset_size)
        # Split the dataset
        train_size = dataset_size - test_size
        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        return train_dataset, test_dataset

    @staticmethod
    def modify_column(df, column, feature_name='apicall'):

        def modify_string(s):
            # Replace '_' with '::'
            s = s.lower()
            s = s.replace('_', '::', 1)
            s = s.replace('.', '/')
            s = s[::-1].replace('/', '.', 1)[::-1]
            s = s.replace(f'{feature_name}::', f'{feature_name}s::')
            s = s.replace('()', '')
            # Replace the last '/' with '.'
            # modified_str = re.sub(r'/(?!.*\/)', '.', s)
            return s
        
        df[column] = df[column].apply(modify_string)
        return df


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
    

