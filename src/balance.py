import sys
import os
import argparse
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-d', '--dataset', metavar = 'DATASET',
        help = 'Dataset (csv file).', type = str, required = True)
    parser.add_argument(
        '--class-column', metavar = 'CLASS_COLUMN',
        help = 'Classification Column.', type = str, default = 'class')
    args = parser.parse_args(argv)
    return args


if __name__=="__main__":
    args = parse_args(sys.argv[1:])

    try:
        dataset = pd.read_csv(args.dataset)
    except BaseException as e:
        print('Exception: {}'.format(e))
        exit(1)
    print(dataset.shape)
    # Split variables into X and y
    X = dataset.drop(args.class_column, axis = 1)
    y = dataset[args.class_column] # target variable
    rus = RandomUnderSampler(random_state=42)
    # Balancing the data
    X_resampled, y_resampled = rus.fit_resample(X, y)
    balanced_dataset = pd.concat([X_resampled, y_resampled], axis = 1)
    print(balanced_dataset.shape)
    print(balanced_dataset)
    balanced_dataset.to_csv('balanced.csv', index = False)
