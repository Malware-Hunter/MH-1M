import pandas as pd

class DataFrameChecker:
    def __init__(self, df):
        """
        Initialize the DataFrameChecker with a DataFrame to inspect.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to check for inconsistencies.
        """
        self.df = df

    def check_nulls(self):
        """
        Checks for any null or NaN values in the DataFrame.

        Returns:
        pd.DataFrame: A DataFrame indicating the number of null values in each column.
        """
        return self.df.isnull().sum()

    def check_data_types(self):
        """
        Checks for data types of each column in the DataFrame.

        Returns:
        pd.DataFrame: A DataFrame showing the data type of each column.
        """
        return self.df.dtypes

    def report_inconsistent_types(self):
        """
        Identifies columns that may contain mixed data types, which could be problematic.

        Returns:
        dict: A dictionary with column names as keys and the types of data found as values.
        """
        inconsistent_types = {}
        for column in self.df.columns:
            unique_types = set(self.df[column].apply(type).unique())
            if len(unique_types) > 1:
                inconsistent_types[column] = unique_types
        return inconsistent_types

    def summary(self):
        """
        Provides a summary report of all inconsistencies in the DataFrame.

        Returns:
        dict: A dictionary containing reports of null values, data types, and inconsistent types.
        """
        return {
            'null_values': self.check_nulls(),
            'data_types': self.check_data_types(),
            'inconsistent_types': self.report_inconsistent_types()
        }