import pandas as pd
from sklearn.impute import SimpleImputer
from autogluon.tabular import TabularPredictor
def remove_highly_correlated_features(df, threshold):
    # Compute the Pearson correlation matrix
    correlation_matrix = df.corr(method='pearson')

    # Initialize an empty list to hold features to be removed
    features_to_remove = []

    # Traverse the correlation matrix to find highly correlated features
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]

            # Check for high absolute correlation
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                # Add one of the features to the list if it's not already there
                if feature1 not in features_to_remove and feature2 not in features_to_remove:
                    features_to_remove.append(feature1)

    # Drop the identified features from the DataFrame
    filtered_df = df.drop(columns=features_to_remove)

    return filtered_df

def find_long_constant_periods(data, threshold):
    start = None
    segments = []
    for i in range(1, len(data)):
        if data[i] == data[i-1] and data[i] != 0:
            if start is None:
                start = i-1
        else:
            if start is not None:
                if (i - start) > threshold:
                    segments.append((start, i))
                start = None
    return segments

def remove_constant_periods(df, segments):
    drop_indices = []
    for start, end in segments:
        drop_indices.extend(range(start, end))
    return df.drop(drop_indices)

def lag_features_by_one_hour(df, column_names):

    # Check if the DataFrame has a time-based index
    df['index'] = df['time']
    df = df.set_index('index')

    # Loop through each column name to create a lagged feature
    for col in column_names:
        lagged_col_name = f"{col}"
        df[lagged_col_name] = df[col].shift(freq='1H')

    return df

def is_estimated(df):
    split_date = '2022-10-27'
    df['is_estimated'] = 0  # Initialize with 0 (indicating observed)
    df.loc[df['time'] >= pd.Timestamp(split_date), 'is_estimated'] = 1  # Set 1 for estimated data
    return df