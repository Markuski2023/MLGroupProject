import pandas as pd
from sklearn.impute import SimpleImputer
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error

def remove_unwanted_rows(df):
    unwanted_rows = (df['direct_rad:W'] == 0) & (df['diffuse_rad:W'] == 0) & (df['pv_measurement'] > 200 & (df['sun_elevation:d'] < 0) & (df['is_day:idx'] == 0))
    cleaned_df = df[~unwanted_rows]
    return cleaned_df
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

def split_df_on_date(df, date_as_string):
    df['time'] = pd.to_datetime(df['time'])

    first_df = df[df['time'] < date_as_string]
    second_df = df[df['time'] >= date_as_string]

    return first_df, second_df

def split_df_on_ratio(df, ratio, random=False):
    if random:
        df = df.sample(frac=1).reset_index(drop=True)
    split_index = int(len(df) * ratio)
    df1 = df.iloc[:split_index]
    df2 = df.iloc[split_index:]

    return df1, df2  

def is_estimated(df):
    split_date = '2022-10-27'
    df['is_estimated'] = 0  # Initialize with 0 (indicating observed)
    df.loc[df['time'] >= pd.Timestamp(split_date), 'is_estimated'] = 1  # Set 1 for estimated data
    return df

def mean_of_the_hour(df):
    # Ensure 'time' column is a datetime object
    df['time'] = pd.to_datetime(df['time'])
    
    # Get number of rows
    n_rows = len(df)
    
    # Iterate through rows
    i = 0
    while i < n_rows - 3: # Subtract 3 to avoid index out of range
        current_time = df.iloc[i]['time']
        # Check if current row's time is a full hour
        if current_time.minute == 0 and current_time.second == 0:
            # Check next three rows
            next_three_rows = df.iloc[i+1:i+4]
            if not all(next_three_rows['time'].dt.minute == 0) or not all(next_three_rows['time'].dt.second == 0):
                # Compute the mean only for numeric columns
                mean_values = next_three_rows.select_dtypes(include=['float64', 'int64']).mean()
                for col in mean_values.index:
                    df.at[i, col] = mean_values[col]
        i += 1
    
    return df