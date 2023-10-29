import pandas as pd
from sklearn.impute import SimpleImputer
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error
import numpy as np

def remove_unwanted_rows(df):
    unwanted_rows = (df['direct_rad:W'] == 0) & (df['diffuse_rad:W'] == 0) & (df['pv_measurement'] > 200) & (df['sun_elevation:d'] < 0) & (df['is_day:idx'] == 0)
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

def lag_features_by_one_hour(df, column_names, time_col='time'):

    # Check if the DataFrame has a time-based index
    df['index'] = df[time_col]
    df = df.set_index('index')

    # Loop through each column name to create a lagged feature
    for col in column_names:
        lagged_col_name = f"{col}"
        df[lagged_col_name] = df[col].shift(freq='1H')

    return df

def is_estimated(df, time_col='time'):
    split_date = '2022-10-27'
    df['is_estimated'] = 0  # Initialize with 0 (indicating observed)
    df.loc[df[time_col] >= pd.Timestamp(split_date), 'is_estimated'] = 1  # Set 1 for estimated data
    return df

def resample_to_hourly(df, datetime_column='date_forecast'):
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df.sort_values(by=datetime_column, inplace=True)

    df.drop_duplicates(subset=[datetime_column], keep='first', inplace=True)

    df.set_index(datetime_column, inplace=True)

    df_hourly = df.resample('H').mean()

    df_hourly.dropna(how='all', inplace=True)

    df_hourly.reset_index(inplace=True)

    return df_hourly

def add_weather_indicators(df, target_variable='pv_measurement'):
    """
    Add weather indicators as binary features for extreme weather conditions.
    'High' and 'Low' indicators are created based on one standard deviation from the mean.
    The function automatically excludes the target variable from consideration.
    """
    all_features = [col for col in df.columns if col != target_variable]

    for feature in all_features:
        std_dev = df[feature].std()
        mean_val = df[feature].mean()
        df[f'{feature}_high'] = (df[feature] > mean_val + std_dev).astype(int)
        df[f'{feature}_low'] = (df[feature] < mean_val - std_dev).astype(int)
    return df

def feature_engineering_1(df):
    df['radiation_ratio'] = df['direct_rad:W'] / (df['diffuse_rad:W'] + 1e-6)
    df['cloud_rad_interaction'] = df['effective_cloud_cover:p'] * (df['direct_rad:W'] + df['diffuse_rad:W'])

    df['wind_magnitude'] = np.sqrt(df['wind_speed_u_10m:ms']**2 + df['wind_speed_v_10m:ms']**2)
    df['wind_direction'] = np.arctan2(df['wind_speed_v_10m:ms'], df['wind_speed_u_10m:ms'])

    df['solar_angle_impact'] = np.sin(np.radians(df['sun_elevation:d']))

    df = add_weather_indicators(df)

    return df