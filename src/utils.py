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

def generate_solar_features_1(data):
    relevant_features = [
        'direct_rad:W', 'clear_sky_rad:W', 'diffuse_rad:W', 'sun_elevation:d', 'sun_azimuth:d',
        'clear_sky_energy_1h:J', 'direct_rad_1h:J', 'effective_cloud_cover:p', 'diffuse_rad_1h:J',
        'is_in_shadow:idx', 'total_cloud_cover:p', 'wind_speed_u_10m:ms', 'snow_water:kgm2',
        'relative_humidity_1000hPa:p', 'is_day:idx', 'wind_speed_v_10m:ms', 'cloud_base_agl:m',
        'fresh_snow_24h:cm', 'wind_speed_10m:ms', 'pressure_100m:hPa'
    ]

    interactions = {}
    ratios = {}
    differences = {}
    lags = {}
    self_interactions = {}

    for col_pair in itertools.combinations(relevant_features, 2):
        interactions[f'{col_pair[0]}_times_{col_pair[1]}'] = data[col_pair[0]] * data[col_pair[1]]
        ratios[f'{col_pair[0]}_div_{col_pair[1]}'] = data[col_pair[0]] / (data[col_pair[1]] + 1e-8)
        differences[f'{col_pair[0]}_minus_{col_pair[1]}'] = data[col_pair[0]] - data[col_pair[1]]
        self_interactions[f'{col_pair[0]}_squared'] = data[col_pair[0]] ** 2

    # Creating lags for all relevant features
    for col in relevant_features:
        lags[f'{col}_lag1'] = data[col].shift(1)
        lags[f'{col}_lag2'] = data[col].shift(3)
        lags[f'{col}_lag3'] = data[col].shift(6)

    # Concatenate all new features with the original data
    data = pd.concat([data, pd.DataFrame(interactions), pd.DataFrame(ratios),
                      pd.DataFrame(differences), pd.DataFrame(lags), pd.DataFrame(self_interactions)], axis=1)

    data['wind_magnitude'] = np.sqrt(data['wind_speed_u_10m:ms']**2 + data['wind_speed_v_10m:ms']**2)
    data['wind_direction'] = np.arctan2(data['wind_speed_v_10m:ms'], data['wind_speed_u_10m:ms'])
    data['solar_angle_impact'] = np.sin(np.radians(data['sun_elevation:d']))

    return data

def generate_solar_features_2(data):
    relevant_features = [
        'clear_sky_rad:W', 'sun_elevation:d', 'direct_rad:W', 'diffuse_rad:W',
        'sun_azimuth:d', 'clear_sky_energy_1h:J', 'cloud_base_agl:m', 'effective_cloud_cover:p',
        'diffuse_rad_1h:J', 'snow_water:kgm2', 'ceiling_height_agl:m', 'total_cloud_cover:p',
        'direct_rad_1h:J'
    ]

    interactions = {}
    ratios = {}
    differences = {}
    lags = {}
    self_interactions = {}

    for col_pair in itertools.combinations(relevant_features, 2):
        interactions[f'{col_pair[0]}_times_{col_pair[1]}'] = data[col_pair[0]] * data[col_pair[1]]
        ratios[f'{col_pair[0]}_div_{col_pair[1]}'] = data[col_pair[0]] / (data[col_pair[1]] + 1e-8)
        differences[f'{col_pair[0]}_minus_{col_pair[1]}'] = data[col_pair[0]] - data[col_pair[1]]
        self_interactions[f'{col_pair[0]}_squared'] = data[col_pair[0]] ** 2

    # Creating lags for all relevant features
    for col in relevant_features:
        lags[f'{col}_lag1'] = data[col].shift(1)
        lags[f'{col}_lag2'] = data[col].shift(3)
        lags[f'{col}_lag3'] = data[col].shift(6)

    # Concatenate all new features with the original data
    data = pd.concat([data, pd.DataFrame(interactions), pd.DataFrame(ratios),
                      pd.DataFrame(differences), pd.DataFrame(lags), pd.DataFrame(self_interactions)], axis=1)

    data['wind_magnitude'] = np.sqrt(data['wind_speed_u_10m:ms']**2 + data['wind_speed_v_10m:ms']**2)
    data['wind_direction'] = np.arctan2(data['wind_speed_v_10m:ms'], data['wind_speed_u_10m:ms'])
    data['solar_angle_impact'] = np.sin(np.radians(data['sun_elevation:d']))

    return data

def generate_solar_features_3(data):
    relevant_features = [
        'direct_rad:W', 'clear_sky_rad:W', 'diffuse_rad:W', 'sun_elevation:d', 'sun_azimuth:d',
        'clear_sky_energy_1h:J', 'direct_rad_1h:J', 'effective_cloud_cover:p', 'diffuse_rad_1h:J',
        'is_in_shadow:idx', 'total_cloud_cover:p', 'wind_speed_u_10m:ms', 'snow_water:kgm2',
        'relative_humidity_1000hPa:p', 'is_day:idx', 'wind_speed_v_10m:ms', 'cloud_base_agl:m',
        'fresh_snow_24h:cm', 'wind_speed_10m:ms', 'pressure_100m:hPa'
    ]

    interactions = {}
    ratios = {}
    differences = {}
    lags = {}
    self_interactions = {}

    for col_pair in itertools.combinations(relevant_features, 2):
        interactions[f'{col_pair[0]}_times_{col_pair[1]}'] = data[col_pair[0]] * data[col_pair[1]]
        ratios[f'{col_pair[0]}_div_{col_pair[1]}'] = data[col_pair[0]] / (data[col_pair[1]] + 1e-8)
        differences[f'{col_pair[0]}_minus_{col_pair[1]}'] = data[col_pair[0]] - data[col_pair[1]]
        self_interactions[f'{col_pair[0]}_squared'] = data[col_pair[0]] ** 2

    # Creating lags for all relevant features
    for col in relevant_features:
        lags[f'{col}_lag1'] = data[col].shift(1)
        lags[f'{col}_lag2'] = data[col].shift(3)
        lags[f'{col}_lag3'] = data[col].shift(6)

    # Concatenate all new features with the original data
    data = pd.concat([data, pd.DataFrame(interactions), pd.DataFrame(ratios),
                      pd.DataFrame(differences), pd.DataFrame(lags), pd.DataFrame(self_interactions)], axis=1)

    data['wind_magnitude'] = np.sqrt(data['wind_speed_u_10m:ms']**2 + data['wind_speed_v_10m:ms']**2)
    data['wind_direction'] = np.arctan2(data['wind_speed_v_10m:ms'], data['wind_speed_u_10m:ms'])
    data['solar_angle_impact'] = np.sin(np.radians(data['sun_elevation:d']))

    return data

def closest_impute(series):

    ffill = series.fillna(method='ffill')
    bfill = series.fillna(method='bfill')

    # Calculate the distances to the nearest non-NaN values
    ffill_dist = series.index.to_series().fillna(method='ffill') - series.index.to_series()
    bfill_dist = series.index.to_series().fillna(method='bfill') - series.index.to_series()

    # Where the forward fill distance is smaller or equal, use ffill, otherwise use bfill
    combined = np.where(ffill_dist <= bfill_dist, ffill, bfill)

    return combined

