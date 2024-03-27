import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('wind_dataset.csv')

data.dropna(inplace=True)

# Check for missing values
print("Missing values:")
print(data.isnull().sum())

# Impute missing values with mean imputation
imputer = SimpleImputer(strategy='mean')
data[['WIND', 'RAIN', 'T.MAX', 'T.MIN', 'T.MIN.G']] = imputer.fit_transform(data[['WIND', 'RAIN', 'T.MAX', 'T.MIN', 'T.MIN.G']])

# Convert DATE column to datetime format
data['DATE'] = pd.to_datetime(data['DATE'])

# Create new features based on DATE
data['MONTH'] = data['DATE'].dt.month
data['DAY'] = data['DATE'].dt.day
data['YEAR'] = data['DATE'].dt.year
data['DAY_OF_YEAR'] = data['DATE'].dt.dayofyear

# Create lagged features for wind speed and temperature
data['WIND_PREV_1'] = data['WIND'].shift(1)
data['WIND_PREV_2'] = data['WIND'].shift(2)
data['WIND_PREV_3'] = data['WIND'].shift(3)
data['T.MAX_PREV_1'] = data['T.MAX'].shift(1)
data['T.MIN_PREV_1'] = data['T.MIN'].shift(1)

# Drop rows with missing values introduced by lagged features
data.dropna(inplace=True)

# Scale the features using StandardScaler
scaler = StandardScaler()
columns_to_scale = ['WIND', 'RAIN', 'T.MAX', 'T.MIN', 'T.MIN.G', 'MONTH', 'DAY', 'YEAR', 'DAY_OF_YEAR',
                    'WIND_PREV_1', 'WIND_PREV_2', 'WIND_PREV_3', 'T.MAX_PREV_1', 'T.MIN_PREV_1']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# Print the preprocessed dataset
print("Preprocessed dataset:")
print(data.head())