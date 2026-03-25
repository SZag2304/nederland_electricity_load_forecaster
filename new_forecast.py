import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

df = pd.read_csv('/Users/szag_2304/Desktop/Project/nederland_load_forecast/Total Load Day Ahead Mar-May 2025.csv')

print(df.head())

# Convert 'Date' column to datetime
df['timestamp'] = df['time'].str.split(' - ').str[0]
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
df.drop(columns=['time'], inplace=True)

# Move 'timestamp' to the first column (index 0)
df.insert(0, 'timestamp', df.pop('timestamp'))

print(df.head())

df_weather = pd.read_csv('/Users/szag_2304/Desktop/Project/nederland_load_forecast/Netherlands Mar-May Weather.csv')

print(df_weather.head())

df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

df_weather = df_weather.set_index('datetime')
df_weather_15min = df_weather.resample('15min').interpolate(method='linear').reset_index()

print(df_weather_15min.head())

# Merge the load and weather data on the timestamp
df_merged = pd.merge(df, df_weather_15min, left_on='timestamp', right_on='datetime', how='inner')

print(df_merged.head())

df_merged['hour'] = df_merged['timestamp'].dt.hour
df_merged['dayofweek'] = df_merged['timestamp'].dt.dayofweek
df_merged['month'] = df_merged['timestamp'].dt.month

# Adding event features (weekend/holiday)
df_merged['is_weekend'] = df_merged['dayofweek'].isin([5, 6]).astype(int)

# Create lag features for the target variable

target_col = 'actual_total_load_mw'

df_merged['lag_1'] = df_merged[target_col].shift(1)   # 15 minutes ago
df_merged['lag_4'] = df_merged[target_col].shift(4)  # 1 hour ago (4 * 15 minutes)
df_merged['lag_96'] = df_merged[target_col].shift(96) # 1 day ago (96 * 15 minutes)
df_merged['lag_672'] = df_merged[target_col].shift(672) # 1 week ago (672 * 15 minutes)

# Train/test split

split_index = int(len(df_merged) * 0.7)
train = df_merged.iloc[:split_index]
test = df_merged.iloc[split_index:]

print(f"Length of merged df: {len(df_merged)}")
print(f"Length of train: {len(train)}")
print(f"Length of test: {len(test)}")

feature = ['temp', 'humidity', 'precip', 'snow', 'windgust', 'winddir', 'windspeed', 'cloudcover', 'visibility', 'solarradiation', 'hour', 'dayofweek', 'month', 'lag_96', 'lag_672']
target = 'actual_total_load_mw'

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50, n_jobs=-1)

model.fit(train[feature], train[target],
     eval_set=[(test[feature], test[target])],
     verbose=False)

preds = model.predict(test[feature])
rmse = np.sqrt(mean_squared_error(test[target], preds))
r2 = r2_score(test[target], preds)
mae = np.mean(np.abs(test[target] - preds))

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")

plt.figure(figsize=(15, 7))
# View last 7 days of test set (672 rows) to see the full week pattern
subset = min(672, len(test))  # Ensure we don't exceed test set length
plt.plot(test['timestamp'].iloc[:subset], test['actual_total_load_mw'].iloc[:subset], label='Actual Load', color='black', alpha=0.6)
plt.plot(test['timestamp'].iloc[:subset], preds[:subset], label='XGBoost Forecast', color='#D43F3A', linestyle='--')
'''plt.figure(figsize=(12, 6))
plt.plot(test['timestamp'], test[target], label='Actual Load', marker='o')
plt.plot(test['timestamp'], preds, label='Predicted Load', marker='x')'''
plt.xlabel('Timestamp')
plt.ylabel('Load (MW)')
plt.title(f'Load Forecasting (R2: {r2:.4f})')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a dataframe for visualization
importances = model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': feature, 'Importance': importances})

# Sort by importance
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('XGBoost Feature Importance (Gain)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Print the top 5 features for the README
print("Top 5 Predictive Features:")
print(feature_imp_df.head(5))