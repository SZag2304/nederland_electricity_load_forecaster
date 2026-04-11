# Netherlands Electricity Load Forecaster ⚡
A machine learning pipeline using XGBoost to predict short-term electricity load in the Netherlands. This project integrates historical grid data from ENTSO-E with localized weather features to generate high-accuracy demand forecasts.

# 📊 Project Overview
Predicting grid load is essential for utility providers to balance supply and demand. This project implements a regression model that utilizes:

**Temporal Features:** Hour, day of week, and month.

**Weather Data:** Temperature, humidity, solar radiation, and wind speed.

**Lagged Variables:** Historical load data from 1 hour, 1 day, and 1 week prior to capture cyclical consumption patterns.

## 🗄️ Data Sources
1.  **Actual Total Load:** Sourced from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/generation/actual/perType/generation?appState=%7B%22sa%22%3A%5B%22BZN%7C10YNL----------L%22%5D%2C%22st%22%3A%22BZN%22%2C%22mm%22%3Atrue%2C%22ma%22%3Afalse%2C%22sp%22%3A%22HALF%22%2C%22dt%22%3A%22TABLE%22%2C%22df%22%3A%5B%222025-04-28%22%2C%222025-05-04%22%5D%2C%22tz%22%3A%22CET%22%7D).
2.  **Weather Data:** Historical weather parameters for the Netherlands sourced via [Visual Crossing Weather](https://www.visualcrossing.com/weather-history/nederland/us/2025-03-31/2025-05-04/). Data was resampled to 15-minute intervals using linear interpolation.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`

## 🚀 Getting Started

### 1. Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 2. Execution
Ensure your CSV files are in the project directory and run:
```bash
python new_forecast.py
```

## 🧠 Feature Engineering

To capture the complex seasonality and external dependencies of the Dutch power grid, the following features were engineered:

### 1. Temporal Features
These features allow the model to understand daily, weekly, and monthly cycles in energy consumption:
* **Hour of Day:** Captures the "duck curve" and peak demand periods.
* **Day of Week:** Accounts for lower demand during weekends.
* **Month:** Captures seasonal variations in heating/cooling needs.
* **Is Weekend:** A binary flag (0 or 1) to differentiate weekday industrial load from weekend patterns.

### 2. Weather Variables
Sourced from **Visual Crossing**, these parameters provide the environmental context for load fluctuations:
* **Temperature & Humidity:** Primary drivers for HVAC-related energy demand.
* **Solar Radiation & Cloud Cover:** Essential for understanding the impact of "behind-the-meter" solar generation on net load.
* **Wind Speed & Direction:** Influences building heat loss and potential wind power correlations.

### 3. Time-Series Lag Features
Lags are the most critical features for high-accuracy forecasting. We use the following intervals (15-minute resolution):
* `lag_1`: The load from the previous 15 minutes (captures immediate momentum).
* `lag_4`: The load from 1 hour ago.
* `lag_96`: The load from exactly 24 hours ago (daily seasonality).
* `lag_672`: The load from exactly 1 week ago (weekly seasonality).

---

## 🤖 Modeling Approach

The project utilizes the **XGBoost Regressor**, a gradient-boosted decision tree algorithm known for its efficiency and predictive power in tabular time-series data.

**Model Configuration:**
* **Estimators:** 1000
* **Learning Rate:** 0.05
* **Early Stopping:** 50 rounds (to prevent overfitting)
* **Data Split:** 80% Training / 20% Testing

---

### Visualization
The script generates a comparison plot for the test set. By focusing on a 7-day subset (672 data points), we can verify that the **XGBoost Forecast** (red dashed line) matches the **Actual Load** (black line) across both peak hours and nighttime troughs.

---

## 📊 Model Performance & Robustness

To ensure the model captures physical grid drivers rather than just immediate persistence, an **Ablation Study** was performed. By removing immediate 15-minute and 1-hour lags, the model was forced to rely on seasonal and environmental variables.

### Performance Metrics (Day-Ahead Logic)
| Metric | Value |
| :--- | :--- |
| **RMSE** (Root Mean Squared Error) | **500.81** |
| **MAE** (Mean Absolute Error) | **368.44** |
| **$R^2$ Score** | **0.8423** |

<img width="1612" height="876" alt="load_forecasting" src="https://github.com/user-attachments/assets/93007300-9563-4185-bfc3-11dd724f4e91" />

---

## 🔍 Feature Importance Analysis

The XGBoost model identifies the following as the primary drivers of electricity demand in the Netherlands:

1. **`lag_96` (Daily Seasonality):** Explains ~44% of the variance. This represents the high correlation between the current load and the load from exactly 24 hours prior.
2. **`dayofweek`:** Captures the distinct load profiles of weekdays vs. weekends.
3. **`lag_672` (Weekly Seasonality):** Captures the 7-day cyclical nature of the European power market.
4. **`hour`:** Represents the standard "Dual-Peak" daily demand curve.
5. **`windgust`:** Indicates a significant meteorological influence on load, likely tied to wind-chill factors or grid-level correlations.

<img width="1312" height="976" alt="feature_importance" src="https://github.com/user-attachments/assets/500a6bb6-651e-4957-8231-eacb5a2b5f28" />

---

## 💡 Technical Takeaway
By shifting away from a "Persistence Model" ($T-1$ lag), this implementation provides a more realistic **Day-Ahead Forecast**. An $R^2$ of 0.84 demonstrates a high degree of reliability for utility planning, where weather and historical weekly patterns are the only known variables 24 hours in advance.

---

## 🛠️ Usage
1. Ensure the data files are in the local directory.
2. Run the processing and training script.
3. The script will output the performance metrics and display a time-series plot.

