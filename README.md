# walmart-sales-prediction-ml

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-green)
![ARIMA](https://img.shields.io/badge/ARIMA-TimeSeries-purple)
![Flask](https://img.shields.io/badge/Flask-WebApp-black)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Predicting Walmart weekly sales using Machine Learning and Time Series Forecasting — with a Flask web app for live predictions. The project covers end-to-end data science: cleaning, EDA, feature engineering, 6 ML models, hyperparameter tuning, ARIMA, SARIMA, Prophet with cross validation, and a deployed web interface.

---

## Web App Demo

Enter store details in the form and get an instant weekly sales prediction powered by the trained RandomForest model.

```
Store No  →  Holiday Week?  →  Temperature  →  CPI  →  Unemployment
Month  →  Year  →  Previous Week Sales  →  Predict
```

> Built with Flask + HTML/CSS — runs locally on `http://127.0.0.1:5000`

---

## Dataset

**Source:** Walmart Store Sales  
**Size:** 6,435 rows × 8 columns  
**Stores:** 45 | **Period:** February 2010 – October 2012

| Column | Description |
|--------|-------------|
| `store` | Store ID (1 to 45) |
| `date` | Week ending date |
| `weekly_sales` | Total sales for that store that week (USD) |
| `holiday_flag` | 1 = holiday week, 0 = normal week |
| `temperature` | Temperature at store location (°F) |
| `fuel_price` | Fuel price that week |
| `cpi` | Consumer Price Index — measures inflation |
| `unemployment` | Local unemployment rate |

---

## Project Structure

```
walmart-sales-prediction-ml/
│
├── walmart_sales.ipynb           # complete notebook (155 cells)
├── Walmart_Sales.csv             # raw dataset
├── app.py                        # Flask backend
├── templates/
│   └── index.html                # prediction web form
├── RandomForestRegressor.pkl     # saved ML model
├── scaler.pkl                    # saved StandardScaler
└── README.md
```

---

## Notebook Walkthrough

### 1. Data Cleaning
- No null values or duplicate rows found
- Converted `date` from string (`dd-mm-yyyy`) to `datetime64`
- Standardised all column names to lowercase

### 2. Exploratory Data Analysis (EDA)

| Finding | Value |
|---------|-------|
| Highest single week | 24 Dec 2010 — $80.9M |
| Holiday week avg sales | $1,122,888 (+8% over normal) |
| Normal week avg sales | $1,041,256 |
| Best month (total) | July — $650M |
| Worst month | January — $332M |
| Best temperature range | 60–80°F |
| Best fuel price range | $3.6 – $4.0 |

**Plots produced:**
- Bar chart: sales by temperature group
- Bar chart: sales by month
- Pie chart: sales by fuel price group
- Bar chart: top 5 highest sales weeks
- Line chart: weekly sales over time (2010–2012)
- Heatmap: correlation matrix
- Pairplot: all numerical features
- Boxplots and histograms per column
- Line chart: sales by store

### 3. Feature Engineering

| Feature | How created |
|---------|-------------|
| `temp_group` | Temperature bucketed into 10 ranges |
| `fuel_group` | Fuel price bucketed into 5 ranges |
| `month` | Extracted from date |
| `year` | Extracted from date |
| `lag1` | Previous week's sales — strongest predictor |

### 4. Model Training

Features used: `store`, `holiday_flag`, `temperature`, `cpi`, `unemployment`, `month`, `year`, `lag1`

Train/test split: 75% / 25% | Scaling: `StandardScaler`

| Model | Train R² | Test R² |
|-------|----------|---------|
| GradientBoostingRegressor | 0.9566 | **0.9399** |
| RandomForestRegressor | 0.9902 | 0.9296 |
| KNeighborsRegressor | 0.9479 | 0.9214 |
| DecisionTreeRegressor | 1.0000 | 0.8901 ⚠️ overfit |
| LinearRegression | 0.8961 | 0.8895 |
| AdaBoostRegressor | 0.7537 | 0.7486 |

> DecisionTree scored 1.0 on training — classic overfitting.

### 5. Hyperparameter Tuning

`RandomizedSearchCV` — 5-fold CV, 20 iterations on top 3 models.

```
RandomForest      → n_estimators=100, max_depth=15, min_samples_split=2    CV: 0.9268
GradientBoosting  → n_estimators=50,  max_depth=5,  learning_rate=0.1      CV: 0.9252
KNeighbors        → n_neighbors=5, weights='distance'                       CV: 0.9182
```

**Final RandomForest metrics:**
```
MAE  : $79,907
RMSE : $151,053
R²   : 0.9293
```

### 6. Time Series Forecasting

Data: weekly sales aggregated across all 45 stores (143 data points)

| Method | Result |
|--------|--------|
| STL Decomposition | Trend + seasonality + residual extracted |
| ADF Test | p = 2.69e-07 → stationary, no differencing needed |
| ARIMA | auto_arima → ARIMA(2,0,0), AIC = 4829.253 |
| SARIMA | SARIMAX(2,0,2)(2,0,2,52) — underperformed (too little data for 52-week terms) |
| Prophet | Best time series model — captures holiday spikes + yearly seasonality |

### 7. Prophet Cross Validation

```python
df_cv = cross_validation(m, initial='365 days', period='90 days', horizon='90 days')
df_p  = performance_metrics(df_cv)
```

Outputs: MAPE across horizon, 6-month future forecast, trend + seasonality components.

---

## Flask Web App

### How it works

```
User fills form  →  app.py receives POST request
→  loads scaler.pkl + RandomForestRegressor.pkl
→  scales input  →  model.predict()  →  shows result in browser
```

### Run locally

```bash
# 1. clone the repo
git clone https://github.com/your-username/walmart-sales-prediction-ml.git
cd walmart-sales-prediction-ml

# 2. install dependencies
pip install flask pandas numpy scikit-learn

# 3. run the app
python app.py

# 4. open in browser
http://127.0.0.1:5000
```

### Input fields

| Field | Example |
|-------|---------|
| Store no | 1 |
| Holiday week | Holiday Week / Normal Week |
| Temperature | 39.93 |
| CPI | 211.28 |
| Unemployment | 8.10 |
| Month | 2 |
| Year | 2010 |
| Previous week sales (lag1) | 1641957.44 |

**Output:** `Predicted Sales: $1,626,585.68`

---

## Install All Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima prophet plotly flask
```

---

## Tech Stack

| Category | Libraries / Tools |
|----------|-------------------|
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn |
| Time Series | statsmodels, pmdarima, prophet |
| Web App | Flask, HTML, CSS |
| Model Saving | pickle |

---

## Conclusion

| Item | Finding |
|------|---------|
| Best ML model | GradientBoostingRegressor (R² = 0.94) |
| Deployed model | RandomForestRegressor (R² = 0.93) |
| Best time series model | Prophet |
| Strongest features | `lag1`, `store` |
| Holiday impact | +8% average sales vs normal weeks |
| Peak sales week | 24 Dec 2010 — $80.9M |
| Peak month (total) | July |

ML models outperform ARIMA and SARIMA on this dataset because weekly sales are driven by many factors — store identity, previous week sales, holidays, CPI — not just time patterns alone. GradientBoosting and RandomForest leverage all these features together, giving them a major accuracy advantage over pure time series approaches.