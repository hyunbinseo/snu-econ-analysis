# %% [markdown]
# # 경제 분석 및 예측과 데이터 지능 실습2: Prophet, NeuralProphet
#
# 본 실습은 시계열 예측을 위한 데이터 전처리와 대표적인 모듈 "Prophet", "NeuralProphet" 활용법을 다루고 있습니다.
#
# References:
# - [Modern Time Series Forecasting Techniques](https://medium.com/dataman-in-ai/mastering-time-series-forecasting-from-classical-foundations-to-cutting-edge-applications-0-1b0ac3da3188)
# - [Introduction to Statistical Learning](https://www.statlearning.com/)
# - [Prophet](https://facebook.github.io/prophet/)
# - [Neuralprophet](https://neuralprophet.com/)
# - [pmdarima](https://alkaline-ml.com/pmdarima/)

# %%
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot
from sklearn import metrics

logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# %% [markdown]
# ### Prophet
#
# Meta에서 개발한 시계열 예측 라이브러리로, 비즈니스 환경에서 복잡한 패턴을 효과적으로 모델링하기 위해 고안되어 널리 활용되고 있습니다.
#
# Prophet은 Generalized Additive Model (GAM) 기반으로 작동하는데, 이는 각 구성요소를 가법적으로 결합하여 전체 시계열을 설명하는 모델입니다.
#
# Prophet의 모델의 표현:
#
# $Y(t)=T(t)+S(t)+H(t)+ϵ(t)$
#
# - T: 트렌드
# - S: 계절성
# - H: 휴일
# - $\epsilon$ : 오차
#

# %%
path = "../datasets"
data = pd.read_csv(path + "/daily-website-visitors.csv", thousands=",")

data["Date"] = pd.to_datetime(data["Date"])
data = data[["First.Time.Visits", "Date"]]
data.columns = ["y", "ds"]
data["y"] = pd.to_numeric(data["y"], errors="coerce")
data = data[data["ds"] >= pd.to_datetime("2017-01-01")]
data = data.sort_values(by="ds")
data

# %%
plt.plot(data["ds"], data["y"])
plt.xlabel("date")
plt.ylabel("Count")
plt.show()

# %%
train_len = int(data.shape[0] * 0.85)
train = data.iloc[:train_len, :]
test = data.iloc[train_len:, :]
[train_len, len(test)]
# [1127, 200]

# %%
m = Prophet()
m.add_country_holidays(country_name="US")
m.fit(train)

future = m.make_future_dataframe(periods=len(test), freq="d")
print(future.tail())

forecast = m.predict(future)
forecast.tail()

# %%
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

# %%
mape = metrics.mean_absolute_percentage_error(
    list(test["y"]), list(forecast.loc[train_len:, "yhat"])
)
mae = metrics.mean_absolute_error(
    list(test["y"]), list(forecast.loc[train_len:, "yhat"])
)
mse = metrics.mean_squared_error(
    list(test["y"]), list(forecast.loc[train_len:, "yhat"])
)

print(f" mape: {mape}")
print(f"mae : {mae}")
print(f"mse : {mse}")

# %%
df_cv = cross_validation(m, initial="730 days", period="180 days", horizon="365 days")
df_cv.head()

# %%
m_perf = performance_metrics(df_cv)
m_perf.head()

# %% [markdown]
# ### 하이퍼 패러미터 찾기

# %%
# Define hyper-parameter grids
changepoint_prior_scale = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5]
seasonality_prior_scale = [1, 5, 10, 15]  # default 10
seasonality_model = ["additive", "multiplicative"]

results = []
iteration = 1

# Loop over all combinations of hyper-parameters
for sm in seasonality_model:
    for s in seasonality_prior_scale:
        for cp in changepoint_prior_scale:
            m = Prophet(
                seasonality_mode=sm,
                seasonality_prior_scale=s,
                changepoint_prior_scale=cp,
            )
            m.add_country_holidays(country_name="US")
            model = m.fit(train)
            future = model.make_future_dataframe(periods=len(test), freq="D")
            forecast = model.predict(future)
            # Compute evaluation metrics on the test set;
            # assume 'train_len' holds the length of the training set
            mape = metrics.mean_absolute_percentage_error(
                list(test["y"]), list(forecast.loc[train_len:, "yhat"])
            )
            mae = metrics.mean_absolute_error(
                list(test["y"]), list(forecast.loc[train_len:, "yhat"])
            )
            mse = metrics.mean_squared_error(
                list(test["y"]), list(forecast.loc[train_len:, "yhat"])
            )
            print(f"Iteration {iteration} -- mape: {mape}")
            results.append([iteration, sm, s, cp, mape, mae, mse])
            iteration += 1

# Convert results list to a DataFrame
results = pd.DataFrame(
    results,
    columns=[
        "iteration",
        "seasonality_mode",
        "seasonality_prior_scale",
        "changepoint_prior_scale",
        "mape",
        "mae",
        "mse",
    ],
)

results.head()

# %%
# Select the best model (first record in results) for diagnostics
sm_best = results.loc[0, "seasonality_mode"]
sp_best = results.loc[0, "seasonality_prior_scale"]
cp_best = results.loc[0, "changepoint_prior_scale"]

m1 = Prophet(
    seasonality_mode=sm_best,
    seasonality_prior_scale=sp_best,
    changepoint_range=cp_best,
)
m1.add_country_holidays(country_name="US")
m1.fit(train)
future = m1.make_future_dataframe(periods=len(test), freq="D")
forecast = m1.predict(future)

m1_cv = cross_validation(m1, initial="100 days", period="180 days", horizon="365 days")
m1_perf = performance_metrics(m1_cv)
print(m1_perf.head())

fig = m1.plot_components(forecast)


# %% [markdown]
# ### Neuralprophet
#
# NeuralProphet는 Prophet 모델의 확장판으로, 딥러닝 모듈을 도입하여 보다 복잡한 시계열 패턴을 모델링할 수 있습니다.
#
# 모델 구성:
# $$
# Y(t) = T(t) + S(t) + E(t) + A(t) + L(t) + F(t)
# $$
#
# - $T(t)$: 시계열의 추세(Trend)
# - $S(t)$: 계절성(Seasonality)
# - $E(t)$: 이벤트(Event) 효과
# - $A(t)$: 자동회귀(Autoregressive) 성분
# - $L(t)$: 레벨(Level) 변화
# - $F(t)$: 잔차(Residual) 또는 기타 효과

# %%
data = pd.read_csv(path + "/bike_sharing_daily.csv")
data["ds"] = pd.to_datetime(data["dteday"])
data.tail()

# %%
df = data[["ds", "cnt"]]
df.columns = ["ds", "y"]

# %%
plt.plot(df["ds"], df["y"])
plt.xlabel("date")
plt.ylabel("Count")
plt.show()

# %%
from neuralprophet import NeuralProphet, set_log_level

set_log_level("ERROR")

m = NeuralProphet()
metrics = m.fit(df)

df_future = m.make_future_dataframe(
    df,
    n_historic_predictions=True,  # include entire history
    periods=365,
)

forecast = m.predict(df_future)

m.plot(forecast)

# %% [markdown]
# ### Only Trend without changepoint

# %% [markdown]
# ### Trend without change points

# %%
m = NeuralProphet(
    n_changepoints=0,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)

df_future = m.make_future_dataframe(df, periods=365, n_historic_predictions=True)
m.set_plotting_backend("matplotlib")
df_train, df_test = m.split_df(df, valid_p=0.2)
metrics = m.fit(df_train, validation_df=df_test, progress="bar")
forecast = m.predict(df_future)
m.plot(forecast)

# %% [markdown]
# ### Trend without changepoint + Seasonality

# %%
m = NeuralProphet(
    n_changepoints=0,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)

df_future = m.make_future_dataframe(df, periods=365, n_historic_predictions=True)
m.set_plotting_backend("matplotlib")
df_train, df_test = m.split_df(df, valid_p=0.2)
metrics = m.fit(df_train, validation_df=df_test, progress="bar")
forecast = m.predict(df_future)
m.plot(forecast)

# %%
m.plot_parameters(components=["trend", "seasonality"])

# %% [markdown]
# ### Trend **with** changepoint + Seasonality

# %%
m = NeuralProphet(
    n_changepoints=10,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)

df_future = m.make_future_dataframe(df, periods=365, n_historic_predictions=True)
m.set_plotting_backend("matplotlib")
df_train, df_test = m.split_df(df, valid_p=0.2)
metrics = m.fit(df_train, validation_df=df_test, progress="bar")
forecast = m.predict(df_future)
m.plot(forecast)

# %%
m.plot_parameters(components=["trend", "seasonality"])

# %% [markdown]
# ### Trend **with** changepoint + Seasonality + Events

# %%
m = NeuralProphet(
    n_changepoints=10,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)

m = m.add_country_holidays("US")

df_future = m.make_future_dataframe(df, periods=365, n_historic_predictions=True)
m.set_plotting_backend("matplotlib")
df_train, df_test = m.split_df(df, valid_p=0.2)
metrics = m.fit(df_train, validation_df=df_test, progress="bar")
forecast = m.predict(df_future)
m.plot(forecast)

# %%
m.plot_parameters(components=["trend", "seasonality"])

# %% [markdown]
# ### Trend **with** changepoint + Seasonality + Events + AR-net

# %%
m = NeuralProphet(
    n_changepoints=10,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    n_lags=10,
)

m = m.add_country_holidays("US")

df_future = m.make_future_dataframe(df, periods=365, n_historic_predictions=True)
m.set_plotting_backend("matplotlib")
df_train, df_test = m.split_df(df, valid_p=0.2)
metrics = m.fit(df_train, validation_df=df_test, progress="bar")
forecast = m.predict(df_future)
m.plot(forecast)

# %%
m.plot_parameters(components=["trend", "seasonality", "autoregression"])

# %% [markdown]
# ### Trend **with** changepoint + Seasonality + Events + AR-net + lagged regressors + future known regressors

# %%
df2 = data[["ds", "cnt", "temp", "casual"]]
df2.columns = ["ds", "y", "temperature", "casual"]
df2.tail()

# %%
fig = df2.plot(x="ds", y="y", figsize=(10, 6))
df2.plot("ds", "temperature", secondary_y=True, ax=fig)

# %%
fig = df2.plot(x="ds", y="y", figsize=(10, 6))
df2.plot("ds", "casual", secondary_y=True, ax=fig)

# %%
m = NeuralProphet(
    n_changepoints=10,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    n_lags=10,
    n_forecasts=50,
)

m = m.add_country_holidays("US")
m = m.add_lagged_regressor("casual", n_lags=2)
m = m.add_future_regressor("temperature")

m.set_plotting_backend("matplotlib")
df2_train, df2_test = m.split_df(df2, valid_p=0.2)
metrics = m.fit(df2_train, validation_df=df2_test, progress="bar")
metrics.tail()

# %%
future_dates = pd.date_range(
    start=df2["ds"].max() + pd.Timedelta(days=1), periods=50, freq="D"
)
future_regressors = pd.DataFrame({"ds": future_dates})
future_regressors["temperature"] = df2["temperature"].iloc[-1]

future = m.make_future_dataframe(
    df=df2, periods=50, n_historic_predictions=True, regressors_df=future_regressors
)

forecast = m.predict(future)

fig_forecast = m.plot(forecast)

# %%
m.plot_parameters(
    components=[
        "trend",
        "seasonality",
        "autoregression",
        "lagged_regressors",
        "future_regressors",
    ]
)

# %% [markdown]
# ## Comparison with pmdarima
#
# neuralprophet으로 추세, 계절성을 잡고 ARIMA를 통해 나머지를 적합하는 접근 방법도 가능합니다.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from neuralprophet import NeuralProphet
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

# -------------------------------
# 1. Baseline NP Forecast
# -------------------------------
# Initialize and configure the NeuralProphet model
m = NeuralProphet(
    n_changepoints=10,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    n_forecasts=1,
)
m = m.add_country_holidays("US")
m.set_plotting_backend("matplotlib")

# Split the data into training and test sets
df_train, df_test = m.split_df(df, valid_p=0.2)

# Fit the model on training data
metrics = m.fit(df_train, validation_df=df_test, progress="bar")
print("Baseline NP Model Metrics:")
print(metrics.tail())

# Forecast NP for the test period
df_future_test = m.make_future_dataframe(
    df_train, periods=len(df_test), n_historic_predictions=False
)
forecast_np = m.predict(df_future_test)

# Rename the forecast column for clarity (baseline NP forecast)
df_np = forecast_np[["ds", "yhat1"]].copy()
df_np.rename(columns={"yhat1": "y_np"}, inplace=True)

# -------------------------------
# 2. Hybrid NP + ARIMA Forecast
# -------------------------------
# Compute NP predictions on the training set and calculate residuals
df_train_pred = m.predict(df_train)
residuals = df_train["y"].values - df_train_pred["yhat1"].values

# Fit an auto-ARIMA model to the residuals (non-seasonal)
model_arima = pm.auto_arima(residuals, seasonal=False)

# Forecast residuals for the test period
residuals_forecast = model_arima.predict(n_periods=len(df_test))

# Create a hybrid forecast by adding ARIMA's residual forecast to NP's forecast
df_hybrid = df_np.copy()
df_hybrid["y_arima"] = df_hybrid["y_np"] + residuals_forecast

# -------------------------------
# 3. NP with AR-net (n_lags=10)
# -------------------------------
# Initialize NP with autoregression (using 10 lagged terms)
m_ar = NeuralProphet(
    n_changepoints=10,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    n_lags=10,
    n_forecasts=1,
)
m_ar = m_ar.add_country_holidays("US")
m_ar.set_plotting_backend("matplotlib")

# Split the data and fit the model
df_train_ar, df_test_ar = m_ar.split_df(df, valid_p=0.2)
metrics_ar = m_ar.fit(df_train_ar, validation_df=df_test_ar, progress="bar")
print("NP with AR-net Model Metrics:")
print(metrics_ar.tail())

# Predict using NP with AR-net on the test set
forecast_arnet = m_ar.predict(df_test_ar)
# Ensure that the forecast dates are aligned with the test set dates
df_test_ar_sorted = df_test_ar.sort_values("ds").reset_index(drop=True)
valid_length = len(forecast_arnet)
df_arnet = pd.DataFrame(
    {
        "ds": pd.to_datetime(df_test_ar_sorted["ds"].values[-valid_length:]),
        "y_arnet": forecast_arnet["yhat1"].values,
    }
)

# -------------------------------
# 4. Merge all forecasts with actual test values
# -------------------------------
# Prepare actual test values
df_actual = df_test[["ds", "y"]].copy()
df_actual.rename(columns={"y": "y_actual"}, inplace=True)
df_actual["ds"] = pd.to_datetime(df_actual["ds"])

# Merge forecasts from baseline NP, the hybrid NP+ARIMA, and NP with AR-net
df_merged = pd.merge(df_np, df_actual, on="ds", how="outer")
df_merged = pd.merge(df_merged, df_hybrid[["ds", "y_arima"]], on="ds", how="outer")
df_merged = pd.merge(df_merged, df_arnet, on="ds", how="outer")
df_merged.sort_values("ds", inplace=True, ignore_index=True)


# -------------------------------
# 5. Compute error metrics for each forecast
# -------------------------------
def compute_metrics(df, y_true_col, y_pred_col):
    # Compare rows where both actual and predicted values are available
    df_valid = df.dropna(subset=[y_true_col, y_pred_col])
    y_true = df_valid[y_true_col].values
    y_pred = df_valid[y_pred_col].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mape


mae_np, rmse_np, mape_np = compute_metrics(df_merged, "y_actual", "y_np")
mae_hybrid, rmse_hybrid, mape_hybrid = compute_metrics(df_merged, "y_actual", "y_arima")
mae_arnet, rmse_arnet, mape_arnet = compute_metrics(df_merged, "y_actual", "y_arnet")

print("\nError Metrics on Test Data:")
print("Baseline NP Forecast:")
print(f"  MAE: {mae_np:.2f}, RMSE: {rmse_np:.2f}, MAPE: {mape_np:.2%}")
print("Hybrid Forecast (NP + ARIMA):")
print(f"  MAE: {mae_hybrid:.2f}, RMSE: {rmse_hybrid:.2f}, MAPE: {mape_hybrid:.2%}")
print("NP with AR-net (n_lags=10):")
print(f"  MAE: {mae_arnet:.2f}, RMSE: {rmse_arnet:.2f}, MAPE: {mape_arnet:.2%}")

metrics_df = pd.DataFrame(
    {
        "Forecasting Method": ["Baseline NP", "Hybrid (NP+ARIMA)", "NP with AR-net"],
        "MAE": [mae_np, mae_hybrid, mae_arnet],
        "RMSE": [rmse_np, rmse_hybrid, rmse_arnet],
        "MAPE": [mape_np, mape_hybrid, mape_arnet],
    }
)
print("\nComparison of Error Metrics:")
print(metrics_df)

# -------------------------------
# 6. Plot the Forecasts
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(
    df_merged["ds"], df_merged["y_actual"], label="Actual Test Data", color="black"
)
plt.plot(df_merged["ds"], df_merged["y_np"], label="NP Forecast", color="blue")
plt.plot(df_merged["ds"], df_merged["y_arima"], label="Hybrid (NP+ARIMA)", color="red")
plt.plot(
    df_merged["ds"],
    df_merged["y_arnet"],
    label="NP with AR-net (n_lags=10)",
    color="green",
)
plt.xlabel("Date")
plt.ylabel("y")
plt.title("Test Period Forecast Comparison")
plt.legend()
plt.show()


# %%
plt.figure(figsize=(12, 6))
plt.plot(
    df_merged["ds"], df_merged["y_actual"], label="Actual Test Data", color="black"
)
plt.plot(df_merged["ds"], df_merged["y_np"], label="NP Forecast", color="blue")
plt.xlabel("Date")
plt.ylabel("y")
plt.title("Test Period Forecast Comparison")
plt.legend()
plt.show()

# %%
