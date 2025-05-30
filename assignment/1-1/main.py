# %% 1. 데이터 수집 및 전처리
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules import get_historical_precipitation
from neuralprophet import NeuralProphet
from prophet import Prophet
from skforecast.metrics import crps_from_quantiles
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100


date_column = "date"
value_column = "rental"

df = pd.read_csv("data_raw.csv")
df[date_column] = pd.to_datetime(df[date_column])
df = df.sort_values(date_column).reset_index(drop=True)

date_range = pd.date_range(
    start=df[date_column].min(),
    end=df[date_column].max(),
    freq="D",
)

df_filled = pd.DataFrame({date_column: date_range}).merge(
    df, on=date_column, how="left"
)

df_filled[value_column] = df_filled[value_column].interpolate()
df_filled.to_csv("data_filled.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(df_filled[date_column], df_filled[value_column], linewidth=1)
plt.title("Rental Data Over Time")
plt.xlabel("Date")
plt.ylabel("Rental")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

df_filled[f"{value_column}_log"] = np.log(df_filled[value_column])

plt.figure(figsize=(12, 6))
plt.plot(df_filled[date_column], df_filled[f"{value_column}_log"], linewidth=1)
plt.title("Log-transformed Rental Data Over Time")
plt.xlabel("Date")
plt.ylabel("Log(Rental)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

df_prophet = df_filled[[date_column, f"{value_column}_log"]].copy()
df_prophet.columns = ["ds", "y"]

train_size = int(len(df_prophet) * 0.85)

df_train = df_prophet[:train_size]
df_test = df_prophet[train_size:]

# %% 2. Prophet 기본 모델
p_model = Prophet(interval_width=0.8)
p_model.add_country_holidays(country_name="KR")
p_model.fit(df_train)

p_dataframe = p_model.make_future_dataframe(periods=len(df_test))
p_forecast = p_model.predict(p_dataframe)
p_forecast_test = p_forecast[train_size:].copy()

p_series_actual = df_test["y"].reset_index(drop=True)
p_series_pred = p_forecast_test["yhat"].reset_index(drop=True)

print(p_forecast_test.head())

p_mae = mean_absolute_error(p_series_actual, p_series_pred)
p_rmse = np.sqrt(mean_squared_error(p_series_actual, p_series_pred))
p_mape = np.mean(np.abs((p_series_actual - p_series_pred) / p_series_actual)) * 100
p_smape = smape(p_series_actual, p_series_pred)
p_crps = np.mean(
    [
        crps_from_quantiles(
            y_true=float(p_series_actual.iloc[i]),
            pred_quantiles=np.array(
                [
                    p_forecast_test["yhat_lower"].iloc[i],
                    p_forecast_test["yhat_upper"].iloc[i],
                ]
            ),
            quantile_levels=np.array([0.1, 0.9]),
        )
        for i in range(len(p_series_actual))
    ]
)

print("\nProphet Results (Log Scale):")
print(f"MAE: {p_mae:.4f}")
print(f"RMSE: {p_rmse:.4f}")
print(f"MAPE: {p_mape:.2f}%")
print(f"sMAPE: {p_smape:.2f}%")
print(f"CRPS: {p_crps:.4f}")

plt.figure(figsize=(15, 8))
plt.plot(df_train["ds"], df_train["y"], label="Train Data", color="blue")
plt.plot(df_test["ds"], df_test["y"], label="Test Data", color="green")
plt.plot(df_test["ds"], p_series_pred, label="Predictions", color="red", linestyle="--")
plt.title("Prophet Model - Log-transformed Rental Data Forecast")
plt.xlabel("Date")
plt.ylabel("Log(Rental)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

p_model.plot(p_forecast)
plt.title("Prophet Model Forecast - Full Timeline")
plt.show()

p_model.plot_components(p_forecast)
plt.show()

# %% 5. Future Regressors 분석
# https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html

df_train2 = pd.merge(
    df_train,
    get_historical_precipitation(
        37.5665,
        126.9780,
        df_train["ds"].min().strftime("%Y-%m-%d"),
        df_train["ds"].max().strftime("%Y-%m-%d"),
    ),
    on="ds",
    how="left",
)

df_train2["precipitation"] = df_train2["precipitation"].fillna(0)

p2_model = Prophet(interval_width=0.8)
p2_model.add_country_holidays(country_name="KR")
p2_model.add_regressor("precipitation")
p2_model.fit(df_train2)

p2_dataframe = p2_model.make_future_dataframe(periods=len(df_test))
p2_dataframe = pd.merge(
    p2_dataframe,
    get_historical_precipitation(
        37.5665,
        126.9780,
        p2_dataframe["ds"].min().strftime("%Y-%m-%d"),
        p2_dataframe["ds"].max().strftime("%Y-%m-%d"),
    ),
    on="ds",
    how="left",
)

p2_forecast = p2_model.predict(p2_dataframe)
p2_forecast_test = p2_forecast[train_size:].copy()

p2_series_actual = df_test["y"].reset_index(drop=True)
p2_series_pred = p2_forecast_test["yhat"].reset_index(drop=True)

p2_mape = np.mean(np.abs((p2_series_actual - p2_series_pred) / p2_series_actual)) * 100

print("\nProphet Results with Precipitation (Log Scale):")
print(f"MAPE: {p2_mape:.2f}%")

p2_model.plot(p2_forecast)
plt.title("Prophet Model Forecast with Precipitation - Full Timeline")
plt.show()


# %% 3. Change Point 민감도 분석
p_cp_scales = [0.01, 0.1, 0.5]
p_cp_results = []
p_cp_models = {}
p_cp_forecasts = {}

for scale in p_cp_scales:
    print(f"\nTraining Prophet model with changepoint_prior_scale={scale}")

    p_model = Prophet(changepoint_prior_scale=scale)
    p_model.add_country_holidays(country_name="KR")
    p_model.fit(df_train)

    p_forecast = p_model.predict(p_model.make_future_dataframe(periods=len(df_test)))
    p_forecast_test = p_forecast[train_size:]

    p_series_pred = p_forecast_test["yhat"].reset_index(drop=True)

    p_mae = mean_absolute_error(p_series_actual, p_series_pred)
    p_mape = np.mean(np.abs((p_series_actual - p_series_pred) / p_series_actual)) * 100

    p_cp_results.append(
        {
            "changepoint_prior_scale": scale,
            "MAE": p_mae,
            "MAPE": p_mape,
        }
    )

    p_cp_models[scale] = p_model
    p_cp_forecasts[scale] = p_forecast

df_cp_results = pd.DataFrame(p_cp_results)

print("\nChangepoint Prior Scale Comparison")
print(df_cp_results.to_string(index=False))

plt.figure(figsize=(15, 10))

for i, scale in enumerate(p_cp_scales):
    plt.subplot(2, 2, i + 1)

    p_forecast_test = p_cp_forecasts[scale][train_size:]
    p_series_pred = p_forecast_test["yhat"].reset_index(drop=True)

    plt.plot(df_train["ds"], df_train["y"], label="Train Data", color="blue", alpha=0.7)
    plt.plot(df_test["ds"], df_test["y"], label="Test Data", color="green", alpha=0.8)
    plt.plot(
        df_test["ds"],
        p_series_pred,
        label="Predictions",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    p_mae_current = df_cp_results[df_cp_results["changepoint_prior_scale"] == scale][
        "MAE"
    ].iloc[0]

    p_mape_current = df_cp_results[df_cp_results["changepoint_prior_scale"] == scale][
        "MAPE"
    ].iloc[0]

    plt.title(
        f"Changepoint Scale: {scale}\nMAE: {p_mae_current:.4f}, MAPE: {p_mape_current:.2f}%"
    )
    plt.xlabel("Date")
    plt.ylabel("Log(Rental)")
    plt.legend(fontsize=8)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.bar(
    df_cp_results["changepoint_prior_scale"].astype(str),
    df_cp_results["MAPE"],
    color=["skyblue", "lightcoral", "lightgreen"],
    alpha=0.7,
)
plt.title("MAPE Comparison by Changepoint Prior Scale")
plt.xlabel("Changepoint Prior Scale")
plt.ylabel("MAPE (%)")
plt.grid(True, alpha=0.3)

for i, (idx, row) in enumerate(df_cp_results.iterrows()):
    plt.text(
        i,
        row["MAPE"] + 0.1,
        f"{row['MAPE']:.2f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 12))

for i, scale in enumerate(p_cp_scales):
    plt.subplot(3, 1, i + 1)

    p_model = p_cp_models[scale]
    p_forecast = p_cp_forecasts[scale]

    plt.plot(
        df_prophet["ds"], df_prophet["y"], label="Actual Data", color="black", alpha=0.6
    )
    plt.plot(
        p_forecast["ds"],
        p_forecast["yhat"],
        label="Forecast",
        color="blue",
        linewidth=2,
    )
    plt.fill_between(
        p_forecast["ds"],
        p_forecast["yhat_lower"],
        p_forecast["yhat_upper"],
        alpha=0.3,
        color="blue",
        label="Confidence Interval",
    )

    p_changepoints = p_model.changepoints
    p_changepoint_effects = np.array(p_model.params["delta"].mean(axis=0))
    p_significant_changepoints = p_changepoints[np.abs(p_changepoint_effects) > 0.01]

    if len(p_significant_changepoints) > 0:
        for cp in p_significant_changepoints:
            plt.axvline(
                x=pd.to_datetime(cp),
                color="red",
                linestyle=":",
                alpha=0.8,
                linewidth=1.5,
            )

    plt.title(
        f"Prophet Forecast with Changepoints (Scale: {scale})\n"
        f"Detected Changepoints: {len(p_significant_changepoints)}"
    )
    plt.xlabel("Date")
    plt.ylabel("Log(Rental)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% 4. NeuralProphet 활용
np_model = NeuralProphet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    n_lags=10,
    n_forecasts=len(df_test),
    quantiles=[0.1, 0.9],
)

np_model.add_country_holidays(country_name="KR")
np_model.fit(df=df_train, freq="D")

np_dataframe = np_model.make_future_dataframe(
    df=df_prophet,
    n_historic_predictions=True,
)

np_forecast = np_model.predict(df=np_dataframe)
np_forecast_test = np_forecast[train_size:].copy().dropna()

np_series_actual = df_test["y"].reset_index(drop=True)
np_series_pred = np_forecast_test["yhat1"].reset_index(drop=True)

print(np_forecast_test.head())
# 'yhat1', 'yhat2', 'yhat3', 'yhat4', 'yhat5', … 'yhat384'
# 'yhat1 10.0%', 'yhat2 10.0%', 'yhat3 10.0%', … 'yhat384 10.0%'
# 'yhat1 90.0%', 'yhat2 90.0%', 'yhat3 90.0%', … 'yhat384 90.0%'

# e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, it is “3 steps old”.
# Reference https://neuralprophet.com/how-to-guides/feature-guides/collect_predictions.html

np_mae = mean_absolute_error(np_series_actual, np_series_pred)
np_rmse = np.sqrt(mean_squared_error(np_series_actual, np_series_pred))
np_mape = np.mean(np.abs((np_series_actual - np_series_pred) / np_series_actual)) * 100
np_smape = smape(np_series_actual, np_series_pred)
np_crps = np.mean(
    [
        crps_from_quantiles(
            y_true=float(np_series_actual.iloc[i]),
            pred_quantiles=np.array(
                [
                    np_forecast_test["yhat1 10.0%"].iloc[i],
                    np_forecast_test["yhat1 90.0%"].iloc[i],
                ]
            ),
            quantile_levels=np.array([0.1, 0.9]),
        )
        for i in range(len(np_series_actual))
    ]
)

print("\nNeuralProphet Results (Log Scale):")
print(f"MAE: {p_mae:.4f}")
print(f"RMSE: {p_rmse:.4f}")
print(f"MAPE: {p_mape:.2f}%")
print(f"sMAPE: {p_smape:.2f}%")
print(f"CRPS: {np_crps:.4f}")

np_model.highlight_nth_step_ahead_of_each_forecast(step_number=1)
np_model.plot(np_forecast)
