# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %% 1-1 Setup
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

# %% 1-2 Prophet
p_model = Prophet()
p_model.add_country_holidays(country_name="KR")
p_model.fit(df_train)

p_forecast = p_model.predict(p_model.make_future_dataframe(periods=len(df_test)))
p_forecast_test = p_forecast[train_size:]

p_series_actual = df_test["y"].reset_index(drop=True)
p_series_pred = p_forecast_test["yhat"].reset_index(drop=True)

p_mae = mean_absolute_error(p_series_actual, p_series_pred)
p_rmse = np.sqrt(mean_squared_error(p_series_actual, p_series_pred))
p_mape = np.mean(np.abs((p_series_actual - p_series_pred) / p_series_actual)) * 100

print("\nOut-of-time Test Results (Log Scale):")
print(f"MAE: {p_mae:.4f}")
print(f"RMSE: {p_rmse:.4f}")
print(f"MAPE: {p_mape:.2f}%")

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

# %% 1-3 Prophet Changepoints
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

# %% 1-4 NeuralProphet
np_model = NeuralProphet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    n_lags=10,
    n_forecasts=len(df_test),
)

np_model.add_country_holidays(country_name="KR")
np_model.fit(df=df_train, freq="D")

np_forecast = np_model.predict(
    np_model.make_future_dataframe(
        df=df_prophet,
        n_historic_predictions=True,
    )
)

np_model.plot(np_forecast)
