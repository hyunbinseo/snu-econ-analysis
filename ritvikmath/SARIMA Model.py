# %%
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

register_matplotlib_converters()
from time import time

# %% [markdown]
# # Catfish Sales Data


# %%
def parser(s):
    return datetime.strptime(s, "%Y-%m-%d")


# %%
# read data
catfish_sales = pd.read_csv(
    "catfish.csv", parse_dates=[0], index_col=0, squeeze=True, date_parser=parser
)

# %%
# infer the frequency of the data
catfish_sales = catfish_sales.asfreq(pd.infer_freq(catfish_sales.index))

# %%
start_date = datetime(1996, 1, 1)
end_date = datetime(2000, 1, 1)
lim_catfish_sales = catfish_sales[start_date:end_date]

# %%
plt.figure(figsize=(10, 4))
plt.plot(lim_catfish_sales)
plt.title("Catfish Sales in 1000s of Pounds", fontsize=20)
plt.ylabel("Sales", fontsize=16)
for year in range(start_date.year, end_date.year):
    plt.axvline(
        pd.to_datetime(str(year) + "-01-01"), color="k", linestyle="--", alpha=0.2
    )

# %% [markdown]
# ## Remove the trend

# %%
first_diff = lim_catfish_sales.diff()[1:]

# %%
plt.figure(figsize=(10, 4))
plt.plot(first_diff)
plt.title("Catfish Sales in 1000s of Pounds", fontsize=20)
plt.ylabel("Sales", fontsize=16)
for year in range(start_date.year, end_date.year):
    plt.axvline(
        pd.to_datetime(str(year) + "-01-01"), color="k", linestyle="--", alpha=0.2
    )
plt.axhline(0, color="k", linestyle="--", alpha=0.2)

# %% [markdown]
# # ACF

# %%
acf_vals = acf(first_diff)
num_lags = 20
plt.bar(range(num_lags), acf_vals[:num_lags])

# %% [markdown]
# ## Based on ACF, we should start with a seasonal MA process

# %% [markdown]
# # PACF

# %%
pacf_vals = pacf(first_diff)
num_lags = 15
plt.bar(range(num_lags), pacf_vals[:num_lags])

# %% [markdown]
# ## Based on PACF, we should start with a seasonal AR process

# %% [markdown]
# # Get training and testing sets

# %%
train_end = datetime(1999, 7, 1)
test_end = datetime(2000, 1, 1)

train_data = lim_catfish_sales[:train_end]
test_data = lim_catfish_sales[train_end + timedelta(days=1) : test_end]

# %% [markdown]
# # Fit the SARIMA Model

# %%
my_order = (0, 1, 0)
my_seasonal_order = (1, 0, 1, 12)
# define model
model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)

# %%
# fit the model
start = time()
model_fit = model.fit()
end = time()
print("Model Fitting Time:", end - start)

# %%
# summary of the model
print(model_fit.summary())

# %%
# get the predictions and residuals
predictions = model_fit.forecast(len(test_data))
predictions = pd.Series(predictions, index=test_data.index)
residuals = test_data - predictions

# %%
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.axhline(0, linestyle="--", color="k")
plt.title("Residuals from SARIMA Model", fontsize=20)
plt.ylabel("Error", fontsize=16)

# %%
plt.figure(figsize=(10, 4))

plt.plot(lim_catfish_sales)
plt.plot(predictions)

plt.legend(("Data", "Predictions"), fontsize=16)

plt.title("Catfish Sales in 1000s of Pounds", fontsize=20)
plt.ylabel("Production", fontsize=16)
for year in range(start_date.year, end_date.year):
    plt.axvline(
        pd.to_datetime(str(year) + "-01-01"), color="k", linestyle="--", alpha=0.2
    )

# %%
print("Mean Absolute Percent Error:", round(np.mean(abs(residuals / test_data)), 4))

# %%
print("Root Mean Squared Error:", np.sqrt(np.mean(residuals**2)))

# %% [markdown]
# # Using the Rolling Forecast Origin

# %%
rolling_predictions = test_data.copy()
for train_end in test_data.index:
    train_data = lim_catfish_sales[: train_end - timedelta(days=1)]
    model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit()

    pred = model_fit.forecast()
    rolling_predictions[train_end] = pred

# %%
rolling_residuals = test_data - rolling_predictions

# %%
plt.figure(figsize=(10, 4))
plt.plot(rolling_residuals)
plt.axhline(0, linestyle="--", color="k")
plt.title("Rolling Forecast Residuals from SARIMA Model", fontsize=20)
plt.ylabel("Error", fontsize=16)

# %%
plt.figure(figsize=(10, 4))

plt.plot(lim_catfish_sales)
plt.plot(rolling_predictions)

plt.legend(("Data", "Predictions"), fontsize=16)

plt.title("Catfish Sales in 1000s of Pounds", fontsize=20)
plt.ylabel("Production", fontsize=16)
for year in range(start_date.year, end_date.year):
    plt.axvline(
        pd.to_datetime(str(year) + "-01-01"), color="k", linestyle="--", alpha=0.2
    )

# %%
print(
    "Mean Absolute Percent Error:",
    round(np.mean(abs(rolling_residuals / test_data)), 4),
)

# %%
print("Root Mean Squared Error:", np.sqrt(np.mean(rolling_residuals**2)))

# %%
