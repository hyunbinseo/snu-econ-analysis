# %%
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.arima_model import ARMA
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
start_date = datetime(2000, 1, 1)
end_date = datetime(2004, 1, 1)
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
plt.axhline(lim_catfish_sales.mean(), color="r", alpha=0.2, linestyle="--")

# %%
first_diff = lim_catfish_sales.diff()[1:]

# %%
plt.figure(figsize=(10, 4))
plt.plot(first_diff)
plt.title("First Difference of Catfish Sales", fontsize=20)
plt.ylabel("Sales", fontsize=16)
for year in range(start_date.year, end_date.year):
    plt.axvline(
        pd.to_datetime(str(year) + "-01-01"), color="k", linestyle="--", alpha=0.2
    )
plt.axhline(first_diff.mean(), color="r", alpha=0.2, linestyle="--")

# %% [markdown]
# # ACF

# %%
acf_vals = acf(first_diff)
plt.bar(range(num_lags), acf_vals[:num_lags])

# %% [markdown]
# ## Based on ACF, we should start with a MA(1) process

# %% [markdown]
# # PACF

# %%
pacf_vals = pacf(first_diff)
plt.bar(range(num_lags), pacf_vals[:num_lags])

# %% [markdown]
# ## Based on PACF, we should start with a AR(4) process

# %% [markdown]
# # Get training and testing sets

# %%
train_end = datetime(2003, 7, 1)
test_end = datetime(2004, 1, 1)

train_data = first_diff[:train_end]
test_data = first_diff[train_end + timedelta(days=1) : test_end]

# %% [markdown]
# # Fit the ARMA Model

# %%
# define model
model = ARMA(train_data, order=(4, 1))

# %%
# fit the model
start = time()
model_fit = model.fit()
end = time()
print("Model Fitting Time:", end - start)

# %%
# summary of the model
print(model_fit.summary())

# %% [markdown]
# ## So the ARMA(4,1) model is:
#
# ## $\hat{y_t} = -0.87y_{t-1} - 0.42y_{t-2} - 0.56y_{t-3} - 0.61y_{t-4} + 0.52\varepsilon_{t-1}$

# %%
# get prediction start and end dates
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

# %%
# get the predictions and residuals
predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data - predictions

# %%
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title("Residuals from AR Model", fontsize=20)
plt.ylabel("Error", fontsize=16)
plt.axhline(0, color="r", linestyle="--", alpha=0.2)

# %%
plt.figure(figsize=(10, 4))

plt.plot(test_data)
plt.plot(predictions)

plt.legend(("Data", "Predictions"), fontsize=16)

plt.title("First Difference of Catfish Sales", fontsize=20)
plt.ylabel("Sales", fontsize=16)

# %%
print("Root Mean Squared Error:", np.sqrt(np.mean(residuals**2)))

# %%
