# %%
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller


# %%
def parser(s):
    return datetime.strptime(s, "%Y-%m-%d")


# %%
def perform_adf_test(series):
    result = adfuller(series)
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])


# %%
# get data
series = pd.read_csv(
    "catfish.csv", parse_dates=[0], index_col=0, squeeze=True, date_parser=parser
)
series = series.asfreq(pd.infer_freq(series.index))
series = series.loc[datetime(2004, 1, 1) :]
series = series.diff().diff().dropna()

# %%
# check stationarity
perform_adf_test(series)

# %%
plt.plot(series)

# %%
plot_pacf(series, lags=10)
plt.show()

# %% [markdown]
# # Either AR(1), AR(4), AR(6), or AR(10)

# %%
plt.figure(figsize=(12, 12))

ar_orders = [1, 4, 6, 10]
fitted_model_dict = {}

for idx, ar_order in enumerate(ar_orders):
    # create AR(p) model
    ar_model = ARMA(series, order=(ar_order, 0))
    ar_model_fit = ar_model.fit()
    fitted_model_dict[ar_order] = ar_model_fit
    plt.subplot(4, 1, idx + 1)
    plt.plot(series)
    plt.plot(ar_model_fit.fittedvalues)
    plt.title("AR(%s) Fit" % ar_order, fontsize=16)

plt.tight_layout()

# %% [markdown]
# # Each model has:
#
# ## a log likelihood ($l$)
# ## a number of parameters ($k$)
# ## a number of samples used for fitting ($n$)

# %% [markdown]
# # AIC = 2$k$ - 2$l$
#
# ## Lower AIC via higher log likelihood or less parameters
#
# # BIC = $\ln(n)k$ - 2$l$
#
# ## Lower BIC via higher log likelihood or less parameters or less samples used in fitting

# %%
# AIC comparison
for ar_order in ar_orders:
    print("AIC for AR(%s): %s" % (ar_order, fitted_model_dict[ar_order].aic))

# %% [markdown]
# ## Based on AIC criteria, pick AR(6)

# %%
# BIC comparison
for ar_order in ar_orders:
    print("BIC for AR(%s): %s" % (ar_order, fitted_model_dict[ar_order].bic))

# %% [markdown]
# ## Based on BIC criteria, pick AR(6)

# %%
