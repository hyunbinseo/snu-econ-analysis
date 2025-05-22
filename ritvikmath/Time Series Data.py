# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

register_matplotlib_converters()

# %% [markdown]
# # Ice Cream Production Data

# %%
# read data
df_ice_cream = pd.read_csv("ice_cream.csv")

# %%
df_ice_cream.head()

# %%
# rename columns to something more understandable
df_ice_cream.rename(columns={"DATE": "date", "IPN31152N": "production"}, inplace=True)

# %%
# convert date column to datetime type
df_ice_cream["date"] = pd.to_datetime(df_ice_cream.date)

# %%
# set date as index
df_ice_cream.set_index("date", inplace=True)

# %%
# just get data from 2010 onwards
start_date = pd.to_datetime("2010-01-01")
df_ice_cream = df_ice_cream[start_date:]

# %%
# show result
df_ice_cream.head()

# %%
plt.figure(figsize=(10, 4))
plt.plot(df_ice_cream.production)
plt.title("Ice Cream Production over Time", fontsize=20)
plt.ylabel("Production", fontsize=16)
for year in range(2011, 2021):
    plt.axvline(
        pd.to_datetime(str(year) + "-01-01"), color="k", linestyle="--", alpha=0.2
    )

# %% [markdown]
# # ACF

# %%
acf_plot = plot_acf(df_ice_cream.production, lags=100)

# %% [markdown]
# ## Based on decaying ACF, we are likely dealing with an Auto Regressive process

# %% [markdown]
# # PACF

# %%
pacf_plot = plot_pacf(df_ice_cream.production)

# %% [markdown]
# ## Based on PACF, we should start with an Auto Regressive model with lags 1, 2, 3, 10, 13

# %% [markdown]
# # On stock data

# %%
import yfinance as yf

# %%
# define the ticker symbol
tickerSymbol = "SPY"

# %%
# get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# %%
# get the historical prices for this ticker
tickerDf = tickerData.history(period="1d", start="2015-1-1", end="2020-1-1")

# %%
tickerDf = tickerDf[["Close"]]

# %%
# see your data
tickerDf.head()

# %%
plt.figure(figsize=(10, 4))
plt.plot(tickerDf.Close)
plt.title("Stock Price over Time (%s)" % tickerSymbol, fontsize=20)
plt.ylabel("Price", fontsize=16)
for year in range(2015, 2021):
    plt.axvline(
        pd.to_datetime(str(year) + "-01-01"), color="k", linestyle="--", alpha=0.2
    )

# %% [markdown]
# ## Stationarity: take first difference of this series

# %%
# take first difference
first_diffs = tickerDf.Close.values[1:] - tickerDf.Close.values[:-1]
first_diffs = np.concatenate([first_diffs, [0]])

# %%
# set first difference as variable in dataframe
tickerDf["FirstDifference"] = first_diffs

# %%
tickerDf.head()

# %%
plt.figure(figsize=(10, 4))
plt.plot(tickerDf.FirstDifference)
plt.title("First Difference over Time (%s)" % tickerSymbol, fontsize=20)
plt.ylabel("Price Difference", fontsize=16)
for year in range(2015, 2021):
    plt.axvline(
        pd.to_datetime(str(year) + "-01-01"), color="k", linestyle="--", alpha=0.2
    )

# %% [markdown]
# # ACF

# %%
acf_plot = plot_acf(tickerDf.FirstDifference)

# %% [markdown]
# ## ACF isn't that informative

# %% [markdown]
# # PACF

# %%
pacf_plot = plot_pacf(tickerDf.FirstDifference)

# %% [markdown]
# ## PACF also doesn't tell us much

# %%


# %%
