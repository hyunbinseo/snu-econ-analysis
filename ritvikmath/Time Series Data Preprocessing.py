# %%
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# # Read the Data


# %%
def parser(s):
    return datetime.strptime(s, "%Y-%m")


# %%
ice_cream_heater_df = pd.read_csv(
    "ice_cream_vs_heater.csv",
    parse_dates=[0],
    index_col=0,
    squeeze=True,
    date_parser=parser,
)

# %%
ice_cream_heater_df = ice_cream_heater_df.asfreq(
    pd.infer_freq(ice_cream_heater_df.index)
)

# %%
heater_series = ice_cream_heater_df.heater

# %%
heater_series


# %%
def plot_series(series):
    plt.figure(figsize=(12, 6))
    plt.plot(heater_series, color="red")
    plt.ylabel('Search Frequency for "Heater"', fontsize=16)

    for year in range(2004, 2021):
        plt.axvline(datetime(year, 1, 1), linestyle="--", color="k", alpha=0.5)


# %%
plot_series(heater_series)

# %% [markdown]
# # Normalize

# %%
avg, dev = heater_series.mean(), heater_series.std()

# %%
heater_series = (heater_series - avg) / dev

# %%
plot_series(heater_series)
plt.axhline(0, linestyle="--", color="k", alpha=0.3)

# %% [markdown]
# # Take First Difference to Remove Trend

# %%
heater_series = heater_series.diff().dropna()

# %%
plot_series(heater_series)
plt.axhline(0, linestyle="--", color="k", alpha=0.3)

# %% [markdown]
# # Remove Increasing Volatility

# %%
annual_volatility = heater_series.groupby(heater_series.index.year).std()

# %%
annual_volatility

# %%
heater_annual_vol = heater_series.index.map(lambda d: annual_volatility.loc[d.year])

# %%
heater_annual_vol

# %%
heater_series = heater_series / heater_annual_vol

# %%
plot_series(heater_series)
plt.axhline(0, linestyle="--", color="k", alpha=0.3)

# %% [markdown]
# # Remove Seasonality

# %%
month_avgs = heater_series.groupby(heater_series.index.month).mean()

# %%
month_avgs

# %%
heater_month_avg = heater_series.index.map(lambda d: month_avgs.loc[d.month])

# %%
heater_month_avg

# %%
heater_series = heater_series - heater_month_avg

# %%
plot_series(heater_series)
plt.axhline(0, linestyle="--", color="k", alpha=0.3)
