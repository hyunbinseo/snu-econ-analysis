# %%
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR

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
plt.figure(figsize=(12, 6))
(ice_cream,) = plt.plot(ice_cream_heater_df["ice cream"])
(heater,) = plt.plot(ice_cream_heater_df["heater"], color="red")

for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle="--", color="k", alpha=0.5)

plt.legend(["Ice Cream", "Heater"], fontsize=16)

# %% [markdown]
# # Normalize

# %%
avgs = ice_cream_heater_df.mean()
devs = ice_cream_heater_df.std()

# %%
for col in ice_cream_heater_df.columns:
    ice_cream_heater_df[col] = (ice_cream_heater_df[col] - avgs.loc[col]) / devs.loc[
        col
    ]

# %%
plt.figure(figsize=(12, 6))
(ice_cream,) = plt.plot(ice_cream_heater_df["ice cream"])
(heater,) = plt.plot(ice_cream_heater_df["heater"], color="red")

for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle="--", color="k", alpha=0.5)

plt.axhline(0, linestyle="--", color="k", alpha=0.3)

plt.legend(["Ice Cream", "Heater"], fontsize=16)

# %% [markdown]
# # Take First Difference to Remove Trend

# %%
ice_cream_heater_df = ice_cream_heater_df.diff().dropna()

# %%
plt.figure(figsize=(12, 6))
(ice_cream,) = plt.plot(ice_cream_heater_df["ice cream"])
(heater,) = plt.plot(ice_cream_heater_df["heater"], color="red")

for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle="--", color="k", alpha=0.5)

plt.axhline(0, linestyle="--", color="k", alpha=0.3)
plt.ylabel("First Difference", fontsize=18)

plt.legend(["Ice Cream", "Heater"], fontsize=16)

# %%
plt.figure(figsize=(12, 6))
(ice_cream,) = plt.plot(ice_cream_heater_df["ice cream"])

for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle="--", color="k", alpha=0.5)

plt.axhline(0, linestyle="--", color="k", alpha=0.3)
plt.ylabel("First Difference", fontsize=18)

plt.legend(["Ice Cream"], fontsize=16)

# %% [markdown]
# # Remove Increasing Volatility

# %%
annual_volatility = ice_cream_heater_df.groupby(ice_cream_heater_df.index.year).std()

# %%
annual_volatility

# %%
ice_cream_heater_df["ice_cream_annual_vol"] = ice_cream_heater_df.index.map(
    lambda d: annual_volatility.loc[d.year, "ice cream"]
)
ice_cream_heater_df["heater_annual_vol"] = ice_cream_heater_df.index.map(
    lambda d: annual_volatility.loc[d.year, "heater"]
)

# %%
ice_cream_heater_df

# %%
ice_cream_heater_df["ice cream"] = (
    ice_cream_heater_df["ice cream"] / ice_cream_heater_df["ice_cream_annual_vol"]
)
ice_cream_heater_df["heater"] = (
    ice_cream_heater_df["heater"] / ice_cream_heater_df["heater_annual_vol"]
)

# %%
plt.figure(figsize=(12, 6))
(ice_cream,) = plt.plot(ice_cream_heater_df["ice cream"])

for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle="--", color="k", alpha=0.5)

plt.axhline(0, linestyle="--", color="k", alpha=0.3)
plt.ylabel("First Difference", fontsize=18)

plt.legend(["Ice Cream"], fontsize=16)

# %%
plt.figure(figsize=(12, 6))
(ice_cream,) = plt.plot(ice_cream_heater_df["ice cream"])
(heater,) = plt.plot(ice_cream_heater_df["heater"], color="red")

for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle="--", color="k", alpha=0.5)

plt.axhline(0, linestyle="--", color="k", alpha=0.3)
plt.ylabel("First Difference", fontsize=18)

plt.legend(["Ice Cream", "Heater"], fontsize=16)

# %% [markdown]
# # Remove Seasonality

# %%
month_avgs = ice_cream_heater_df.groupby(ice_cream_heater_df.index.month).mean()

# %%
print(month_avgs)

# %%
ice_cream_heater_df["ice_cream_month_avg"] = ice_cream_heater_df.index.map(
    lambda d: month_avgs.loc[d.month, "ice cream"]
)
ice_cream_heater_df["heater_month_avg"] = ice_cream_heater_df.index.map(
    lambda d: month_avgs.loc[d.month, "heater"]
)

# %%
ice_cream_heater_df

# %%
ice_cream_heater_df["ice cream"] = (
    ice_cream_heater_df["ice cream"] - ice_cream_heater_df["ice_cream_month_avg"]
)
ice_cream_heater_df["heater"] = (
    ice_cream_heater_df["heater"] - ice_cream_heater_df["heater_month_avg"]
)

# %%
ice_cream_heater_df

# %%
plt.figure(figsize=(12, 6))
(ice_cream,) = plt.plot(ice_cream_heater_df["ice cream"])

for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle="--", color="k", alpha=0.5)

plt.axhline(0, linestyle="--", color="k", alpha=0.3)
plt.ylabel("First Difference", fontsize=18)

plt.legend(["Ice Cream"], fontsize=16)

# %%
plt.figure(figsize=(12, 6))
(ice_cream,) = plt.plot(ice_cream_heater_df["ice cream"])
(heater,) = plt.plot(ice_cream_heater_df["heater"], color="red")

for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle="--", color="k", alpha=0.5)

plt.axhline(0, linestyle="--", color="k", alpha=0.3)
plt.ylabel("First Difference", fontsize=18)

plt.legend(["Ice Cream", "Heater"], fontsize=16)

# %% [markdown]
# # PACF - Heater

# %%
plot_pacf(ice_cream_heater_df["heater"])
plt.show()

# %% [markdown]
# ## So consider an AR(2)

# %% [markdown]
# # Correlation between "heater" and lagged "ice cream"

# %%
for lag in range(1, 14):
    heater_series = ice_cream_heater_df["heater"].iloc[lag:]
    lagged_ice_cream_series = ice_cream_heater_df["ice cream"].iloc[:-lag]
    print("Lag: %s" % lag)
    print(pearsonr(heater_series, lagged_ice_cream_series))
    print("------")

# %% [markdown]
# # Fit a VAR Model

# %%
ice_cream_heater_df = ice_cream_heater_df[["ice cream", "heater"]]

# %%
model = VAR(ice_cream_heater_df)

# %%
model_fit = model.fit(maxlags=13)

# %%
model_fit.summary()

# %% [markdown]
# # So our final model is:

# %% [markdown]
# $$
# \hat{h}_t = - 0.41h_{t-1} - 0.19h_{t-2} + 0.2i_{t-13}
# $$

# %%
