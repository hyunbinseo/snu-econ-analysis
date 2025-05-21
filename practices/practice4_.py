# %% [markdown]
# # 경제 분석 및 예측과 데이터 지능 실습4: Dynamic Factor Model
#
# OECD데이터를 활용해 VAR 분석을 실습합니다. Statsmodels을 주로 다루고, 실제 데이터에 적용해 봅니다.
#
# 실습 전 터미널에 아래 코드를 실행하여 환경설정을 맞춰줍니다.
#
# conda env update -n forecasting_lecture -f environment.yml --prune

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# %%
# 1. Load & preprocess data
# --------------------------
df = pd.read_csv("../datasets/querterly_oecd_account_data.csv", parse_dates=["date"])
df["date"] = df["date"].dt.to_period("Q")

vars_to_use = ["Y", "C", "G", "I", "EX", "IM"]

# %%
df

# %% [markdown]
# ### 분기별 변화율을 계산하고 연도별 변화율로 annualize 합니다.
#
# pct_change : $\frac{x_t - x_{t-1}}{x_{t-1}}$

# %%
# compute annualized q-to-q growth by country
df_growth = (
    df.set_index(["date", "country"])[vars_to_use]
    .sort_index()
    .groupby("country")
    .pct_change()  # quarter-to-quarter % change
    .mul(400)  # annualize
    .dropna()
    .reset_index()
)

# %%
df_growth

# %%
# pivot to a wide panel (MultiIndex columns)
panel = df_growth.pivot(index="date", columns="country", values=vars_to_use)

# fill any gaps (forward/backward) rather than drop
panel = panel.interpolate().ffill().bfill()

# drop any series that ended up constant
panel = panel.loc[:, panel.std() > 0]

# NOW flatten the MultiIndex into single strings: "Y_CAN", "C_DEU", etc.
panel.columns = [f"{var}_{ctry}" for var, ctry in panel.columns]

# standardize (mean=0, std=1)
endog = (panel - panel.mean()) / panel.std()

# %%
panel

# %%
# 2. Specify & fit DynamicFactorMQ
# ---------------------------------
k_factors = 3  # number of common factors
factor_ord = 4  # VAR order for factor dynamics

model = sm.tsa.DynamicFactorMQ(
    endog,
    factors=k_factors,
    factor_orders=factor_ord,
    idiosyncratic_ar1=True,
    standardize=False,
)

# %%
results = model.fit(disp=False)
print(results.summary())

# %%
# 3. Extract & plot smoothed factors
# -----------------------------------
factors = results.factors.smoothed
plt.figure(figsize=(10, 4))
for col in factors.columns:
    plt.plot(factors.index.to_timestamp(), factors[col], label=col)
plt.title("Smoothed Common Factors")
plt.xlabel("Date")
plt.ylabel("Factor Value")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# 4. Forecasting
# --------------
fc_res = results.get_forecast(steps=16)
fc_mean = fc_res.predicted_mean
fc_ci = fc_res.conf_int()

# rename "lower Y_CAN" → "Y_CAN_lower" etc.
fc_ci.columns = [
    f"{name.split(' ', 1)[1]}_{name.split(' ', 1)[0]}" for name in fc_ci.columns
]

# %%
endog.columns

# %%
series = "Y_KOR"
plt.figure(figsize=(10, 4))
plt.plot(endog.index.to_timestamp(), endog[series], label="Observed")
plt.plot(fc_mean.index.to_timestamp(), fc_mean[series], label="Forecast")
plt.fill_between(
    fc_ci.index.to_timestamp(),
    fc_ci[f"{series}_lower"],
    fc_ci[f"{series}_upper"],
    color="gray",
    alpha=0.3,
)
plt.title(f"{series}: In-Sample & Out-of-Sample")
plt.xlabel("Date")
plt.ylabel("Standardized Growth")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# 5. Diagnostics
# --------------
# Ljung–Box p-values per series
lb_pvals = {
    col: sm.stats.acorr_ljungbox(results.resid[col], lags=[10], return_df=True)[
        "lb_pvalue"
    ].iloc[0]
    for col in results.resid.columns
}
lb_df = pd.DataFrame.from_dict(
    lb_pvals, orient="index", columns=["Ljung-Box p-value (lag=10)"]
)
print("\nLjung-Box test:\n", lb_df)

# %%
results.get_coefficients_of_determination()

# %%
results.plot_coefficients_of_determination()
plt.tight_layout()
plt.show()
