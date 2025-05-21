# %% [markdown]
# # 경제 분석 및 예측과 데이터 지능 실습4.5: Volatility Modeling
#
# 파이썬 라이브러리 arch를 활용하여 변동성 모델링을 실습해 봅니다.
#
# 출처: https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_modeling.html

# %%


# %%
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(16, 6))
plt.rc("savefig", dpi=90)
plt.rc("font", family="sans-serif")
plt.rc("font", size=14)

# %%
import datetime as dt

import arch.data.sp500

st = dt.datetime(1988, 1, 1)
en = dt.datetime(2018, 1, 1)
data = arch.data.sp500.load()
market = data["Adj Close"]
returns = 100 * market.pct_change().dropna()
ax = returns.plot()
xlim = ax.set_xlim(returns.index.min(), returns.index.max())


# %%
from arch import arch_model

am = arch_model(returns)
res = am.fit(update_freq=5)
print(res.summary())


# %%
fig = res.plot(annualize="D")


# %%
am = arch_model(returns, p=1, o=1, q=1, power=1.0)
res = am.fit(update_freq=5)
print(res.summary())

# %%
am = arch_model(returns, p=1, o=1, q=1, power=1.0, dist="StudentsT")
res = am.fit(update_freq=5)
print(res.summary())

# %%
fixed_res = am.fix([0.0235, 0.01, 0.06, 0.0, 0.9382, 8.0])
print(fixed_res.summary())


# %%
import pandas as pd

df = pd.concat([res.conditional_volatility, fixed_res.conditional_volatility], axis=1)
df.columns = ["Estimated", "Fixed"]
subplot = df.plot()
subplot.set_xlim(xlim)


# %%
import arch.data.core_cpi

core_cpi = arch.data.core_cpi.load()
ann_inflation = 100 * core_cpi.CPILFESL.pct_change(12).dropna()
fig = ann_inflation.plot()


# %%
from arch.univariate import ARX

ar = ARX(100 * ann_inflation, lags=[1, 3, 12])
print(ar.fit().summary())

# %%
from arch.univariate import GARCH

ar.volatility = GARCH(p=5, q=5)
res = ar.fit(update_freq=0, disp="off")
print(res.summary())

# %%
from arch.univariate import EGARCH

ar.volatility = EGARCH(p=1, o=1, q=1)
res = ar.fit(update_freq=0, disp="off")
print(res.summary())

# %%
