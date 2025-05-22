# %% [markdown]
# # GARCH Stock Forecasting

# %% [markdown]
# ## Read Data

# %%
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %% [markdown]
# ## DIS Volatility

# %%
start = datetime(2015, 1, 1)
end = datetime(2020, 6, 10)

# %%
dis = web.DataReader("DIS", "yahoo", start=start, end=end)

# %%
returns = 100 * dis.Close.pct_change().dropna()

# %%
plt.figure(figsize=(10, 4))
plt.plot(returns)
plt.ylabel("Pct Return", fontsize=16)
plt.title("DIS Returns", fontsize=20)

# %% [markdown]
# ## PACF

# %%
plot_pacf(returns**2)
plt.show()

# %% [markdown]
# ## Fit GARCH(3,3)

# %%
model = arch_model(returns, p=3, q=3)

# %%
model_fit = model.fit()

# %%
model_fit.summary()

# %% [markdown]
# ## Try GARCH(3,0) = ARCH(3)

# %%
model = arch_model(returns, p=3, q=0)

# %%
model_fit = model.fit()

# %%
model_fit.summary()

# %%
rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = returns[: -(test_size - i)]
    model = arch_model(train, p=3, q=0)
    model_fit = model.fit(disp="off")
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

# %%
rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365:])

# %%
plt.figure(figsize=(10, 4))
(true,) = plt.plot(returns[-365:])
(preds,) = plt.plot(rolling_predictions)
plt.title("Volatility Prediction - Rolling Forecast", fontsize=20)
plt.legend(["True Returns", "Predicted Volatility"], fontsize=16)

# %% [markdown]
# # S&P 500

# %%
start = datetime(2000, 1, 1)
end = datetime(2020, 6, 10)

# %%
spy = web.DataReader("SPY", "yahoo", start=start, end=end)

# %%
returns = 100 * spy.Close.pct_change().dropna()

# %%
plt.figure(figsize=(10, 4))
plt.plot(returns)
plt.ylabel("Pct Return", fontsize=16)
plt.title("SPY Returns", fontsize=20)

# %% [markdown]
# ## PACF

# %%
plot_pacf(returns**2)
plt.show()

# %% [markdown]
# ## Fit GARCH(2,2)

# %%
model = arch_model(returns, p=2, q=2)

# %%
model_fit = model.fit()

# %%
model_fit.summary()

# %% [markdown]
# ## Rolling Forecast

# %%
rolling_predictions = []
test_size = 365 * 5

for i in range(test_size):
    train = returns[: -(test_size - i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp="off")
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

# %%
rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365 * 5 :])

# %%
plt.figure(figsize=(10, 4))
(true,) = plt.plot(returns[-365 * 5 :])
(preds,) = plt.plot(rolling_predictions)
plt.title("Volatility Prediction - Rolling Forecast", fontsize=20)
plt.legend(["True Returns", "Predicted Volatility"], fontsize=16)

# %% [markdown]
# # How to use the model

# %%
train = returns
model = arch_model(train, p=2, q=2)
model_fit = model.fit(disp="off")

# %%
pred = model_fit.forecast(horizon=7)
future_dates = [returns.index[-1] + timedelta(days=i) for i in range(1, 8)]
pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)

# %%
plt.figure(figsize=(10, 4))
plt.plot(pred)
plt.title("Volatility Prediction - Next 7 Days", fontsize=20)

# %%
