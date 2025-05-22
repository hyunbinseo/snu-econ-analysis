# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller


# %%
def perform_adf_test(series):
    result = adfuller(series)
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])


# %% [markdown]
# # Original Series

# %%
ts = pd.read_csv("original_series.csv")
ts.index = np.arange(1, len(ts) + 1)

# %%
plt.figure(figsize=(10, 4))
plt.plot(ts)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel("Hours Since Published", fontsize=16)

plt.yticks(np.arange(0, 50000, 10000), fontsize=14)
plt.ylabel("Views", fontsize=16)

# %% [markdown]
# # Original Series: $v_t$
#
# # (1) Normalize ($v_t \rightarrow n_t$): $n_t = \frac{v_t - \mu}{\sigma}$
#
# # (2) Exponentiate ($n_t \rightarrow e_t$): $e_t = e^{n_t}$
#
# # (3) First Difference ($e_t \rightarrow d_t$): $d_t = e_t - e_{t-1}$
#
# # $d_t = e^{\frac{v_t - \mu}{\sigma}} - e^{\frac{v_{t-1} - \mu}{\sigma}}$
#
# # ------------------------------------------------------------

# %% [markdown]
# # (1) Normalize

# %%
mu = np.mean(ts).iloc[0]
sigma = np.std(ts).iloc[0]

norm_ts = (ts - mu) / sigma

# %%
plt.figure(figsize=(10, 4))
plt.plot(norm_ts)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel("Hours Since Published", fontsize=16)

plt.yticks(np.arange(-3, 2), fontsize=14)
plt.ylabel("Norm. Views", fontsize=16)

plt.axhline(0, color="k", linestyle="--")

# %% [markdown]
# # (2) Exponentiate

# %%
exp_ts = np.exp(norm_ts)

# %%
plt.figure(figsize=(10, 4))
plt.plot(exp_ts)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel("Hours Since Published", fontsize=16)

plt.yticks(np.arange(0, 3.5, 0.5), fontsize=14)
plt.ylabel("Exp. Norm. Views", fontsize=16)

# %%
perform_adf_test(exp_ts)

# %% [markdown]
# # (3) First Difference

# %%
diff_ts = exp_ts.diff().dropna()

# %%
plt.figure(figsize=(10, 4))
plt.plot(diff_ts)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel("Hours Since Published", fontsize=16)

plt.yticks(np.arange(-0.2, 0.3, 0.1), fontsize=14)
plt.ylabel("First Diff. \nExp. Norm. Views", fontsize=16)

# %%
perform_adf_test(diff_ts)

# %% [markdown]
# # Fit AR Model

# %%
plot_pacf(diff_ts)
plt.show()

# %%
plot_acf(diff_ts)
plt.show()

# %%
# create the model
model = ARMA(diff_ts, order=(4, 1))

# %%
model_fit = model.fit()

# %% [markdown]
# # Predict Out 3 Hours

# %%
prediction_info = model_fit.forecast(3)

predictions = prediction_info[0]
lower_bound = prediction_info[2][:, 0]
upper_bound = prediction_info[2][:, 1]

# %%
plt.figure(figsize=(10, 4))
plt.plot(diff_ts)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel("Hours Since Published", fontsize=16)

plt.yticks(np.arange(-0.2, 0.3, 0.1), fontsize=14)
plt.ylabel("First Diff. \nExp. Norm. Views", fontsize=16)

plt.plot(np.arange(len(ts) + 1, len(ts) + 4), predictions, color="g")
plt.fill_between(
    np.arange(len(ts) + 1, len(ts) + 4), lower_bound, upper_bound, color="g", alpha=0.1
)

# %% [markdown]
# # Undo Transformations: ($\hat{d}_{t+1} \rightarrow \hat{v}_{t+1}$)
#
# # $\hat{v}_{t+1} = \sigma \ln(\hat{d}_{t+1} + e^{\frac{v_t - \mu}{\sigma}}) + \mu$


# %%
def undo_transformations(predictions, series, mu, sigma):
    first_pred = (
        sigma * np.log(predictions[0] + np.exp((series.iloc[-1] - mu) / sigma)) + mu
    )
    orig_predictions = [first_pred]

    for i in range(len(predictions[1:])):
        next_pred = (
            sigma
            * np.log(predictions[i + 1] + np.exp((orig_predictions[-1] - mu) / sigma))
            + mu
        )
        orig_predictions.append(next_pred)

    return np.array(orig_predictions).flatten()


# %%
orig_preds = undo_transformations(predictions, ts, mu, sigma)
orig_lower_bound = undo_transformations(lower_bound, ts, mu, sigma)
orig_upper_bound = undo_transformations(upper_bound, ts, mu, sigma)

# %%
plt.figure(figsize=(10, 4))
plt.plot(ts)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel("Hours Since Published", fontsize=16)

plt.yticks(np.arange(0, 50000, 10000), fontsize=14)
plt.ylabel("Views", fontsize=16)

plt.plot(np.arange(len(ts) + 1, len(ts) + 4), orig_preds, color="g")
plt.fill_between(
    np.arange(len(ts) + 1, len(ts) + 4),
    orig_lower_bound,
    orig_upper_bound,
    color="g",
    alpha=0.1,
)

# %%
plt.figure(figsize=(10, 4))
plt.plot(ts)

plt.xticks(np.arange(0, 78), fontsize=14)
plt.xlabel("Hours Since Published", fontsize=16)

plt.yticks(np.arange(40000, 46000, 1000), fontsize=14)
plt.ylabel("Views", fontsize=16)

plt.plot(np.arange(len(ts) + 1, len(ts) + 4), orig_preds, color="g")
plt.fill_between(
    np.arange(len(ts) + 1, len(ts) + 4),
    orig_lower_bound,
    orig_upper_bound,
    color="g",
    alpha=0.1,
)
plt.xlim(64, 76)
plt.ylim(40000, 45000)

# %%
