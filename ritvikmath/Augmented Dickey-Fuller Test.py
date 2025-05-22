# %%
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller


# %%
def generate_ar_process(lags, coefs, length):
    # cast coefs to np array
    coefs = np.array(coefs)

    # initial values
    series = [np.random.normal() for _ in range(lags)]

    for _ in range(length):
        # get previous values of the series, reversed
        prev_vals = series[-lags:][::-1]

        # get new value of time series
        new_val = np.sum(np.array(prev_vals) * coefs) + np.random.normal()

        series.append(new_val)

    return np.array(series)


# %%
def perform_adf_test(series):
    result = adfuller(series)
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])


# %% [markdown]
# # AR(1) Process

# %% [markdown]
# ## Stationary

# %%
ar_1_process = generate_ar_process(1, [0.5], 100)
plt.figure(figsize=(10, 4))
plt.plot(ar_1_process)
plt.title("Stationary AR(1) Process", fontsize=18)

# %%
perform_adf_test(ar_1_process)

# %% [markdown]
# ## Non-Stationary

# %%
ar_1_process_unit_root = generate_ar_process(1, [1], 100)
plt.figure(figsize=(10, 4))
plt.plot(ar_1_process_unit_root)
plt.title("Non-Stationary AR(1) Process", fontsize=18)

# %%
perform_adf_test(ar_1_process_unit_root)

# %% [markdown]
# # AR(2) Process

# %% [markdown]
# ## Stationary

# %%
ar_2_process = generate_ar_process(2, [0.5, 0.3], 100)
plt.figure(figsize=(10, 4))
plt.plot(ar_2_process)
plt.title("Stationary AR(2) Process", fontsize=18)

# %%
perform_adf_test(ar_2_process)

# %% [markdown]
# ## Non-Stationary

# %%
ar_2_process_unit_root = generate_ar_process(2, [0.7, 0.3], 100)
plt.figure(figsize=(10, 4))
plt.plot(ar_2_process_unit_root)
plt.title("Non-Stationary AR(2) Process", fontsize=18)

# %%
perform_adf_test(ar_2_process_unit_root)

# %%
