# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# %%
# build the time series, just a simple AR(1)
t1 = [0.1 * np.random.normal()]
for _ in range(100):
    t1.append(0.5 * t1[-1] + 0.1 * np.random.normal())

# %%
# build the time series that is granger caused by t1
t2 = [item + 0.1 * np.random.normal() for item in t1]

# %%
# adjust t1 and t2
t1 = t1[3:]
t2 = t2[:-3]

# %%
plt.figure(figsize=(10, 4))
plt.plot(t1, color="b")
plt.plot(t2, color="r")

plt.legend(["t1", "t2"], fontsize=16)

# %%
ts_df = pd.DataFrame(columns=["t2", "t1"], data=zip(t2, t1))

# %%
ts_df

# %%
gc_res = grangercausalitytests(ts_df, 3)

# %%
