from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class TimeSeriesPreprocessor:
    def __init__(self, show_plots: bool = True):
        self.show_plots = show_plots
        self.preprocessing_steps = []

    def handle_missing_values(self, ts: pd.Series) -> Tuple[pd.Series, str]:
        """Handle missing values"""
        print("=== Step 1: Missing Value Treatment ===")
        print(f"Original data size: {len(ts)}")
        print(f"Missing values count: {ts.isnull().sum()}")

        if self.show_plots:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Original data
            ts.plot(
                ax=axes[0, 0],
                title="Original Data (with missing values)",
                color="blue",
                alpha=0.7,
            )
            axes[0, 0].scatter(
                ts[ts.isnull()].index,
                [ts.mean()] * ts.isnull().sum(),
                color="red",
                s=50,
                label="Missing positions",
            )
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Linear interpolation
        ts_interpolated = ts.interpolate(method="linear")

        # Forward/backward fill
        ts_filled = ts.fillna(method="ffill").fillna(method="bfill")

        # Choose method based on missing ratio
        missing_ratio = ts.isnull().sum() / len(ts)

        if missing_ratio < 0.1:
            ts_clean = ts_interpolated
            method = "linear_interpolation"
        else:
            ts_clean = ts_filled
            method = "forward_backward_fill"

        if self.show_plots:
            # Processed data
            ts_interpolated.plot(
                ax=axes[0, 1], title="Linear Interpolation Applied", color="green"
            )
            axes[0, 1].grid(True, alpha=0.3)

            ts_filled.plot(ax=axes[1, 0], title="Forward/Backward Fill", color="orange")
            axes[1, 0].grid(True, alpha=0.3)

            ts_clean.plot(
                ax=axes[1, 1], title=f"Final Choice: {method}", color="purple"
            )
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        print(f"Selected method: {method}")
        print(f"Missing values after treatment: {ts_clean.isnull().sum()}")

        return ts_clean, method

    def apply_log_transformation(self, ts: pd.Series) -> Tuple[pd.Series, bool]:
        """Apply log transformation"""
        print("\n=== Step 2: Log Transformation ===")

        # Convert to positive values
        if (ts <= 0).any():
            print("Negative/zero values exist → Adding constant")
            min_val = ts.min()
            ts_positive = ts - min_val + 1
        else:
            ts_positive = ts

        ts_log = np.log(ts_positive)

        # Normality test
        sample_size = min(5000, len(ts.dropna()))
        _, p_before = stats.shapiro(ts.dropna().iloc[:sample_size])
        _, p_after = stats.shapiro(ts_log.dropna().iloc[:sample_size])

        print(f"Normality p-value before log: {p_before:.4f}")
        print(f"Normality p-value after log: {p_after:.4f}")

        log_applied = p_after > p_before
        ts_result = ts_log if log_applied else ts

        if self.show_plots:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # Time series plots
            ts.plot(ax=axes[0, 0], title="Original Time Series", color="blue")
            axes[0, 0].grid(True, alpha=0.3)

            ts_log.plot(ax=axes[0, 1], title="Log Transformed Time Series", color="red")
            axes[0, 1].grid(True, alpha=0.3)

            ts_result.plot(
                ax=axes[0, 2],
                title=f"Final Choice (Log applied: {log_applied})",
                color="green",
            )
            axes[0, 2].grid(True, alpha=0.3)

            # Histograms
            ts.hist(ax=axes[1, 0], bins=50, alpha=0.7, color="blue")
            axes[1, 0].set_title(f"Original Distribution (p={p_before:.4f})")
            axes[1, 0].grid(True, alpha=0.3)

            ts_log.hist(ax=axes[1, 1], bins=50, alpha=0.7, color="red")
            axes[1, 1].set_title(f"Log Transform Distribution (p={p_after:.4f})")
            axes[1, 1].grid(True, alpha=0.3)

            ts_result.hist(ax=axes[1, 2], bins=50, alpha=0.7, color="green")
            axes[1, 2].set_title("Final Choice Distribution")
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        print(f"Log transformation applied: {log_applied}")
        return ts_result, log_applied

    def remove_seasonality(
        self, ts: pd.Series, freq: str = "D"
    ) -> Tuple[pd.Series, bool]:
        """Remove seasonality"""
        print("\n=== Step 3: Seasonality Analysis and Removal ===")

        # Set period
        periods = {"D": 7, "M": 12, "H": 24, "W": 4}
        period = periods.get(freq, 7)

        try:
            if len(ts.dropna()) >= 2 * period:
                decomposition = seasonal_decompose(
                    ts.dropna(), model="additive", period=period
                )

                seasonal_strength = np.var(decomposition.seasonal) / np.var(ts.dropna())
                print(f"Seasonal strength: {seasonal_strength:.4f}")

                if seasonal_strength > 0.1:
                    ts_deseasonalized = ts - decomposition.seasonal
                    seasonal_removed = True
                    print("Seasonality removal applied")
                else:
                    ts_deseasonalized = ts
                    seasonal_removed = False
                    print("Seasonality is weak, not removed")

                if self.show_plots:
                    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

                    ts.plot(ax=axes[0, 0], title="Original Data", color="blue")
                    axes[0, 0].grid(True, alpha=0.3)

                    decomposition.trend.plot(
                        ax=axes[0, 1], title="Trend Component", color="green"
                    )
                    axes[0, 1].grid(True, alpha=0.3)

                    decomposition.seasonal.plot(
                        ax=axes[1, 0], title="Seasonal Component", color="orange"
                    )
                    axes[1, 0].grid(True, alpha=0.3)

                    if seasonal_removed:
                        ts_deseasonalized.plot(
                            ax=axes[1, 1],
                            title="After Seasonality Removal",
                            color="red",
                        )
                    else:
                        ts.plot(
                            ax=axes[1, 1], title="Original Maintained", color="blue"
                        )
                    axes[1, 1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.show()

            else:
                ts_deseasonalized = ts
                seasonal_removed = False
                print("Insufficient data for seasonality analysis")

        except Exception as e:
            print(f"Seasonal decomposition failed: {e}")
            ts_deseasonalized = ts
            seasonal_removed = False

        return ts_deseasonalized, seasonal_removed

    def check_stationarity(self, ts: pd.Series) -> Dict[str, Any]:
        """Check stationarity"""
        print("\n=== Step 4: Stationarity Test ===")

        adf_result = adfuller(ts.dropna())

        stationarity_info = {
            "adf_statistic": adf_result[0],
            "p_value": adf_result[1],
            "critical_values": adf_result[4],
            "is_stationary": adf_result[1] < 0.05,
        }

        print(f"ADF test statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print(f"Critical values: {adf_result[4]}")

        if adf_result[1] < 0.05:
            print("✅ Time series is stationary.")
        else:
            print("❌ Time series is non-stationary. Consider differencing.")

        if self.show_plots:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Time series plot
            ts.plot(ax=axes[0, 0], title="Final Preprocessed Time Series")
            axes[0, 0].grid(True, alpha=0.3)

            # Rolling mean and std
            rolling_mean = ts.rolling(window=30).mean()
            rolling_std = ts.rolling(window=30).std()

            ts.plot(ax=axes[0, 1], alpha=0.7, label="Original")
            rolling_mean.plot(ax=axes[0, 1], color="red", label="Rolling Mean (30)")
            rolling_std.plot(ax=axes[1, 0], color="orange", label="Rolling Std (30)")
            axes[0, 1].legend()
            axes[0, 1].set_title("Rolling Mean Trend")
            axes[0, 1].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].set_title("Rolling Standard Deviation")
            axes[1, 0].grid(True, alpha=0.3)

            # ACF plot (Autocorrelation Function)
            from statsmodels.tsa.stattools import acf

            autocorr = acf(ts.dropna(), nlags=40, fft=True)
            axes[1, 1].plot(autocorr)
            axes[1, 1].axhline(y=0, color="k", linestyle="-", alpha=0.3)
            axes[1, 1].axhline(
                y=1.96 / np.sqrt(len(ts)), color="r", linestyle="--", alpha=0.7
            )
            axes[1, 1].axhline(
                y=-1.96 / np.sqrt(len(ts)), color="r", linestyle="--", alpha=0.7
            )
            axes[1, 1].set_title("Autocorrelation Function (ACF)")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return stationarity_info

    def preprocess(
        self,
        data: pd.DataFrame,
        date_col: str = None,
        value_col: str = None,
        freq: str = "D",
    ) -> Tuple[pd.Series, Dict]:
        """Complete preprocessing pipeline"""
        print("🚀 Time Series Data Preprocessing Started\n")

        # Data preparation
        if isinstance(data, pd.DataFrame):
            if date_col:
                data[date_col] = pd.to_datetime(data[date_col])
                data = data.set_index(date_col)
            ts = data[value_col] if value_col else data.iloc[:, 0]
        else:
            ts = data.copy()

        # Step-by-step processing
        ts_clean, missing_method = self.handle_missing_values(ts)
        ts_transformed, log_applied = self.apply_log_transformation(ts_clean)
        ts_final, seasonal_removed = self.remove_seasonality(ts_transformed, freq)
        stationarity_info = self.check_stationarity(ts_final)

        # Summary
        preprocessing_info = {
            "original_length": len(ts),
            "missing_count": ts.isnull().sum(),
            "missing_method": missing_method,
            "log_applied": log_applied,
            "seasonal_removed": seasonal_removed,
            "stationarity": stationarity_info,
        }

        print("\n✅ Preprocessing completed!")
        print(f"📊 Processing results: {preprocessing_info}")

        return ts_final, preprocessing_info


def create_sample_data() -> pd.DataFrame:
    """Create sample data"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Trend + seasonality + noise
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    weekly = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + weekly + noise

    # Add missing values
    missing_indices = np.random.choice(len(values), size=80, replace=False)
    values[missing_indices] = np.nan

    return pd.DataFrame({"date": dates, "value": values})
