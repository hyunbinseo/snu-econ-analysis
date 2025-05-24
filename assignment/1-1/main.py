# %%
import numpy as np
import pandas as pd
from preprocessing import TimeSeriesPreprocessor, create_sample_data


# %%
def main():
    """Main execution function"""
    print("=" * 60)
    print("🔍 Time Series Data Preprocessing Pipeline")
    print("=" * 60)

    # Generate sample data
    print("📊 Generating sample data...")
    df = create_sample_data()

    print(f"✅ Data generated successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(
        f"   Missing values: {df['value'].isnull().sum()}/{len(df)} ({(df['value'].isnull().sum() / len(df) * 100):.1f}%)"
    )

    return df


# %%
# Load the data
df = main()
df.head()

# %%
# Initialize preprocessor with plotting enabled
preprocessor = TimeSeriesPreprocessor(show_plots=True)
print("🔧 Preprocessor initialized with plotting enabled")

# %%
# Execute the complete preprocessing pipeline
print("🚀 Starting preprocessing pipeline...")
processed_data, preprocessing_info = preprocessor.preprocess(
    data=df, date_col="date", value_col="value", freq="D"
)

# %%
# Display comprehensive results
print("=" * 60)
print("📈 PREPROCESSING RESULTS SUMMARY")
print("=" * 60)

print(f"📊 Data Information:")
print(f"   Original length: {preprocessing_info['original_length']}")
print(f"   Final length: {len(processed_data)}")
print(f"   Missing values handled: {preprocessing_info['missing_count']}")

print(f"\n🔧 Applied Methods:")
print(f"   Missing value method: {preprocessing_info['missing_method']}")
print(
    f"   Log transformation: {'✅ Applied' if preprocessing_info['log_applied'] else '❌ Not applied'}"
)
print(
    f"   Seasonality removal: {'✅ Applied' if preprocessing_info['seasonal_removed'] else '❌ Not applied'}"
)

# %%
# Detailed stationarity results
stationarity = preprocessing_info["stationarity"]
print(f"📈 Stationarity Test Results:")
print(f"   ADF statistic: {stationarity['adf_statistic']:.4f}")
print(f"   p-value: {stationarity['p_value']:.4f}")
print(f"   Critical values: {stationarity['critical_values']}")
print(
    f"   Is stationary: {'✅ Yes' if stationarity['is_stationary'] else '❌ No (consider differencing)'}"
)

# %%
# Final data statistics
print(f"📊 Final Processed Data Statistics:")
print(f"   Mean: {processed_data.mean():.4f}")
print(f"   Standard Deviation: {processed_data.std():.4f}")
print(f"   Minimum: {processed_data.min():.4f}")
print(f"   Maximum: {processed_data.max():.4f}")
print(f"   Skewness: {processed_data.skew():.4f}")
print(f"   Kurtosis: {processed_data.kurtosis():.4f}")

# %%
# Save results to files
output_df = processed_data.to_frame("processed_value")
output_df.to_csv("processed_timeseries.csv")

info_df = pd.DataFrame([preprocessing_info])
info_df.to_csv("preprocessing_info.csv", index=False)

print("💾 Files saved:")
print("   ✅ processed_timeseries.csv")
print("   ✅ preprocessing_info.csv")

# %%
# Display first and last few values of processed data
print("🔍 Processed Data Preview:")
print("First 10 values:")
print(processed_data.head(10))
print("\nLast 10 values:")
print(processed_data.tail(10))


# %%
# Additional analysis functions
def perform_adf_test(series):
    """Perform ADF test on time series"""
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series.dropna())
    print("=" * 40)
    print("ADF Test Results:")
    print("=" * 40)
    print("ADF Statistic: %.6f" % result[0])
    print("p-value: %.6f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")

    if result[1] <= 0.05:
        print("✅ Reject null hypothesis - Data is stationary")
    else:
        print("❌ Fail to reject null hypothesis - Data is non-stationary")


# %%
# Run additional ADF test on final data
perform_adf_test(processed_data)

# %%
# Quick visualization of final result
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
df.set_index("date")["value"].plot(title="Original Data", alpha=0.7)
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
processed_data.plot(title="Preprocessed Data", color="red")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Preprocessing pipeline completed successfully!")

# %%
