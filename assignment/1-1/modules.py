import pandas as pd
import requests


def get_historical_precipitation(latitude, longitude, start_date, end_date):
    """
    Fetches historical daily precipitation sum for a given location and date range.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'ds' (date) and 'precipitation' columns,
                      or an empty DataFrame if data fetching fails.
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",  # Request daily precipitation sum
        "timezone": "Asia/Seoul",
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        if "daily" in data:
            # Extract dates and precipitation sums
            dates = data["daily"]["time"]
            precipitation_sums = data["daily"]["precipitation_sum"]

            # Create a DataFrame
            precipitation_df = pd.DataFrame(
                {"ds": pd.to_datetime(dates), "precipitation": precipitation_sums}
            )
            return precipitation_df
        else:
            print("No daily precipitation data found in the response.")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
    except KeyError as e:
        print(f"Unexpected data structure from API: {e}")
        return pd.DataFrame()
