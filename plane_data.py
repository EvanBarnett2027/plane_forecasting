import requests
import pandas as pd
import io

def download_ord_weather_data(start_date="2020-01-01", end_date="2025-12-31"):
    """
    Downloads historical ASOS/METAR weather data for Chicago O'Hare (ORD)
    from the Iowa Environmental Mesonet (IEM) API.
    
    Fetches hourly data, parses timestamps to datetime objects, and drops rows 
    where core weather variables are completely missing to return a clean DataFrame
    ready for machine learning feature engineering.
    """
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    
    # Parse dates to extract year, month, and day for the API query
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # IEM API query parameters for ASOS data
    params = {
        "station": "ORD",
        "data": "all",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": "Etc/UTC",        # Use UTC to avoid Daylight Saving Time issues
        "format": "onlycomma",  # Easiest format for Pandas parsing
        "latlon": "no",
        "missing": "empty",     # Blanks for missing values
        "trace": "0.0001",      # Convert trace precipitation ('T') to small float
        "direct": "yes"
    }
    
    print(f"Downloading hourly data for ORD from {start_date} to {end_date}...")
    response = requests.get(url, params=params)
    response.raise_for_status() 
    
    # Read text response into Pandas DataFrame
    df = pd.read_csv(
        io.StringIO(response.text),
        low_memory=False,
        na_values=['M', 'null'] # Treat 'M' (Missing) and 'null' as Pandas NaN
    )
    
    # Clean up column names to avoid whitespace issues
    df.columns = df.columns.str.strip()
    
    # Parse timestamps into datetime objects
    if 'valid' in df.columns:
        df['valid'] = pd.to_datetime(df['valid'], utc=True)
    else:
        raise KeyError("Expected timestamp column 'valid' not found in the API response.")
        
    # Drop rows where critical weather variables (temperature) are missing.
    # Interim sub-hourly reports often have wind/altimeter but lack temperature.
    initial_len = len(df)
    df = df.dropna(subset=['tmpf'])
    print(f"Dropped {initial_len - len(df)} rows missing temperature data.")
    
    # Regularize to strict hourly observations for Time-Series ML
    df['hour_index'] = df['valid'].dt.round('h')
    df = df.drop_duplicates(subset=['hour_index'], keep='first')
    
    # Set the clean hourly timestamp as index and clean up
    df = df.set_index('hour_index').sort_index()
    df = df.drop(columns=['valid'])
    
    print("Data download and cleaning complete.")
    return df

if __name__ == "__main__":
    # Fetch the ORD data
    ord_df = download_ord_weather_data(start_date="2020-01-01", end_date="2025-12-31")
    
    # Save the cleaned data to a CSV file
    output_file = "ord_weather_data.csv"
    ord_df.to_csv(output_file)
    print(f"\nData successfully saved to {output_file}")
    
    # Display the first few rows and dataframe information
    print("\nSample Data:")
    print(ord_df[['tmpf', 'dwpf', 'sknt', 'relh', 'alti', 'p01i']].head())
    
    print("\nDataFrame Info:")
    print(ord_df.info())
