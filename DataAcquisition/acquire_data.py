import kagglehub
from pathlib import Path as path
def acquire_walmart_data(dataset_name: str, download_path: str) -> None:
    try:
        if not path(download_path).exists():
            print(f"INFO >> Creating Walmart data directory: {download_path}")
            output = kagglehub.dataset_download(dataset_name, output_dir=download_path)
            print(f"INFO >> Walmart dataset downloaded successfully: {output}")
            # Store info about the downloaded dataset in a text file    
            with open(f"{download_path}/info.txt", "w") as f:
                f.write(f"{dataset_name}: {output}")
            
        else:
            with open(f"{download_path}/info.txt", "r") as f:
                info = f.read()
                print(f"INFO >> Walmart dataset already available: {info}")
    except Exception as e:
        print(f"Error occurred while downloading dataset: {e}")


import pandas as pd
from fredapi import Fred
import os
def acquire_fred_data(api_key: str, series_ids: str, start_date: str, end_date: str, download_path: str) -> None:
    # Check for unavilable FRED data
    available_data = [f for f in os.listdir(download_path)]
    is_all_available = True
    for column_name, series_id in series_ids.items():
        if f"{series_id}.csv" not in available_data:
            del series_ids[column_name]
            is_all_available = False

    # Return if all data is already downloaded and stored
    if is_all_available:
        print(f"INFO >> Data is already stored at {download_path}")
        return None


    # Initialize FRED API client
    fred = Fred(api_key)

    # Fetch a list of data frames for each id
    for column_name, series_id in series_ids.items():
        data = fred.get_series(series_id= series_id, observation_start=start_date, observation_end=end_date)
        df = pd.DataFrame({"index": data.index, column_name: data.values})
        df.to_csv(download_path + f"/{series_id}" + ".csv")
        print(f'INFO >> Data for series id {series_id} acquired and stored at {download_path} as .csv file.')


if __name__ == "__main__":

    # Walmart Sales dataset from Kaggle
    dataset_name = "mikhail1681/walmart-sales"
    download_path = "DataAcquisition/WalmartData"
    acquire_walmart_data(dataset_name, download_path)

    # External API 1: FRED API (St. Louis Fed) 
    fred_api = "21469988bde463e03fb54e173498e2de"
    # Fetch series data
    series_ids = {
        "consumer_sentiment": "UMCSENT",
        "advanced_retail_asles": "RSXFS",
        "personal_consum_expend": "PCE",
        "personal_saving_rate": "PSAVERT",
        "household_dept_srvc_ratio": "TDSP",
        "proucer_price_index": "PPIACO",
        "effective_federal_fund": "FEDFUNDS"
    }
    download_path = "DataAcquisition/FRED"
    acquire_fred_data(api_key=fred_api, 
                      series_ids=series_ids, 
                      start_date='01-01-2010', 
                      end_date='12-31-2012', 
                      download_path=download_path)
