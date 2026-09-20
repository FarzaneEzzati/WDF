"""
fetch_integrate_data.py

Combines data acquisition (Kaggle Walmart sales + FRED economic series) and
integration (merging them into one data frame) into a single stage. This stage
is intentionally kept separate from feature engineering: it is only
responsible for producing one clean, merged, in-memory DataFrame.

Usage:
    from src.data.fetch_integrate_data import get_integrated_data

    df = get_integrated_data()
    # df now goes straight into engineer_features(df)
"""

from pathlib import Path
import os as _os
import pandas as pd
import kagglehub
from fredapi import Fred
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Walmart sales data (Kaggle)
# ---------------------------------------------------------------------------

def fetch_walmart_data(
    dataset_name: str,    # The kaggle name of the dataset
    download_path: str,   # The path to save the data
    dataframe_name: str,  # The name of the datafeam to save: <Walmart_Sales>
) -> pd.DataFrame:
    """
    Download the Walmart sales dataset from Kaggle if not already cached
    locally, then load it as a DataFrame.

    Returns
    -------
    pd.DataFrame
        The raw Walmart sales table.
    """
    
    download_dir = Path(download_path)
    csv_path = download_dir / f"{dataframe_name}.csv"

    if not download_dir.exists():
        logger.info(f"Creating Walmart data directory: {download_path}")
        kagglehub.dataset_download(dataset_name, output_dir=download_path)
        with open(download_dir / "info.txt", "w") as f:
            f.write(dataset_name)
        logger.info(f"Walmart dataset '{dataframe_name}' downloaded to {download_path}")
    else:
        logger.info(f"Walmart dataset already cached at {download_path}")

    if not csv_path.exists():
        logger.error(
            FileNotFoundError(
            f"Expected Walmart CSV at {csv_path} but it was not found after download."
            )
        )
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Economic indicators (FRED API)
# ---------------------------------------------------------------------------

def fetch_fred_data(
    api_key: str,
    series_ids: dict,
    start_date: str,
    end_date: str,
    download_path: str,
) -> dict:
    """
    Download each FRED series (if not already cached), and return all of
    them as a dict of DataFrames: {column_name: DataFrame}.

    This always returns usable DataFrames, whether the data was just
    downloaded or already existed on disk.
    """
    download_dir = Path(download_path)
    download_dir.mkdir(parents=True, exist_ok=True)

    cached_files = {f.stem for f in download_dir.glob("*.csv")}  # Set of file stems (names) ending in .csv
    needed_series = set(series_ids.values())                     # Set of series id required

    if not needed_series.issubset(cached_files):
        logger.info(f"Fetching missing FRED series into {download_path}")
        fred = Fred(api_key)
        for column_name, series_id in series_ids.items():
            if series_id in cached_files:
                continue
            data = fred.get_series(
                series_id=series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            df = pd.DataFrame({"index": data.index, column_name: data.values})
            df.to_csv(download_dir / f"{series_id}.csv", index=False)
            logger.info(f"Series '{series_id}' fetched and cached.")
    else:
        logger.info(f"All FRED series already cached at {download_path}")

    # Always load from disk into memory.
    fred_dfs = {
        column_name: pd.read_csv(download_dir / f"{series_id}.csv")
        for column_name, series_id in series_ids.items()
    }
    return fred_dfs


# ---------------------------------------------------------------------------
# Integration: merge Walmart sales with FRED series into one table
# ---------------------------------------------------------------------------

def integrate_datasets(
    walmart_df: pd.DataFrame,
    fred_dfs: dict,
    quarterly_series_name: str = "household_dept_srvc_ratio",
) -> pd.DataFrame:
    """
    Merge Walmart weekly sales with monthly/quarterly FRED series.

    Parameters
    ----------
    walmart_df : pd.DataFrame
        Raw Walmart sales table with a 'Date' column.
    fred_dfs : dict
        {column_name: DataFrame} as returned by fetch_fred_data. Each
        DataFrame has an 'index' column (dates) and one value column.
    quarterly_series_name : str
        Key in fred_dfs that is on a quarterly cadence (merged via
        merge_asof instead of an exact YearMonth join).

    Returns
    -------
    pd.DataFrame
        Walmart sales merged with all FRED series, one row per original
        Walmart row.
    """
    logger.info(f"Integrating Walamrt_Sales and FRED Data Sets...")
    walmart_df = walmart_df.copy()
    walmart_df["Date"] = pd.to_datetime(walmart_df["Date"])
    walmart_df["YearMonth"] = walmart_df["Date"].dt.to_period("M")
    logger.info("Date converted to YearMonth for Walmart_Sales.")

    # Monthly series: exact YearMonth join
    for column_name, df in fred_dfs.items():
        if column_name == quarterly_series_name:
            continue
        df = df.copy()
        df["index"] = pd.to_datetime(df["index"])
        df["YearMonth"] = df["index"].dt.to_period("M")
        df_clean = df.drop(columns=["index"]).drop_duplicates(subset=["YearMonth"])
        walmart_df = pd.merge(walmart_df, df_clean, on="YearMonth", how="left")
        logger.info(f"Merged monthly series '{column_name}'.")

    # Quarterly series: merge_asof (backward-fill to most recent quarter)
    if quarterly_series_name in fred_dfs:
        q_df = fred_dfs[quarterly_series_name].copy()
        q_df["Date"] = pd.to_datetime(q_df["index"])
        q_df = q_df.drop(columns=["index"]).sort_values("Date")

        walmart_df = walmart_df.sort_values("Date")
        walmart_df = pd.merge_asof(walmart_df, q_df, on="Date", direction="backward")
        logger.info(f"Merged quarterly series '{quarterly_series_name}' via merge_asof.")

    walmart_df = walmart_df.drop(columns=["YearMonth"])
    return walmart_df


# ---------------------------------------------------------------------------
# Orchestration: this is the single entry point engineer.py / main.py calls
# ---------------------------------------------------------------------------

def get_integrated_data(
    kaggle_dataset: str = "mikhail1681/walmart-sales",
    kaggle_download_path: str = "../../data/raw/walmart",
    walmart_dataframe_name: str = "Walmart_Sales",
    fred_api_key: str = _os.environ.get("FRED_API_KEY"),
    fred_series_ids: dict = None,
    fred_start_date: str = "01-01-2010",
    fred_end_date: str = "12-31-2012",
    fred_download_path: str = "../../data/raw/fred",
) -> pd.DataFrame:
    """
    Fetches Walmart sales + FRED series and integrates them into a single
    in-memory DataFrame, ready to be handed to engineer_features().
    """
    if fred_api_key is None:
        fred_api_key = "21469988bde463e03fb54e173498e2de" # External API 1: FRED API (St. Louis Fed) 


    if fred_series_ids is None:
        fred_series_ids = {
            "consumer_sentiment": "UMCSENT",
            "advanced_retail_sales": "RSXFS",
            "personal_consum_expend": "PCE",
            "personal_saving_rate": "PSAVERT",
            "household_dept_srvc_ratio": "TDSP",
            "producer_price_index": "PPIACO",
            "effective_federal_fund": "FEDFUNDS",
        }

    walmart_df = fetch_walmart_data(
        dataset_name=kaggle_dataset,
        download_path=kaggle_download_path,
        dataframe_name=walmart_dataframe_name,
    )

    fred_dfs = fetch_fred_data(
        api_key=fred_api_key,
        series_ids=fred_series_ids,
        start_date=fred_start_date,
        end_date=fred_end_date,
        download_path=fred_download_path,
    )

    integrated_df = integrate_datasets(walmart_df, fred_dfs)
    logger.info(f"Integration complete. Shape: {integrated_df.shape}")
    return integrated_df
