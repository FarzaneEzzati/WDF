# Data Acquisition Pipeline: Walmart & FRED Indicators

Automated data acquisition script to fetch historical sales data from Kaggle (2010-2012) alongside macro-economic indicators from the Federal Reserve Economic Data (FRED) API.

## Features

* **Kaggle Ingestion:** Retrieves the Walmart sales dataset via `kagglehub` and writes dataset metadata (`info.txt`).
* **FRED Ingestion:** Downloads key U.S. economic time-series indicators between specified start and end dates.
* **Redundancy Checks:** Inspects local storage paths prior to downloading to avoid unnecessary API calls and redundant downloads.

---

## Prerequisites & Installation

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install pandas fredapi kagglehub