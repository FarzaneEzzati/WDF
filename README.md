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
```

# Walmart Sales & FRED Macroeconomic Data Integration

This script automates the data integration pipeline for augmenting the **Walmart Store Sales** with macroeconomic time-series indicators from the **FRED**.

---

## Overview

The integration merges two granularities of external economic data into weekly retail sales records:
1. **Monthly FRED Indicators:** Merged using a left-join on converted `YearMonth` periods.
2. **Quarterly FRED Indicators (`TDSP`):** Merged using an As-Of join (`pd.merge_asof`) on sorted dates to assign the most recent known quarterly indicator without lookahead bias.

---

## Required Directory Structure

Place your input CSV files in the following directory hierarchy before executing the script:

```text
Data/
├── Walmart/
│   └── Walmart_Sales.csv
├── FRED/
│   ├── CPIAUCSL.csv
│   ├── GASREGW.csv
│   ├── UMCSENT.csv
│   ├── UNRATE.csv
│   └── TDSP.csv         # Quarterly Household Debt Service Ratio
└── inegrated_data.csv   # (Generated output location)