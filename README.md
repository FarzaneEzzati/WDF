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

# Data Integration/Storage Pipeline

This script automates the data integration pipeline for augmenting the **Walmart Store Sales** with macroeconomic time-series indicators from the **FRED** and storing data in an SQLite database.

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
│   ├── FEDFUNDS.csv
│   ├── PCE.csv
│   ├── PPIACO.csv
│   ├── PSAVERT.csv
│   ├── RSXFS.csv
│   ├── UMCSENT.csv
│   └── TDSP.csv         # Quarterly Household Debt Service Ratio
└── Walmart.db   # (Generated output location)
```

# Statistical Analysis & Feature Engineering

This script executes statistical hypothesis testing, macroeconomic feature evaluation, and time-aware feature engineering on Walmart store sales data stored in an SQLite database. The output is a fully transformed dataset exported back to SQLite, prepared for downstream machine learning models. The transformed dataset is named "sales_economic_time_data".

---

## Pipeline Workflow

### 1. Dataset Acquisition
* Connects to SQLite database `Data/Walmart.db`.
* Queries the source table `sales_economic_data` into a Pandas DataFrame.

### 2. Group-Wise Statistical Validation
Evaluates distribution assumptions across store groups to determine the valid statistical testing strategy:
* **In-Group Normality:** Evaluated per store using the **Shapiro-Wilk test** (`stats.shapiro`).
* **Inter-Group Homogeneity:** Evaluated across all stores using **Levene's test** (`stats.levene`).
* **Hypothesis Decisioning:** 
  * If normality and variance homogeneity hold ($p \ge 0.05$), ANOVA is indicated.
  * If assumptions fail ($p < 0.05$), non-parametric alternatives (**Kruskal-Wallis** / **Mann-Whitney U**) are selected to test for significant sales differences across stores.

### 3. Feature Correlation & Significance Analysis
* **Sales Normalization:** Standardizes sales per store using Z-score calculation.
* **Spearman Rank Correlation:** Evaluates monotonic relationships ($p < 0.05$) between normalized sales and macroeconomic/store features (`Fuel_Price`, `Temperature`, `Unemployment`, `Holiday_Flag`, FRED metrics, etc.).
* Identifies non-significant features for downstream pruning.

### 4. Time-Aware Feature Engineering
* **Chronological Sorting:** Enforces strict date ordering per store (`Store`, `Date`) to prevent index alignment issues.
* **Cyclical Seasonality Encoding:** Transforms calendar week numbers into continuous $360^\circ$ cyclical coordinates without creating leading `NaN` dropouts, as the dataset is small.
* **Rolling Window Aggregations:** Computes a 4-week moving average shifted by 1 period (`shift(1)`) to eliminate target leakage.
* **Imputation:** Implements backward filling (`bfill`) on initial window gaps to retain 100% of rows.

### 5. Data Persistence
* Writes the processed feature matrix to a new SQLite table: `sales_economic_time_data`.

---

## Dependencies
* `pandas`
* `numpy`
* `scipy`
* `statsmodels`
* `sqlite3`

---
