# Walmart Demand Forecast (WDF)

An end-to-end pipeline that predicts Walmart weekly store sales by combining historical sales data with macroeconomic indicators from FRED, then benchmarks several regression models against a mean baseline.

## Project Structure

```text
WDF/
├── data/
│   └── raw/
│       ├── walmart/            # Walmart_Sales.csv (via kagglehub)
│       └── fred/                # FEDFUNDS, PCE, PPIACO, PSAVERT, RSXFS, TDSP, UMCSENT
│
├── src/
│   ├── data_prep/
│   │   ├── fetch_integrate.py   # download + merge Walmart sales with FRED series
│   │   ├── engineer.py          # statistical tests + time-aware feature engineering
│   │   ├── split.py             # chronological (and random) train/test split
│   │   └── feature_engineering.ipynb
│   ├── models/
│   │   ├── base.py              # BaseModel interface (fit/predict/get_params)
│   │   ├── linear_models.py     # LinearRegressionModel
│   │   ├── tree_models.py       # RandomForestRegressorModel, XGBoostModel
│   │   └── other_models.py      # (planned: KNN, SVR, MLP)
│   ├── evaluate.py              # RMSE, MAE, R², MAPE
│   └── tests/                   # unit tests for split, models, evaluate
│
├── reports/
│   ├── pipeline.log             # run log
│   └── models_evaluations.csv   # metrics per model, sorted by RMSE
│
├── data_lookup.ipynb            # exploratory notebook
└── main.py                      # orchestrates the full pipeline
```

## Pipeline Overview

Running `main.py` executes the full pipeline end to end:

### 1. Data Acquisition & Integration (`src/data_prep/fetch_integrate.py`)
* Downloads the Walmart weekly sales dataset from Kaggle (`mikhail1681/walmart-sales`) via `kagglehub`, caching it under `data/raw/walmart/`.
* Downloads seven FRED economic indicators via the FRED API, caching each under `data/raw/fred/`:
  * `UMCSENT` — Consumer Sentiment
  * `RSXFS` — Advanced Retail Sales
  * `PCE` — Personal Consumption Expenditures
  * `PSAVERT` — Personal Saving Rate
  * `TDSP` — Household Debt Service Ratio (quarterly)
  * `PPIACO` — Producer Price Index
  * `FEDFUNDS` — Effective Federal Funds Rate
* Merges monthly indicators onto Walmart sales via an exact `YearMonth` join, and merges the quarterly `TDSP` series via `pd.merge_asof` (backward direction) to avoid lookahead bias.

### 2. Statistical Testing & Feature Engineering (`src/data_prep/engineer.py`)
* Tests within-store normality (Shapiro-Wilk) and cross-store variance homogeneity (Levene's test) to decide between parametric and non-parametric analysis.
* Runs Kruskal-Wallis and Spearman rank correlation (on Z-score normalized sales) to flag statistically non-significant features.
* Engineers time-aware features:
  * Cyclical week-of-year encoding (`sin_week`, `cos_week`).
  * A 4-week rolling mean of sales, shifted by one period to prevent target leakage, with `bfill` on the initial gap.

### 3. Train/Test Split (`src/data_prep/split.py`)
* `time_based_split` sorts by date and splits chronologically (default 80/20) so the model is always trained on the past and evaluated on the future — no random shuffling, which would leak future information into training.
* `random_split` is also available for non-temporal experimentation but is not used in the main pipeline.

### 4. Models (`src/models/`)
All models implement a common `BaseModel` interface (`fit`, `predict`, `get_params`):
* `LinearRegressionModel`
* `RandomForestRegressorModel`
* `XGBoostModel`

### 5. Evaluation (`src/evaluate.py`)
Each model, plus a mean-prediction baseline, is scored on RMSE, MAE, R², and MAPE. Results are written to `reports/models_evaluations.csv`, sorted by RMSE.

## Latest Results

| Model | RMSE | MAE | R² | MAPE (%) |
|---|---|---|---|---|
| RandomForest | 97,994.69 | 65,385.53 | 0.967 | 6.43 |
| XGBoost | 98,237.10 | 67,924.53 | 0.967 | 6.95 |
| LinearRegression | 117,919.30 | 91,170.68 | 0.952 | 10.72 |
| Baseline (mean) | 570,716.27 | 472,384.82 | 0.000 | 67.38 |

All three models substantially outperform the mean baseline, with the tree-based models (RandomForest, XGBoost) edging out linear regression.

## Prerequisites & Installation

Requires Python 3.8+ and:

```bash
pip install pandas numpy scipy statsmodels scikit-learn xgboost fredapi kagglehub
```

Set a `FRED_API_KEY` environment variable to use your own FRED API key (a fallback key is used otherwise).

## Usage

```bash
python main.py
```

This runs data acquisition/integration, feature engineering, model training/evaluation, and writes logs to `reports/pipeline.log` and metrics to `reports/models_evaluations.csv`.

## Tests

Unit tests live in `src/tests/` and cover the split logic, model wrappers, and evaluation metrics:

```bash
pytest src/tests/
```
