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
│   │   ├── engineer.py          # time-aware feature engineering
│   │   ├── split.py             # chronological (and random) train/test split
│   │   └── statistical_analysis.ipynb  # exploratory statistical tests
│   ├── models/
│   │   ├── base.py              # BaseModel interface (fit/predict/get_params)
│   │   └── models.py            # all model wrappers (linear and tree)
│   ├── evaluate.py              # RMSE, MAE, R², MAPE
│   └── tests/                   # unit tests for split, models, evaluate
│
├── reports/
│   ├── pipeline.log             # run log
│   └── models_evaluations.csv   # metrics per model, sorted by RMSE
│
├── test.ipynb                   # notebook test runner (all or selected tests)
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
* Parses Walmart dates explicitly as day-month-year (`%d-%m-%Y`). Every week ends on a Friday.
* Merges monthly indicators onto Walmart sales via an exact `YearMonth` join, and merges the quarterly `TDSP` series via `pd.merge_asof` (backward direction) to avoid lookahead bias.

### 2. Feature Engineering (`src/data_prep/engineer.py`)
* Cyclical week-of-year encoding (`sin_week`, `cos_week`).
* A 4-week rolling mean of sales per store, shifted by one period to prevent target leakage, with `bfill` on the initial gap.

Statistical analysis is kept out of the pipeline, in `src/data_prep/statistical_analysis.ipynb`. It tests within-store normality (Shapiro-Wilk), cross-store variance homogeneity (Levene), store differences (Kruskal-Wallis) and feature significance (Spearman on per-store Z-scored sales). Its findings guide changes to the pipeline rather than running on every execution.

### 3. Train/Test Split (`src/data_prep/split.py`)
* `time_based_split` sorts by date and splits chronologically (default 80/20) so the model is always trained on the past and evaluated on the future — no random shuffling, which would leak future information into training.
* `random_split` is also available for non-temporal experimentation but is not used in the main pipeline.

### 4. Models (`src/models/`)
All model wrappers live in `src/models/models.py` and implement a common `BaseModel` interface (`fit`, `predict`, `get_params`):
* Linear models: `LinearRegressionModel`, `LassoModel`, `ElasticNetModel`. Each one-hot encodes `Store`. Lasso and ElasticNet also standardise the other features so their penalty treats every feature equally.
* Tree models: `RandomForestRegressorModel`, `XGBoostModel`. They use the features as they are.

### 5. Evaluation (`src/evaluate.py`)
Each model, plus a mean-prediction baseline, is scored on RMSE, MAE, R², and MAPE. Results are written to `reports/models_evaluations.csv`, sorted by RMSE.

## Latest Results

Test-set scores on the most recent 20% of weeks, from `reports/models_evaluations.csv`:

| Model | RMSE | MAE | R² | MAPE (%) |
|---|---|---|---|---|
| LinearRegression | 70,742.19 | 50,083.21 | 0.982 | 4.92 |
| XGBoost | 73,497.20 | 54,993.44 | 0.981 | 5.97 |
| RandomForest | 73,779.40 | 52,480.19 | 0.981 | 5.16 |
| Lasso | 112,287.21 | 85,019.81 | 0.956 | 10.19 |
| ElasticNet | 190,897.12 | 159,973.23 | 0.872 | 23.07 |
| Baseline (mean) | 571,873.16 | 473,115.57 | 0.000 | 67.41 |

* Every model clearly beats the mean baseline.
* Linear regression with one-hot encoded stores is the best model on every metric. XGBoost and RandomForest are close behind. XGBoost has slightly lower RMSE, while RandomForest has lower MAE and MAPE.
* Lasso and ElasticNet use default penalty strengths that have not been tuned yet, which is why they trail plain linear regression. Tuning them with time-ordered cross-validation is the next step.

## Prerequisites & Installation

Requires Python 3.8+ and:

```bash
pip install pandas numpy scipy statsmodels scikit-learn xgboost fredapi kagglehub pytest
```

Set a `FRED_API_KEY` environment variable to use your own FRED API key (a fallback key is used otherwise).

## Usage

Run from the project root, because data paths such as `data/raw/walmart` are relative to it:

```bash
python main.py
```

This runs data acquisition/integration, feature engineering, model training/evaluation, and writes logs to `reports/pipeline.log` and metrics to `reports/models_evaluations.csv`.

## Tests

Unit tests live in `src/tests/` and cover the split logic, model wrappers, and evaluation metrics. Model tests run against all five model wrappers using synthetic regression data, and check that each model fits, predicts with the right shape and finite values, and beats a mean baseline.

Run them from the project root:

```bash
python -m pytest src/tests/                                              # all tests
python -m pytest src/tests/test_models.py                                # one file
python -m pytest src/tests/test_models.py::test_get_params_returns_dict  # one test
python -m pytest src/tests/ -k "shape"                                   # filter by name
```

### Running tests from the notebook

`test.ipynb` runs the same tests without leaving Jupyter. Edit the settings at the top of the test runner cell and run it:

* `TESTS`: an empty list runs all tests, or list files and single tests such as `"src/tests/test_models.py::test_get_params_returns_dict"`.
* `KEYWORD`: an optional `-k` name filter.
* `VERBOSE` and `STOP_ON_FIRST`: toggle `-v` and `-x`.
* `LIST_ONLY`: set to `True` to print every available test name without running anything.

Output streams into the notebook as the tests run, and the final line shows the pytest exit code, where 0 means all tests passed.
