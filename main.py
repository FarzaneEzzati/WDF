"""
main.py

This file serves as the main pipeline for the Walmart sales prediction project. It contains the primary steps for:

* Data acquisition, integration, and feature engineering
* Specifying the machine learning models to be trained on the data
* Generating and storing reports

"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("reports/pipeline.log"),
        logging.StreamHandler()  # also prints to console
    ]
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
from src.data_prep.fetch_integrate import get_integrated_data
from src.data_prep.engineer import generate_time_features
df = get_integrated_data()
df = generate_time_features(df)


# ---------------------------------------------------------------------------
# Data Train-Test Split
# ---------------------------------------------------------------------------
from src.data_prep.split import time_based_split
X_train, X_test, y_train, y_test = time_based_split(
    df=df, date_column="Date", target_column="Weekly_Sales", test_size=0.2) 
X_train = X_train.drop(columns=["Date"])
X_test = X_test.drop(columns=["Date"])

# ---------------------------------------------------------------------------
# Defining Machine Learning models
# ---------------------------------------------------------------------------
from src.models.models import (
    LinearRegressionModel,
    LassoModel,
    ElasticNetModel,
    RandomForestRegressorModel,
    XGBoostModel,
)
models = {
    "LinearRegression": LinearRegressionModel(),
    "Lasso": LassoModel(),
    "ElasticNet": ElasticNetModel(),
    "RandomForest": RandomForestRegressorModel(n_estimators=200),
    "XGBoost": XGBoostModel(max_depth=6),
}

# ---------------------------------------------------------------------------
# Implementing Machine Learning models
# ---------------------------------------------------------------------------
from src.evaluate import evaluate, evaluate_baseline, evaluations_to_dataframe
models_evals = []
for name, model in models.items():
    logger.info(f"Model <{name}> started to fit.")
    model.fit(X_train, y_train)
    logger.info(f"Model <{name}> started to predict.")
    preds = model.predict(X_test)
    models_evals.append(evaluate(name, y_test, preds))

logger.info(f"Model <Baseline (mean)> evaluation started.")
models_evals.append(evaluate_baseline(y_train))

models_evals_df = evaluations_to_dataframe(models_evals)

eval_path = "reports/models_evaluations.csv"
logger.info(f"Models evluations stored at {eval_path}")
models_evals_df.to_csv(path_or_buf=eval_path, index=False)