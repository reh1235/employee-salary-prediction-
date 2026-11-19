"""
Train the Employee Salary Prediction models.

- Loads data/employee_salary_data.csv
- 80/20 train-test split (random_state=42)
- Trains Linear Regression and Random Forest on identical pipelines
  (median/most-frequent imputation + one-hot encoding + scaling,
  all fitted ONLY on the training split -> no data leakage)
- Evaluates on the held-out test set: R2, MAE, MSE, RMSE
- Saves the best model (highest test R2) to models/best_model.joblib,
  metrics to models/metrics.json and test predictions to
  models/test_predictions.csv

Run from the project root:  python train_model.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.preprocessing import TARGET, build_model_pipeline, get_feature_names

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "employee_salary_data.csv"
MODELS_DIR = BASE_DIR / "models"


def evaluate(y_true, y_pred):
    """Real evaluation metrics on the test set (no fake values)."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mse": round(float(mse), 2),
        "rmse": round(float(mse ** 0.5), 2),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {DATA_PATH.name}")
    print(f"Missing values in total: {int(df.isna().sum().sum())}")

    feature_cols = [c for c in df.columns if c not in ("EmployeeID", TARGET)]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows (80/20 split)")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    fitted = {}
    predictions = pd.DataFrame({"Actual": y_test.values})
    importances = []

    for name, model in models.items():
        pipe = build_model_pipeline(model)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        y_pred = pipe.predict(X_test)
        results[name] = evaluate(y_test, y_pred)
        predictions[name] = y_pred
        m = results[name]
        print(f"{name}: R2={m['r2']}  MAE={m['mae']:,.0f}  "
              f"MSE={m['mse']:,.0f}  RMSE={m['rmse']:,.0f}")

        if name == "Random Forest":
            rf = pipe.named_steps["model"]
            names = get_feature_names(pipe)
            ranked = sorted(
                zip(names, rf.feature_importances_),
                key=lambda t: t[1], reverse=True,
            )[:10]
            importances = [
                {
                    "feature": f.replace("num__", "").replace("cat__", ""),
                    "importance": round(float(v), 4),
                }
                for f, v in ranked
            ]

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\nBest model on the test set: {best_name} "
          f"(R2 = {results[best_name]['r2']})")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], MODELS_DIR / "best_model.joblib")
    predictions.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    metrics = {
        "best_model": best_name,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": 0.2,
        "random_state": 42,
        "target": TARGET,
        "features": feature_cols,
        "currency": "PKR",
        "models": results,
        "feature_importance": importances,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved: models/best_model.joblib, models/metrics.json, "
          f"models/test_predictions.csv")


if __name__ == "__main__":
    main()
