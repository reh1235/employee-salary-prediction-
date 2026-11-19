# Employee Salary Prediction Using Machine Learning

Predict an employee's **annual salary (PKR)** from experience, education,
department, job role and age — with a professional dark navy-blue
Streamlit dashboard.

Built with Python, pandas, scikit-learn and Streamlit. Everything is free
and open source, and every number shown comes from the real trained model
and the real test set.

## Quick Start (Windows + VS Code)

1. **Install Python** from [python.org](https://www.python.org/downloads/)
   (3.10 or newer). During setup, tick **"Add python.exe to PATH"**.

2. **Open this folder in VS Code**, then open a terminal
   (`Terminal → New Terminal`) and create a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install the libraries:**

   ```powershell
   pip install -r requirements.txt
   ```

4. **Train the models** (already trained once — run again any time the
   data changes):

   ```powershell
   python train_model.py
   ```

5. **Start the dashboard:**

   ```powershell
   streamlit run app.py
   ```

   Then open the link shown in the terminal (usually
   http://localhost:8501).

## Dataset

`data/employee_salary_data.csv` — a realistic sample dataset of **500
employees** (PKR salaries). Replace it with your own CSV any time; just
keep the same column names, then re-run `train_model.py`.

| Column | Description |
|---|---|
| EmployeeID | Identifier only, not used for prediction |
| Age | 22–51 years (15 missing values) |
| YearsExperience | 0–25 years |
| EducationLevel | High School, Bachelor's, Master's, PhD (10 missing) |
| Department | IT, Finance, HR, Marketing, Operations |
| JobRole | 22 roles (Software Engineer, Data Analyst, …) |
| Salary | Annual salary in PKR (target) |

Salary range: PKR 500,000 – 5,054,000 per year, average about PKR 2,047,000.

## How It Works

1. **Preprocessing** (`src/preprocessing.py`): median imputation + scaling
   for numbers, most-frequent imputation + one-hot encoding for categories.
   Everything is inside a scikit-learn Pipeline fitted **only on the
   training data**, so there is no data leakage.
2. **Split:** 400 train / 100 test rows (80/20, `random_state=42`).
3. **Models:** Linear Regression and Random Forest (200 trees), trained on
   identical pipelines and compared on the same held-out test set.
4. **Best model** (highest test R²) is saved to
   `models/best_model.joblib` and used by the dashboard.

## Model Results (test set, 100 employees)

| Model | R² | MAE (PKR) | MSE (PKR²) | RMSE (PKR) |
|---|---:|---:|---:|---:|
| **Linear Regression** | **0.9448** | **163,091** | **38,509,214,416** | **196,238** |
| Random Forest | 0.9048 | 208,857 | 66,414,618,532 | 257,710 |

Best model: **Linear Regression** (R² = 0.9448). Full details, including
feature importance, are in `models/metrics.json`, and per-employee test
predictions are in `models/test_predictions.csv`.

Top salary drivers (Random Forest feature importance): YearsExperience
(0.81), EducationLevel_PhD (0.03), Age (0.03), EducationLevel_Master's
(0.02).

## Sample Prediction

Input: 5 years experience, Master's degree, IT department, Data Analyst,
age 28.

**Predicted annual salary: PKR 2,240,075**

## Dashboard Sections

- **Salary Analysis** — salary vs experience scatter with trend line,
  average pay by experience band, correlation heatmap.
- **Salary Distribution** — employees by department (pie), average salary
  by department (bar), salary spread (box plot), pay by education.
- **Model Evaluation** — R², MAE, MSE, RMSE cards, model comparison,
  feature importance, actual vs predicted scatter.
- **Predict Salary** — input form (experience, education, department,
  job role, age) with a green result card, prediction history and
  CSV download.
- **About Project** — dataset info, model info and run instructions.

## Project Structure

```
employee_salary_prediction/
├── app.py                  # Streamlit dashboard
├── train_model.py          # Trains + evaluates both models
├── requirements.txt        # All libraries
├── README.md               # This file
├── data/
│   └── employee_salary_data.csv
├── models/
│   ├── best_model.joblib       # Trained model used by the dashboard
│   ├── metrics.json            # Real evaluation results
│   └── test_predictions.csv    # Actual vs predicted (test set)
├── notebooks/
│   └── data_exploration.ipynb  # First look at the data
└── src/
    ├── __init__.py
    └── preprocessing.py        # Leakage-safe preprocessing pipeline
```

## Notes

- Predictions are estimates from the sample data, not guaranteed salaries.
- The sample data was generated with a fixed seed, so results are
  reproducible: re-running `train_model.py` gives the same metrics.
