"""
Employee Salary Prediction Dashboard (Streamlit).

Dark navy-blue HR analytics dashboard with four sections:
Salary Analysis, Salary Distribution, Model Evaluation and Predict Salary,
plus prediction history with CSV download and an About page.

Run from the project root:  streamlit run app.py
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "employee_salary_data.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
PREDICTIONS_PATH = BASE_DIR / "models" / "test_predictions.csv"

# ---- Theme colours ----
NAVY = "#0A1628"
CARD = "#13233F"
CARD_BORDER = "#22395A"
TEXT = "#E6EDF3"
MUTED = "#9FB0C3"
ACCENT = "#2E86DE"
TEAL = "#1ABC9C"
GREEN = "#1E9E6A"
PALETTE = ["#2E86DE", "#1ABC9C", "#F39C12", "#E74C3C", "#9B59B6", "#16A085"]

plt.rcParams.update({
    "figure.facecolor": CARD,
    "axes.facecolor": CARD,
    "axes.edgecolor": CARD_BORDER,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "axes.grid": True,
    "grid.color": CARD_BORDER,
    "grid.alpha": 0.5,
    "font.size": 11,
})

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide",
)

st.markdown(f"""
<style>
.stApp {{ background-color: {NAVY}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background-color: #0D1B31; }}
h1, h2, h3 {{ color: {TEXT}; }}
.kpi-card {{
    background: {CARD}; border: 1px solid {CARD_BORDER}; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 10px;
}}
.kpi-card .kpi-title {{ color: {MUTED}; font-size: 14px; margin-bottom: 4px; }}
.kpi-card .kpi-value {{ color: {TEXT}; font-size: 26px; font-weight: 700; }}
.kpi-card .kpi-sub {{ color: {MUTED}; font-size: 13px; }}
.result-card {{
    background: linear-gradient(135deg, #14532D, #1E9E6A);
    border-radius: 14px; padding: 26px 28px; margin: 18px 0; color: white;
}}
.result-card .result-label {{ font-size: 15px; opacity: 0.9; }}
.result-card .result-value {{ font-size: 38px; font-weight: 800; margin: 6px 0; }}
.result-card .result-sub {{ font-size: 14px; opacity: 0.9; }}
.stButton > button {{
    background-color: {ACCENT}; color: white; font-weight: 700;
    border-radius: 10px; padding: 10px 26px; border: none;
}}
.stButton > button:hover {{ background-color: #1F6FC4; color: white; }}
</style>
""", unsafe_allow_html=True)


# ---- Data / model loading ----
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


@st.cache_data
def load_test_predictions():
    if not PREDICTIONS_PATH.exists():
        return None
    return pd.read_csv(PREDICTIONS_PATH)


def fmt_pkr(value):
    return f"PKR {value:,.0f}"


def millions_formatter(value, _pos):
    return f"{value / 1e6:.1f}M"


def kpi_card(title, value, sub=""):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style_plot(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, color=TEXT, fontsize=13, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)


# ---- Load data ----
try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Make sure data/employee_salary_data.csv exists.")
    st.stop()

model = load_model()
metrics = load_metrics()

# ---- Sidebar ----
st.sidebar.title("💼 Salary Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["📈 Salary Analysis", "📊 Salary Distribution", "🧪 Model Evaluation",
     "🔮 Predict Salary", "ℹ️ About Project"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Employees:** {len(df):,}")
st.sidebar.markdown(f"**Departments:** {df['Department'].nunique()}")
st.sidebar.markdown(f"**Job roles:** {df['JobRole'].nunique()}")
if metrics:
    st.sidebar.markdown(f"**Best model:** {metrics['best_model']}")
    st.sidebar.markdown(f"**Test R²:** {metrics['models'][metrics['best_model']]['r2']}")

st.title("Employee Salary Prediction")
st.markdown("HR analytics dashboard — explore salaries, compare models and "
            "predict pay for new employees.")
st.markdown("---")


# ============================ SALARY ANALYSIS ============================
if page == "📈 Salary Analysis":
    st.header("Salary Analysis")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Average Salary", fmt_pkr(df["Salary"].mean()), "per year")
    with c2:
        kpi_card("Median Salary", fmt_pkr(df["Salary"].median()), "per year")
    with c3:
        kpi_card("Highest Salary", fmt_pkr(df["Salary"].max()),
                 df.loc[df["Salary"].idxmax(), "JobRole"])
    with c4:
        kpi_card("Average Experience", f"{df['YearsExperience'].mean():.1f} years",
                 f"Age range {int(df['Age'].min())}–{int(df['Age'].max())}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Salary vs Experience")
        clean = df.dropna(subset=["YearsExperience", "Salary"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(clean["YearsExperience"], clean["Salary"],
                   c=ACCENT, alpha=0.55, s=28)
        slope, intercept = np.polyfit(clean["YearsExperience"], clean["Salary"], 1)
        xs = np.array([clean["YearsExperience"].min(), clean["YearsExperience"].max()])
        ax.plot(xs, slope * xs + intercept, color=TEAL, linewidth=2.5,
                label=f"Trend: +{fmt_pkr(slope)} / year")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
        ax.legend(facecolor=CARD, edgecolor=CARD_BORDER, labelcolor=TEXT)
        style_plot(ax, "Salary grows with experience",
                   "Years of Experience", "Annual Salary (PKR)")
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Average Salary by Experience Band")
        bands = pd.cut(df["YearsExperience"], bins=[-1, 2, 5, 10, 15, 25],
                       labels=["0–2 yrs", "3–5 yrs", "6–10 yrs", "11–15 yrs", "16+ yrs"])
        avg_by_band = df.groupby(bands, observed=True)["Salary"].mean()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(avg_by_band.index.astype(str), avg_by_band.values, color=TEAL)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
        for bar, val in zip(ax.patches, avg_by_band.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val / 1e6:.2f}M", ha="center", va="bottom", color=TEXT, fontsize=10)
        style_plot(ax, "More experience, higher average pay",
                   "Experience Band", "Average Annual Salary (PKR)")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation: Age, Experience and Salary")
    corr = df[["Age", "YearsExperience", "Salary"]].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"color": "white"})
    ax.set_title("Experience is the strongest salary driver", color=TEXT, pad=12)
    st.pyplot(fig)
    plt.close(fig)


# ========================== SALARY DISTRIBUTION ==========================
elif page == "📊 Salary Distribution":
    st.header("Salary Distribution")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Employees by Department")
        dept_counts = df["Department"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        wedges, _, autotexts = ax.pie(
            dept_counts.values, labels=dept_counts.index, autopct="%1.1f%%",
            colors=PALETTE, startangle=90, pctdistance=0.8,
            textprops={"color": TEXT, "fontsize": 11},
            wedgeprops={"edgecolor": CARD, "linewidth": 2},
        )
        for t in autotexts:
            t.set_color("white")
            t.set_fontsize(10)
        style_plot(ax, "Workforce share by department")
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Average Salary by Department")
        avg_dept = df.groupby("Department")["Salary"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(avg_dept.index, avg_dept.values, color=PALETTE[:len(avg_dept)])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
        for bar, val in zip(ax.patches, avg_dept.values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {fmt_pkr(val)}", va="center", color=TEXT, fontsize=10)
        style_plot(ax, "IT pays the most on average",
                   "Average Annual Salary (PKR)", "")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Salary Spread by Department")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=df, x="Department", y="Salary", hue="Department",
                palette=PALETTE[:5], legend=False, ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
    style_plot(ax, "Distribution of salaries inside each department",
               "Department", "Annual Salary (PKR)")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Average Salary by Education Level")
    order = ["High School", "Bachelor's", "Master's", "PhD"]
    avg_edu = df.groupby("EducationLevel")["Salary"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(avg_edu.index, avg_edu.values, color=ACCENT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
    for bar, val in zip(ax.patches, avg_edu.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val / 1e6:.2f}M", ha="center", va="bottom", color=TEXT, fontsize=10)
    style_plot(ax, "Higher education, higher average pay",
               "Education Level", "Average Annual Salary (PKR)")
    st.pyplot(fig)
    plt.close(fig)


# ============================ MODEL EVALUATION ============================
elif page == "🧪 Model Evaluation":
    st.header("Model Evaluation")
    if metrics is None:
        st.error("No evaluation results found. Run `python train_model.py` first.")
        st.stop()

    best = metrics["best_model"]
    best_m = metrics["models"][best]
    st.success(f"Best model on the held-out test set ({metrics['test_rows']} employees): "
               f"**{best}**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("R² Score", f"{best_m['r2']:.4f}", "higher is better (max 1.0)")
    with c2:
        kpi_card("MAE", fmt_pkr(best_m["mae"]), "average error")
    with c3:
        kpi_card("MSE", fmt_pkr(best_m["mse"]), "mean squared error")
    with c4:
        kpi_card("RMSE", fmt_pkr(best_m["rmse"]), "typical error size")

    st.subheader("Model Comparison (test set)")
    comp = pd.DataFrame(metrics["models"]).T.reset_index().rename(
        columns={"index": "Model"})
    comp[["r2", "mae", "mse", "rmse"]] = comp[["r2", "mae", "mse", "rmse"]].round(2)
    st.dataframe(comp, width="stretch", hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("R² Comparison")
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(metrics["models"].keys())
        r2s = [metrics["models"][n]["r2"] for n in names]
        ax.bar(names, r2s, color=[TEAL, ACCENT])
        ax.set_ylim(0, 1.05)
        for bar, val in zip(ax.patches, r2s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.4f}", ha="center", color=TEXT, fontsize=11)
        style_plot(ax, "Random Forest vs Linear Regression", "", "R² Score")
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Feature Importance (Random Forest)")
        imp = pd.DataFrame(metrics["feature_importance"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(imp["feature"][::-1], imp["importance"][::-1], color=TEAL)
        style_plot(ax, "What drives salary most", "Importance", "")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Actual vs Predicted Salary (test set)")
    preds = load_test_predictions()
    if preds is not None and best in preds.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(preds["Actual"], preds[best], c=ACCENT, alpha=0.6, s=30)
        lo = min(preds["Actual"].min(), preds[best].min())
        hi = max(preds["Actual"].max(), preds[best].max())
        ax.plot([lo, hi], [lo, hi], color="#E74C3C", linestyle="--",
                linewidth=2, label="Perfect prediction")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
        ax.legend(facecolor=CARD, edgecolor=CARD_BORDER, labelcolor=TEXT)
        style_plot(ax, f"{best}: predictions track actual salaries closely",
                   "Actual Salary (PKR)", "Predicted Salary (PKR)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Test predictions file not found; retrain with train_model.py.")


# ============================== PREDICT SALARY ==============================
elif page == "🔮 Predict Salary":
    st.header("Predict Salary")
    if model is None:
        st.error("Trained model not found. Run `python train_model.py` first, "
                 "then refresh this page.")
        st.stop()

    roles_by_dept = (df.groupby("Department")["JobRole"].unique()
                     .apply(sorted).to_dict())
    departments = sorted(roles_by_dept.keys())
    education_levels = ["High School", "Bachelor's", "Master's", "PhD"]

    # No st.form here on purpose: widgets inside a form do not trigger a rerun,
    # so the Job Role list would not refresh when Department changes. With plain
    # widgets, picking a department immediately updates the roles below.
    col1, col2 = st.columns(2)
    with col1:
        experience = st.slider("Years of Experience", 0, 25, 5)
        education = st.selectbox("Education Level", education_levels, index=1)
    with col2:
        age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
        department = st.selectbox("Department", departments)
    job_role = st.selectbox("Job Role", roles_by_dept[department])

    if st.button("💰 Predict Salary"):
        if experience > age - 18:
            st.warning("Experience looks high for this age — please double-check "
                       "the inputs. Showing the prediction anyway.")
        new_employee = pd.DataFrame([{
            "Age": age,
            "YearsExperience": experience,
            "EducationLevel": education,
            "Department": department,
            "JobRole": job_role,
        }])
        predicted = float(model.predict(new_employee)[0])
        # Keep the result in session state so it stays visible if the user
        # changes an input afterwards.
        st.session_state.last_prediction = {
            "predicted": predicted,
            "age": age,
            "experience": experience,
            "education": education,
            "department": department,
            "job_role": job_role,
        }
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Age": age,
            "YearsExperience": experience,
            "EducationLevel": education,
            "Department": department,
            "JobRole": job_role,
            "PredictedSalaryPKR": round(predicted),
        })

    if "last_prediction" in st.session_state:
        last = st.session_state.last_prediction
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">Predicted Annual Salary</div>'
            f'<div class="result-value">{fmt_pkr(last["predicted"])}</div>'
            f'<div class="result-sub">{last["job_role"]} · {last["department"]} · '
            f'{last["education"]} · {last["experience"]} years experience · Age {last["age"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Prediction by {metrics['best_model'] if metrics else 'the trained model'} "
                   f"— an estimate based on the sample dataset, not a guaranteed salary.")

    st.subheader("Prediction History")
    if "history" in st.session_state and st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, width="stretch", hide_index=True)
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download history as CSV", data=csv,
                           file_name="salary_predictions.csv",
                           mime="text/csv")
    else:
        st.info("No predictions yet — use the form above and your results will "
                "appear here.")


# ================================ ABOUT =================================
elif page == "ℹ️ About Project":
    st.header("About This Project")
    st.markdown(
        "This application predicts an employee's **annual salary (PKR)** from "
        "their experience, education, department, job role and age, using "
        "machine learning trained on employee data. It is a learning project: "
        "every number shown comes from the real trained model and the real "
        "test set — nothing is faked."
    )

    st.subheader("Dataset Information")
    info = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(t) for t in df.dtypes],
        "Missing": df.isna().sum().values,
        "Unique values": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(info, width="stretch", hide_index=True)
    st.markdown(f"**Rows:** {len(df):,} employees · **Target:** Salary (PKR/year) · "
                f"**Features:** Age, YearsExperience, EducationLevel, Department, JobRole")

    st.subheader("Model Information")
    if metrics:
        st.markdown(f"- **Models trained:** Linear Regression and Random Forest "
                    f"(200 trees)")
        st.markdown(f"- **Split:** {metrics['train_rows']} train / "
                    f"{metrics['test_rows']} test rows (80/20, random_state=42)")
        st.markdown(f"- **Best model:** {metrics['best_model']} "
                    f"(R² = {metrics['models'][metrics['best_model']]['r2']})")
        st.markdown("- **Preprocessing:** median imputation + scaling for numbers, "
                    "most-frequent imputation + one-hot encoding for categories — "
                    "fitted on training data only (no data leakage)")
    else:
        st.info("Train the model first (`python train_model.py`) to see model details.")

    st.subheader("How to Run (Windows + VS Code)")
    st.code(
        "python -m venv .venv\n"
        ".venv\\Scripts\\activate\n"
        "pip install -r requirements.txt\n"
        "python train_model.py\n"
        "streamlit run app.py",
        language="powershell",
    )
    st.caption("Built with Python, pandas, scikit-learn and Streamlit — "
               "all free and open source.")
