"""
ChurnGuard AI 2.1
Predictive churn scoring and retention dashboard for the Telco churn dataset.

What this script does:
    1. Loads the local CSV or Excel file instead of generating synthetic data.
    2. Cleans and preprocesses the customer records.
    3. Trains churn and survival models on the real file contents.
    4. Serves a structured localhost dashboard with customer summaries.
    5. Keeps API endpoints available for programmatic access.

Run locally:
    python churnguard.py

Open the dashboard at:
    http://127.0.0.1:8000
"""

from __future__ import annotations

import html
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from lifelines import CoxPHFitter
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_FILE = Path(__file__).with_name("WA_Fn-UseC_-Telco-Customer-Churn.csv")
DEFAULT_PORT = 8000
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

CLASSIFICATION_FEATURE_COLUMNS: list[str] = []
SURVIVAL_FEATURE_COLUMNS: list[str] = []
raw_df: Optional[pd.DataFrame] = None
feature_df: Optional[pd.DataFrame] = None
churn_model: Optional[Pipeline] = None
model_comparison: pd.DataFrame = pd.DataFrame()
split_data: Optional[tuple] = None
X_train_split: Optional[pd.DataFrame] = None
survival_model: Optional[CoxPHFitter] = None
survival_preprocessor: Optional[ColumnTransformer] = None
scored_customers_df: pd.DataFrame = pd.DataFrame()
cached_profiles: dict[str, dict] = {}
active_data_file: Path = DATA_FILE

BASE_ALLOWED_ACTIONS = [
    "10% Loyalty Discount",
    "15% Loyalty Discount",
    "Proactive Support Call",
    "Plan / Usage Upgrade Offer",
    "No Incentive - Monitor Only",
]


def initialize_churnguard(data_path: Optional[str] = None, top_n: int = 12, force: bool = False) -> Path:
    global raw_df
    global feature_df
    global CLASSIFICATION_FEATURE_COLUMNS
    global SURVIVAL_FEATURE_COLUMNS
    global churn_model
    global model_comparison
    global split_data
    global X_train_split
    global survival_model
    global survival_preprocessor
    global scored_customers_df
    global cached_profiles
    global active_data_file

    source = resolve_data_path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")

    if (
        not force
        and feature_df is not None
        and churn_model is not None
        and survival_model is not None
        and active_data_file.resolve() == source.resolve()
    ):
        return source

    print(f"Loading data from {source.name} and training ChurnGuard models...")
    loaded_df = load_customer_data(str(source))
    cleaned_df = clean_customer_data(loaded_df)

    classification_columns = get_classification_feature_columns(cleaned_df)
    survival_columns = get_survival_feature_columns(cleaned_df)

    if not classification_columns:
        raise ValueError("No classification features were found after cleaning the dataset")
    if not survival_columns:
        raise ValueError("No survival features were found after cleaning the dataset")

    CLASSIFICATION_FEATURE_COLUMNS = classification_columns
    SURVIVAL_FEATURE_COLUMNS = survival_columns

    trained_churn_model, trained_comparison, trained_split = train_classification_models(cleaned_df)
    trained_survival_model, trained_survival_preprocessor = train_survival_model(cleaned_df)

    raw_df = loaded_df
    feature_df = cleaned_df
    churn_model = trained_churn_model
    model_comparison = trained_comparison
    split_data = trained_split
    X_train_split = trained_split[0]
    survival_model = trained_survival_model
    survival_preprocessor = trained_survival_preprocessor
    active_data_file = source
    scored_customers_df, cached_profiles = build_customer_cache(top_n)

    print("Models ready.")
    return source


def ensure_initialized() -> None:
    if feature_df is None or churn_model is None or survival_model is None or survival_preprocessor is None:
        initialize_churnguard()


def resolve_data_path(path: Optional[str]) -> Path:
    if not path:
        return DATA_FILE

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(__file__).parent / candidate
    return candidate


def load_customer_data(path: Optional[str] = None) -> pd.DataFrame:
    source = resolve_data_path(path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(source)

    raise ValueError("Supported input formats are .csv, .xls, and .xlsx")


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [column.strip() for column in out.columns]

    for column in out.select_dtypes(include="object").columns:
        out[column] = out[column].astype(str).str.strip()
        out[column] = out[column].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    if TARGET_COLUMN not in out.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' was not found")
    if ID_COLUMN not in out.columns:
        raise ValueError(f"Expected id column '{ID_COLUMN}' was not found")

    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        if "tenure" in out.columns and "MonthlyCharges" in out.columns:
            fallback = out["tenure"].fillna(0) * out["MonthlyCharges"].fillna(0)
            out["TotalCharges"] = out["TotalCharges"].fillna(fallback)
        out["TotalCharges"] = out["TotalCharges"].fillna(out["TotalCharges"].median())

    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    out = out.dropna(subset=[TARGET_COLUMN]).copy()
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)

    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    numeric_defaults = ["tenure", "MonthlyCharges", "TotalCharges"]
    for column in numeric_defaults:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            if out[column].isna().any():
                out[column] = out[column].fillna(out[column].median())

    for column in out.select_dtypes(include="object").columns:
        out[column] = out[column].fillna("Unknown")

    return out.reset_index(drop=True)


def get_classification_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in {ID_COLUMN, TARGET_COLUMN}]


def get_survival_feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "SeniorCitizen",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaperlessBilling",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        "Partner",
        "Dependents",
    ]
    return [column for column in preferred if column in df.columns]


def build_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scale", StandardScaler())]), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        remainder="drop",
    )


def make_model_pipeline(df: pd.DataFrame, model) -> Pipeline:
    preprocessor = build_preprocessor(df, CLASSIFICATION_FEATURE_COLUMNS)
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def train_classification_models(df: pd.DataFrame):
    X = df[CLASSIFICATION_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
        ),
    }

    results: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipeline = make_model_pipeline(df, estimator)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results.append(
            {
                "model": name,
                "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
                "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
                "recall": round(float(recall_score(y_test, predictions)), 4),
                "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            }
        )
        fitted_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        by=["roc_auc", "recall"],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted_models[str(best_model_name)]

    return best_model, results_df, (X_train, X_test, y_train, y_test)


def transform_dataframe(preprocessor: ColumnTransformer, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    transformed = preprocessor.transform(df[feature_columns])
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=df.index)


def train_survival_model(df: pd.DataFrame):
    survival_df = df[SURVIVAL_FEATURE_COLUMNS + ["tenure", TARGET_COLUMN]].copy()
    survival_preprocessor = build_preprocessor(df, SURVIVAL_FEATURE_COLUMNS)
    survival_preprocessor.fit(survival_df[SURVIVAL_FEATURE_COLUMNS])

    design_matrix = transform_dataframe(survival_preprocessor, survival_df, SURVIVAL_FEATURE_COLUMNS)
    design_matrix["tenure"] = survival_df["tenure"].astype(float)
    design_matrix[TARGET_COLUMN] = survival_df[TARGET_COLUMN].astype(int)

    cox = CoxPHFitter(penalizer=0.08)
    cox.fit(design_matrix, duration_col="tenure", event_col=TARGET_COLUMN)
    return cox, survival_preprocessor


def urgency_bucket(expected_days: float) -> str:
    if expected_days <= 30:
        return "High priority - this week"
    if expected_days <= 90:
        return "Monitor - this month"
    return "Lower immediate priority"


def estimate_time_to_churn(cox_model: CoxPHFitter, survival_preprocessor: ColumnTransformer, customer_row: pd.DataFrame) -> float:
    matrix = transform_dataframe(survival_preprocessor, customer_row, SURVIVAL_FEATURE_COLUMNS)
    median_survival = cox_model.predict_median(matrix)
    value = median_survival.values[0] if hasattr(median_survival, "values") else float(median_survival)
    if np.isnan(value) or np.isinf(value):
        value = 365.0
    return round(float(value), 1)


def build_customer_drivers(customer_row: pd.DataFrame) -> list[dict]:
    row = customer_row.iloc[0]
    drivers: list[dict] = []

    tenure = float(row.get("tenure", 0))
    monthly = float(row.get("MonthlyCharges", 0))
    contract = str(row.get("Contract", "Unknown"))
    payment_method = str(row.get("PaymentMethod", "Unknown"))

    if tenure <= 6:
        drivers.append({"feature": "tenure", "value": tenure, "impact": "increases risk", "score": 3.0})
    elif tenure <= 24:
        drivers.append({"feature": "tenure", "value": tenure, "impact": "moderately increases risk", "score": 2.0})

    if contract == "Month-to-month":
        drivers.append({"feature": "Contract", "value": contract, "impact": "increases risk", "score": 3.2})
    elif contract == "One year":
        drivers.append({"feature": "Contract", "value": contract, "impact": "slightly decreases risk", "score": 0.8})
    elif contract == "Two year":
        drivers.append({"feature": "Contract", "value": contract, "impact": "decreases risk", "score": 1.6})

    if monthly >= 80:
        drivers.append({"feature": "MonthlyCharges", "value": monthly, "impact": "increases risk", "score": 2.4})
    elif monthly >= 60:
        drivers.append({"feature": "MonthlyCharges", "value": monthly, "impact": "moderately increases risk", "score": 1.4})

    if str(row.get("TechSupport", "Unknown")) in {"No", "No internet service"}:
        drivers.append({"feature": "TechSupport", "value": row.get("TechSupport", "Unknown"), "impact": "increases risk", "score": 1.8})

    if str(row.get("OnlineSecurity", "Unknown")) in {"No", "No internet service"}:
        drivers.append({"feature": "OnlineSecurity", "value": row.get("OnlineSecurity", "Unknown"), "impact": "increases risk", "score": 1.6})

    if payment_method == "Electronic check":
        drivers.append({"feature": "PaymentMethod", "value": payment_method, "impact": "increases risk", "score": 1.2})

    if str(row.get("PaperlessBilling", "Unknown")) == "Yes":
        drivers.append({"feature": "PaperlessBilling", "value": "Yes", "impact": "slightly increases risk", "score": 0.9})

    if not drivers:
        drivers.append({"feature": "portfolio", "value": "balanced profile", "impact": "neutral", "score": 0.0})

    drivers.sort(key=lambda item: abs(item["score"]), reverse=True)
    return drivers[:3]


def estimate_retention_uplift(customer_row: pd.DataFrame, churn_prob: float) -> float:
    row = customer_row.iloc[0]

    tenure = float(row.get("tenure", 0))
    monthly = float(row.get("MonthlyCharges", 0))
    contract = str(row.get("Contract", "Unknown"))
    support = str(row.get("TechSupport", "Unknown"))
    security = str(row.get("OnlineSecurity", "Unknown"))
    payment_method = str(row.get("PaymentMethod", "Unknown"))
    paperless = str(row.get("PaperlessBilling", "Unknown"))

    short_tenure = 1.0 if tenure <= 12 else 0.0
    high_charge = min(monthly / 120.0, 1.0)
    contract_risk = {"Month-to-month": 1.0, "One year": 0.35, "Two year": 0.0}.get(contract, 0.25)
    service_friction = 0.0
    service_friction += 0.5 if support in {"No", "No internet service"} else 0.0
    service_friction += 0.35 if security in {"No", "No internet service"} else 0.0
    service_friction += 0.15 if payment_method == "Electronic check" else 0.0
    service_friction += 0.1 if paperless == "Yes" else 0.0

    score = (
        0.35 * churn_prob
        + 0.18 * short_tenure
        + 0.16 * high_charge
        + 0.12 * contract_risk
        + 0.12 * service_friction
        - 0.10 * (1.0 if tenure >= 48 else 0.0)
    )
    return round(float(score - 0.25), 4)


ALLOWED_ACTIONS = BASE_ALLOWED_ACTIONS

ACTION_COST = {
    "10% Loyalty Discount": lambda monthly_charge: round(monthly_charge * 0.10 * 3, 2),
    "15% Loyalty Discount": lambda monthly_charge: round(monthly_charge * 0.15 * 3, 2),
    "Proactive Support Call": lambda monthly_charge: 8.0,
    "Plan / Usage Upgrade Offer": lambda monthly_charge: 12.0,
    "No Incentive - Monitor Only": lambda monthly_charge: 0.0,
}


def recommend_action(churn_prob: float, expected_days: float, uplift: float, shap_drivers: List[dict], monthly_charge: float) -> dict:
    if uplift <= 0.03:
        action = "No Incentive - Monitor Only"
        reason = "The predicted retention response is too small to justify a spend."
    else:
        top_driver_names = [driver["feature"] for driver in shap_drivers]

        if "TechSupport" in top_driver_names and churn_prob > 0.5:
            action = "Proactive Support Call"
            reason = "Support friction appears to be the main issue, so a human call is the cleanest next step."
        elif "Contract" in top_driver_names and churn_prob > 0.5:
            action = "Plan / Usage Upgrade Offer"
            reason = "The contract pattern suggests the current plan may not fit the customer well enough."
        elif "MonthlyCharges" in top_driver_names and uplift > 0.08:
            action = "15% Loyalty Discount"
            reason = "Price pressure is a leading factor and the customer looks responsive to a discount."
        elif uplift > 0.05:
            action = "10% Loyalty Discount"
            reason = "A modest retention offer is justified by the response score."
        else:
            action = "No Incentive - Monitor Only"
            reason = "The risk is real, but the estimated response does not support an offer cost."

    return {
        "recommended_action": action,
        "estimated_cost": ACTION_COST[action](monthly_charge),
        "reason": reason,
        "urgency": urgency_bucket(expected_days),
    }


RETENTION_VALUE_MONTHS = 12
ASSUMED_SUCCESS_MULTIPLIER = 1.0


def calculate_roi(monthly_charge: float, uplift: float, intervention_cost: float) -> dict:
    revenue_at_risk = round(monthly_charge * RETENTION_VALUE_MONTHS, 2)
    expected_recovery = round(revenue_at_risk * max(uplift, 0) * ASSUMED_SUCCESS_MULTIPLIER, 2)
    net_recovery = round(expected_recovery - intervention_cost, 2)
    roi = round(net_recovery / intervention_cost, 2) if intervention_cost > 0 else None

    return {
        "revenue_at_risk": revenue_at_risk,
        "expected_recovery": expected_recovery,
        "intervention_cost": intervention_cost,
        "net_expected_recovery": net_recovery,
        "roi": roi,
    }


def build_full_customer_profile(customer_row: pd.DataFrame) -> dict:
    ensure_initialized()
    churn_prob = float(churn_model.predict_proba(customer_row[CLASSIFICATION_FEATURE_COLUMNS])[:, 1][0])
    expected_days = estimate_time_to_churn(survival_model, survival_preprocessor, customer_row)
    drivers = build_customer_drivers(customer_row)
    uplift = estimate_retention_uplift(customer_row, churn_prob)
    monthly_charge = float(customer_row["MonthlyCharges"].values[0])

    return {
        "customer_id": customer_row[ID_COLUMN].values[0],
        "churn_probability": round(churn_prob, 4),
        "expected_days_to_churn": expected_days,
        "top_drivers": drivers,
        "uplift_score": uplift,
        "monthly_charge": monthly_charge,
    }


def build_customer_profile_from_row(row: pd.Series, churn_prob: float) -> dict:
    ensure_initialized()
    customer_row = row.to_frame().T
    expected_days = estimate_time_to_churn(survival_model, survival_preprocessor, customer_row)
    drivers = build_customer_drivers(customer_row)
    uplift = estimate_retention_uplift(customer_row, churn_prob)
    monthly_charge = float(row["MonthlyCharges"])

    return {
        "customer_id": row[ID_COLUMN],
        "churn_probability": round(float(churn_prob), 4),
        "expected_days_to_churn": expected_days,
        "top_drivers": drivers,
        "uplift_score": uplift,
        "monthly_charge": monthly_charge,
    }


def score_customer_by_id(customer_id: str) -> Optional[dict]:
    ensure_initialized()
    row = feature_df[feature_df[ID_COLUMN] == customer_id]
    if row.empty:
        return None
    return build_full_customer_profile(row)


def build_customer_cache(top_n: int = 12) -> tuple[pd.DataFrame, dict[str, dict]]:
    ensure_initialized()
    probabilities = churn_model.predict_proba(feature_df[CLASSIFICATION_FEATURE_COLUMNS])[:, 1]
    ranking_frame = feature_df[[ID_COLUMN, "MonthlyCharges", "tenure"]].copy()
    ranking_frame["churn_probability"] = probabilities
    ranking_frame = ranking_frame.sort_values(
        by=["churn_probability", "MonthlyCharges"],
        ascending=[False, False],
    ).reset_index(drop=True)

    cached_profiles: dict[str, dict] = {}
    for _, row in ranking_frame.head(top_n).iterrows():
        customer_row = feature_df[feature_df[ID_COLUMN] == row[ID_COLUMN]]
        cached_profiles[row[ID_COLUMN]] = build_full_customer_profile(customer_row)

    return ranking_frame, cached_profiles


def get_high_risk_customer_rows(limit: int = 12) -> pd.DataFrame:
    ensure_initialized()
    return scored_customers_df.head(limit).reset_index(drop=True)


def build_scored_customers_frame(top_n: int = 12) -> pd.DataFrame:
    ensure_initialized()
    scored_rows = []
    probabilities = churn_model.predict_proba(feature_df[CLASSIFICATION_FEATURE_COLUMNS])[:, 1]
    ranking_frame = feature_df[[ID_COLUMN, "MonthlyCharges"]].copy()
    ranking_frame["churn_probability"] = probabilities
    ranking_frame = ranking_frame.sort_values(
        by=["churn_probability", "MonthlyCharges"],
        ascending=[False, False],
    ).head(top_n).reset_index(drop=True)

    for _, row in ranking_frame.iterrows():
        customer_row = feature_df[feature_df[ID_COLUMN] == row[ID_COLUMN]]
        profile = build_full_customer_profile(customer_row)
        action = recommend_action(
            profile["churn_probability"],
            profile["expected_days_to_churn"],
            profile["uplift_score"],
            profile["top_drivers"],
            profile["monthly_charge"],
        )
        scored_rows.append(
            {
                "customer_id": profile["customer_id"],
                "churn_probability": profile["churn_probability"],
                "expected_days_to_churn": profile["expected_days_to_churn"],
                "urgency": action["urgency"],
                "recommended_action": action["recommended_action"],
                "estimated_cost": action["estimated_cost"],
                "monthly_charge": profile["monthly_charge"],
            }
        )

    return pd.DataFrame(scored_rows).reset_index(drop=True)


def render_dashboard(selected_customer_id: Optional[str] = None) -> str:
    ensure_initialized()
    selected_profile = score_customer_by_id(selected_customer_id) if selected_customer_id else None
    selected_action = None
    selected_roi = None
    selected_message = None

    if selected_customer_id:
        if selected_profile is None:
            selected_message = f"No customer found for id {html.escape(selected_customer_id)}."
        else:
            selected_action = recommend_action(
                selected_profile["churn_probability"],
                selected_profile["expected_days_to_churn"],
                selected_profile["uplift_score"],
                selected_profile["top_drivers"],
                selected_profile["monthly_charge"],
            )
            selected_roi = calculate_roi(
                selected_profile["monthly_charge"],
                selected_profile["uplift_score"],
                selected_action["estimated_cost"],
            )

    total_customers = len(feature_df)
    churn_rate = round(float(feature_df[TARGET_COLUMN].mean() * 100), 2)
    avg_monthly = round(float(feature_df["MonthlyCharges"].mean()), 2)
    avg_tenure = round(float(feature_df["tenure"].mean()), 2)
    high_risk_count = int((feature_df[TARGET_COLUMN] == 1).sum())
    high_risk_rows = get_high_risk_customer_rows(12)

    selected_html = ""
    if selected_profile is not None and selected_action is not None and selected_roi is not None:
        driver_lines = "".join(
            f"<li><strong>{html.escape(str(driver['feature']))}</strong>: {html.escape(str(driver['value']))} - {html.escape(str(driver['impact']))}</li>"
            for driver in selected_profile["top_drivers"]
        )
        selected_html = f"""
        <section class="panel section">
            <h2>Customer Detail</h2>
            <div class="detail-grid">
                <div class="detail-card"><span>Customer ID</span><strong>{html.escape(str(selected_profile['customer_id']))}</strong></div>
                <div class="detail-card"><span>Churn Probability</span><strong>{selected_profile['churn_probability'] * 100:.1f}%</strong></div>
                <div class="detail-card"><span>Expected Days to Churn</span><strong>{selected_profile['expected_days_to_churn']}</strong></div>
                <div class="detail-card"><span>Uplift Score</span><strong>{selected_profile['uplift_score']}</strong></div>
                <div class="detail-card"><span>Recommended Action</span><strong>{html.escape(selected_action['recommended_action'])}</strong></div>
                <div class="detail-card"><span>Estimated Cost</span><strong>${selected_action['estimated_cost']:.2f}</strong></div>
            </div>
            <div class="subpanel">
                <h3>Top Drivers</h3>
                <ul>{driver_lines}</ul>
            </div>
            <div class="subpanel">
                <h3>ROI</h3>
                <div class="detail-grid compact">
                    <div class="detail-card"><span>Revenue at Risk</span><strong>${selected_roi['revenue_at_risk']:.2f}</strong></div>
                    <div class="detail-card"><span>Expected Recovery</span><strong>${selected_roi['expected_recovery']:.2f}</strong></div>
                    <div class="detail-card"><span>Net Expected Recovery</span><strong>${selected_roi['net_expected_recovery']:.2f}</strong></div>
                    <div class="detail-card"><span>ROI</span><strong>{selected_roi['roi'] if selected_roi['roi'] is not None else 'n/a'}</strong></div>
                </div>
                <p>{html.escape(selected_action['reason'])}</p>
            </div>
        </section>
        """
    elif selected_message:
        selected_html = f'<section class="panel section"><p>{html.escape(selected_message)}</p></section>'

    model_table = model_comparison.to_html(index=False, classes="data-table", border=0)
    risk_table = high_risk_rows.to_html(index=False, classes="data-table", border=0)

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>ChurnGuard Dashboard</title>
        <style>
            :root {{
                --bg: #f4f7fb;
                --panel: #ffffff;
                --ink: #14213d;
                --muted: #5c677d;
                --line: #d9e2ec;
                --accent: #0f766e;
                --accent-soft: rgba(15, 118, 110, 0.12);
                --warning: #b45309;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                font-family: Arial, Helvetica, sans-serif;
                background: linear-gradient(180deg, #eef4fb 0%, #f8fbfd 100%);
                color: var(--ink);
            }}
            .wrap {{ max-width: 1280px; margin: 0 auto; padding: 28px 18px 40px; }}
            .hero {{
                background: linear-gradient(135deg, #0f172a 0%, #0f766e 100%);
                color: white;
                border-radius: 24px;
                padding: 28px;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.18);
            }}
            .hero h1 {{ margin: 0 0 8px; font-size: 2rem; }}
            .hero p {{ margin: 0; max-width: 72ch; opacity: 0.95; line-height: 1.5; }}
            .grid {{ display: grid; gap: 16px; }}
            .stats {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 18px; }}
            .card, .panel, .subpanel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; }}
            .card {{ padding: 18px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05); }}
            .card span, .detail-card span {{ display: block; color: var(--muted); font-size: 0.86rem; margin-bottom: 6px; }}
            .card strong {{ font-size: 1.35rem; }}
            .section {{ margin-top: 22px; }}
            .section h2 {{ margin: 0 0 12px; font-size: 1.2rem; }}
            .panel {{ padding: 18px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05); }}
            .detail-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
            .detail-grid.compact {{ margin-top: 12px; }}
            .detail-card {{ background: #f8fbfd; border: 1px solid var(--line); border-radius: 14px; padding: 14px; }}
            .detail-card strong {{ font-size: 1rem; }}
            .subpanel {{ padding: 16px; margin-top: 14px; background: #fbfdff; }}
            .subpanel h3 {{ margin: 0 0 10px; font-size: 1rem; }}
            .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 18px; }}
            .toolbar form {{ display: flex; gap: 10px; flex-wrap: wrap; }}
            input[type=text] {{
                padding: 12px 14px;
                border-radius: 12px;
                border: 1px solid var(--line);
                min-width: 280px;
                font-size: 0.98rem;
            }}
            button, .button-link {{
                display: inline-block;
                padding: 12px 16px;
                border-radius: 12px;
                border: 0;
                background: var(--accent);
                color: white;
                text-decoration: none;
                font-weight: 700;
                cursor: pointer;
            }}
            .button-link.secondary {{ background: #1d4ed8; }}
            .note {{ color: rgba(255,255,255,0.88); margin-top: 10px; font-size: 0.95rem; }}
            .table-wrap {{ overflow-x: auto; }}
            table.data-table {{ width: 100%; border-collapse: collapse; margin: 0; }}
            table.data-table th, table.data-table td {{
                border-bottom: 1px solid var(--line);
                padding: 10px 12px;
                text-align: left;
                vertical-align: top;
                white-space: nowrap;
            }}
            table.data-table th {{ background: #f2f7fb; font-size: 0.9rem; }}
            ul {{ margin: 0; padding-left: 18px; line-height: 1.6; }}
            .footer {{ margin-top: 18px; color: var(--muted); font-size: 0.9rem; }}
            @media (max-width: 720px) {{
                .wrap {{ padding: 14px; }}
                .hero {{ padding: 20px; border-radius: 18px; }}
                .hero h1 {{ font-size: 1.6rem; }}
                input[type=text] {{ min-width: 100%; width: 100%; }}
            }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <section class="hero">
                <h1>ChurnGuard Dashboard</h1>
                <p>Local, structured view of the Telco churn file loaded from disk. The dashboard shows dataset health, model comparison, high-risk customers, and an optional customer drill-down.</p>
                <div class="toolbar">
                    <form method="get" action="/">
                        <input type="text" name="customer_id" placeholder="Enter customerID to inspect" value="{html.escape(selected_customer_id or '')}">
                        <button type="submit">Inspect Customer</button>
                    </form>
                    <a class="button-link secondary" href="/api/summary">JSON Summary</a>
                    <a class="button-link" href="/docs">API Docs</a>
                </div>
                <div class="note">Data source: {html.escape(str(active_data_file.name))}</div>
            </section>

            <section class="grid stats section">
                <div class="card"><span>Total Customers</span><strong>{total_customers:,}</strong></div>
                <div class="card"><span>Churn Rate</span><strong>{churn_rate:.2f}%</strong></div>
                <div class="card"><span>Average Monthly Charges</span><strong>${avg_monthly:.2f}</strong></div>
                <div class="card"><span>Average Tenure</span><strong>{avg_tenure:.1f} months</strong></div>
                <div class="card"><span>Churned Customers</span><strong>{high_risk_count:,}</strong></div>
            </section>

            <section class="section panel">
                <h2>Model Comparison</h2>
                <div class="table-wrap">{model_table}</div>
            </section>

            {selected_html}

            <section class="section panel">
                <h2>High-Risk Customers</h2>
                <div class="table-wrap">{risk_table}</div>
            </section>

            <div class="footer">
                API endpoints: /health, /api/summary, /api/customer/&lt;customer_id&gt;, /score, /recommend, /roi
            </div>
        </div>
    </body>
    </html>
    """


app = FastAPI(title="ChurnGuard AI 2.1", description="Churn scoring, retention suggestions, and a local dashboard")


class ScoreResponse(BaseModel):
    customer_id: str
    churn_probability: float
    expected_days_to_churn: float
    urgency: str
    top_drivers: list
    uplift_score: float


class RecommendResponse(BaseModel):
    customer_id: str
    recommended_action: str
    estimated_cost: float
    reason: str
    urgency: str


class ROIResponse(BaseModel):
    customer_id: str
    revenue_at_risk: float
    expected_recovery: float
    intervention_cost: float
    net_expected_recovery: float
    roi: Optional[float]


@app.on_event("startup")
def warmup_models():
    initialize_churnguard()


@app.get("/health")
def health():
    ensure_initialized()
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def dashboard(customer_id: Optional[str] = None):
    ensure_initialized()
    return HTMLResponse(render_dashboard(customer_id))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_alias(customer_id: Optional[str] = None):
    ensure_initialized()
    return HTMLResponse(render_dashboard(customer_id))


@app.get("/api/summary")
def api_summary():
    ensure_initialized()
    high_risk_rows = get_high_risk_customer_rows(10)
    return {
        "data_file": active_data_file.name,
        "total_customers": int(len(feature_df)),
        "churn_rate_percent": round(float(feature_df[TARGET_COLUMN].mean() * 100), 2),
        "average_monthly_charges": round(float(feature_df["MonthlyCharges"].mean()), 2),
        "average_tenure": round(float(feature_df["tenure"].mean()), 2),
        "model_comparison": model_comparison.to_dict(orient="records"),
        "high_risk_customers": high_risk_rows.to_dict(orient="records"),
    }


@app.get("/api/customer/{customer_id}")
def api_customer(customer_id: str):
    ensure_initialized()
    profile = score_customer_by_id(customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="customer not found")

    action = recommend_action(
        profile["churn_probability"],
        profile["expected_days_to_churn"],
        profile["uplift_score"],
        profile["top_drivers"],
        profile["monthly_charge"],
    )
    roi_result = calculate_roi(profile["monthly_charge"], profile["uplift_score"], action["estimated_cost"])
    return {"profile": profile, "recommendation": action, "roi": roi_result}


@app.post("/score", response_model=ScoreResponse)
def score(customer_id: str):
    ensure_initialized()
    profile = score_customer_by_id(customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="customer not found")

    return {
        "customer_id": profile["customer_id"],
        "churn_probability": profile["churn_probability"],
        "expected_days_to_churn": profile["expected_days_to_churn"],
        "urgency": urgency_bucket(profile["expected_days_to_churn"]),
        "top_drivers": profile["top_drivers"],
        "uplift_score": profile["uplift_score"],
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(customer_id: str):
    ensure_initialized()
    profile = score_customer_by_id(customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="customer not found")

    action = recommend_action(
        profile["churn_probability"],
        profile["expected_days_to_churn"],
        profile["uplift_score"],
        profile["top_drivers"],
        profile["monthly_charge"],
    )
    return {
        "customer_id": profile["customer_id"],
        "recommended_action": action["recommended_action"],
        "estimated_cost": action["estimated_cost"],
        "reason": action["reason"],
        "urgency": action["urgency"],
    }


@app.post("/roi", response_model=ROIResponse)
def roi(customer_id: str):
    ensure_initialized()
    profile = score_customer_by_id(customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="customer not found")

    action = recommend_action(
        profile["churn_probability"],
        profile["expected_days_to_churn"],
        profile["uplift_score"],
        profile["top_drivers"],
        profile["monthly_charge"],
    )
    roi_result = calculate_roi(profile["monthly_charge"], profile["uplift_score"], action["estimated_cost"])
    return {"customer_id": profile["customer_id"], **roi_result}


@app.post("/batch-score")
def batch_score(customer_ids: List[str]):
    ensure_initialized()
    results = []
    for customer_id in customer_ids:
        profile = score_customer_by_id(customer_id)
        if profile:
            results.append(profile)
    return {"count": len(results), "results": results}


def run_full_report(sample_size: int = 8):
    ensure_initialized()
    print("=" * 72)
    print("CHURNGUARD AI 2.1 - MODEL COMPARISON")
    print("=" * 72)
    print(model_comparison.to_string(index=False))

    print("\n" + "=" * 72)
    print("SAMPLE CUSTOMER-LEVEL RETENTION REPORT")
    print("=" * 72)

    sample_customers = feature_df.sample(sample_size, random_state=RANDOM_STATE)

    total_revenue_at_risk = 0.0
    total_expected_recovery = 0.0
    total_cost = 0.0

    for _, cust in sample_customers.iterrows():
        row = feature_df[feature_df[ID_COLUMN] == cust[ID_COLUMN]]
        profile = build_full_customer_profile(row)
        action = recommend_action(
            profile["churn_probability"],
            profile["expected_days_to_churn"],
            profile["uplift_score"],
            profile["top_drivers"],
            profile["monthly_charge"],
        )
        roi_result = calculate_roi(profile["monthly_charge"], profile["uplift_score"], action["estimated_cost"])

        total_revenue_at_risk += roi_result["revenue_at_risk"]
        total_expected_recovery += roi_result["expected_recovery"]
        total_cost += roi_result["intervention_cost"]

        print(f"\nCustomer: {profile['customer_id']}")
        print(f"  Churn probability      : {profile['churn_probability'] * 100:.1f}%")
        print(f"  Expected time to churn : {profile['expected_days_to_churn']} days ({action['urgency']})")
        print(f"  Top drivers            : {', '.join(driver['feature'] for driver in profile['top_drivers'])}")
        print(f"  Uplift score           : {profile['uplift_score']}")
        print(f"  Recommended action     : {action['recommended_action']}  (cost: ${action['estimated_cost']})")
        print(f"  Reason                 : {action['reason']}")
        print(f"  Revenue at risk        : ${roi_result['revenue_at_risk']}")
        print(f"  Expected recovery      : ${roi_result['expected_recovery']}")
        print(f"  Net expected recovery  : ${roi_result['net_expected_recovery']}")
        print(f"  ROI                    : {roi_result['roi']}")

    print("\n" + "=" * 72)
    print("SAMPLE PORTFOLIO SUMMARY (for the customers above)")
    print("=" * 72)
    print(f"  Total revenue at risk   : ${round(total_revenue_at_risk, 2)}")
    print(f"  Total expected recovery : ${round(total_expected_recovery, 2)}")
    print(f"  Total intervention cost : ${round(total_cost, 2)}")
    net_total = round(total_expected_recovery - total_cost, 2)
    print(f"  Net expected recovery   : ${net_total}")
    print("=" * 72)
    print(f"\nDashboard: http://127.0.0.1:{DEFAULT_PORT}")


if __name__ == "__main__":
    import uvicorn

    initialize_churnguard()
    run_full_report()
    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT, reload=False)
