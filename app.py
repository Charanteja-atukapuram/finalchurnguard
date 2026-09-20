from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

import churnguard


st.set_page_config(
    page_title="Churn Guard Analysis",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def load_runtime(data_file: str) -> str:
    churnguard.initialize_churnguard(data_path=data_file, force=False)
    return data_file


def render_portfolio_charts(feature_df: pd.DataFrame) -> None:
    chart_cols = st.columns(3)

    churn_distribution = (
        feature_df[churnguard.TARGET_COLUMN]
        .map({0: "Stayed", 1: "Churned"})
        .value_counts()
        .rename_axis("status")
        .reset_index(name="customers")
    )
    with chart_cols[0]:
        st.write("**Customer churn distribution**")
        st.bar_chart(churn_distribution, x="status", y="customers")

    contract_churn = (
        feature_df.groupby("Contract", as_index=False)[churnguard.TARGET_COLUMN]
        .mean()
        .assign(churn_rate=lambda frame: frame[churnguard.TARGET_COLUMN] * 100)
        .drop(columns=churnguard.TARGET_COLUMN)
        .sort_values("churn_rate", ascending=False)
    )
    with chart_cols[1]:
        st.write("**Churn rate by contract**")
        st.line_chart(contract_churn, x="Contract", y="churn_rate")

    tenure_bins = pd.cut(
        feature_df["tenure"],
        bins=[-1, 6, 12, 24, 48, float("inf")],
        labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"],
    )
    tenure_churn = (
        feature_df.assign(tenure_band=tenure_bins)
        .groupby("tenure_band", observed=False, as_index=False)[churnguard.TARGET_COLUMN]
        .mean()
        .assign(churn_rate=lambda frame: frame[churnguard.TARGET_COLUMN] * 100)
        .drop(columns=churnguard.TARGET_COLUMN)
    )
    with chart_cols[2]:
        st.write("**Churn rate by tenure**")
        st.area_chart(tenure_churn, x="tenure_band", y="churn_rate")


def render_customer_charts(profile: dict, customer_row: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    chart_cols = st.columns(2)

    risk_chart = pd.DataFrame(
        {
            "indicator": ["Churn probability", "Uplift score"],
            "score": [profile["churn_probability"] * 100, max(profile["uplift_score"], 0) * 100],
        }
    )
    with chart_cols[0]:
        st.write("**Customer risk indicators (%)**")
        st.bar_chart(risk_chart, x="indicator", y="score")

    customer_comparison = pd.DataFrame(
        {
            "measure": ["Monthly charges", "Tenure (months)"],
            "customer": [
                float(customer_row["MonthlyCharges"].iloc[0]),
                float(customer_row["tenure"].iloc[0]),
            ],
            "portfolio average": [
                float(feature_df["MonthlyCharges"].mean()),
                float(feature_df["tenure"].mean()),
            ],
        }
    )
    with chart_cols[1]:
        st.write("**Customer vs portfolio average**")
        st.bar_chart(customer_comparison, x="measure", y=["customer", "portfolio average"])


def render_customer_section(customer_id: str) -> None:
    profile = churnguard.score_customer_by_id(customer_id)
    if profile is None:
        st.warning(f"No customer found for id: {customer_id}")
        return

    action = churnguard.recommend_action(
        profile["churn_probability"],
        profile["expected_days_to_churn"],
        profile["uplift_score"],
        profile["top_drivers"],
        profile["monthly_charge"],
    )
    roi_result = churnguard.calculate_roi(
        profile["monthly_charge"],
        profile["uplift_score"],
        action["estimated_cost"],
    )
    customer_row = churnguard.feature_df[
        churnguard.feature_df[churnguard.ID_COLUMN] == customer_id
    ]

    st.subheader("Customer Detail")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Churn Probability", f"{profile['churn_probability'] * 100:.1f}%")
    metric_cols[1].metric("Expected Days to Churn", f"{profile['expected_days_to_churn']}")
    metric_cols[2].metric("Uplift Score", f"{profile['uplift_score']}")
    metric_cols[3].metric("Urgency", action["urgency"])

    render_customer_charts(profile, customer_row, churnguard.feature_df)

    st.write("**Recommended Action**")
    st.info(
        f"{action['recommended_action']} | Estimated cost: ${action['estimated_cost']:.2f}\n\n"
        f"Reason: {action['reason']}"
    )

    st.write("**Top Drivers**")
    st.table(profile["top_drivers"])

    st.write("**ROI Estimate**")
    roi_cols = st.columns(4)
    roi_cols[0].metric("Revenue at Risk", f"${roi_result['revenue_at_risk']:.2f}")
    roi_cols[1].metric("Expected Recovery", f"${roi_result['expected_recovery']:.2f}")
    roi_cols[2].metric("Net Expected Recovery", f"${roi_result['net_expected_recovery']:.2f}")
    roi_cols[3].metric("ROI", "n/a" if roi_result["roi"] is None else f"{roi_result['roi']}")


def resolve_customer_id(typed_customer_id: str, selected_customer_id: str) -> Optional[str]:
    if typed_customer_id.strip():
        return typed_customer_id.strip()
    if selected_customer_id != "None":
        return selected_customer_id
    return None


def main() -> None:
    st.title("Churn Guard Analysis")
    st.write(
        "Predictive churn scoring and retention recommendation dashboard built from your existing project logic."
    )

    data_path = Path(churnguard.DATA_FILE)
    try:
        load_runtime(str(data_path))
    except FileNotFoundError:
        st.error(
            f"Dataset file is missing: {data_path.name}. Add it to the project root and rerun."
        )
        return
    except ValueError as exc:
        st.error(f"Data validation failed: {exc}")
        return
    except Exception:
        st.error("Could not initialize the churn models. Please check requirements and dataset format.")
        return

    if churnguard.feature_df is None:
        st.error("Model runtime is unavailable.")
        return

    st.sidebar.header("Controls")
    top_n = st.sidebar.slider("High-risk customers to display", min_value=5, max_value=30, value=12)

    high_risk_rows = churnguard.get_high_risk_customer_rows(top_n)
    customer_options = ["None"] + high_risk_rows[churnguard.ID_COLUMN].astype(str).tolist()
    selected_customer_id = st.sidebar.selectbox("Pick a high-risk customer", customer_options, index=0)
    typed_customer_id = st.sidebar.text_input("Or enter customerID manually", "")

    selected_id = resolve_customer_id(typed_customer_id, selected_customer_id)
    if st.sidebar.button("Analyze Customer"):
        if selected_id is None:
            st.warning("Please select or enter a customerID.")
        else:
            render_customer_section(selected_id)

    st.header("Portfolio Summary")
    feature_df = churnguard.feature_df
    summary_cols = st.columns(5)
    summary_cols[0].metric("Total Customers", f"{len(feature_df):,}")
    summary_cols[1].metric("Churn Rate", f"{feature_df[churnguard.TARGET_COLUMN].mean() * 100:.2f}%")
    summary_cols[2].metric("Avg Monthly Charges", f"${feature_df['MonthlyCharges'].mean():.2f}")
    summary_cols[3].metric("Avg Tenure", f"{feature_df['tenure'].mean():.1f} months")
    summary_cols[4].metric("Churned Customers", f"{int((feature_df[churnguard.TARGET_COLUMN] == 1).sum()):,}")

    st.subheader("Portfolio Churn Charts")
    render_portfolio_charts(feature_df)

    st.subheader("Model Comparison")
    st.dataframe(churnguard.model_comparison, width="stretch")

    st.subheader("High-Risk Customers")
    st.dataframe(high_risk_rows, width="stretch")
    st.download_button(
        "Download high-risk table (CSV)",
        data=high_risk_rows.to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
