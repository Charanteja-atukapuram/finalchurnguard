# ChurnGuard AI 2.1

ChurnGuard predicts customer churn risk, estimates time-to-churn, and suggests retention actions with ROI estimates using the Telco churn dataset.

## Features
- Data cleaning and preprocessing for Telco churn records
- Classification model comparison (Logistic Regression vs Random Forest)
- Survival analysis for expected time-to-churn
- Retention recommendation and ROI estimation
- Streamlit dashboard for interactive analysis

## Tech Stack
- Python
- Streamlit
- pandas, numpy
- scikit-learn
- lifelines
- FastAPI (existing API endpoints preserved)

## Run Locally
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Streamlit app:

```bash
streamlit run app.py
```

3. (Optional) Start existing FastAPI server:

```bash
python churnguard.py
```

## Streamlit Community Cloud Deployment
1. Push this project to GitHub (including `app.py`, `requirements.txt`, `runtime.txt`, and dataset file).
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set **Main file path** to:

```text
app.py
```

4. Deploy.
