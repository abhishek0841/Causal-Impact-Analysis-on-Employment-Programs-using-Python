
# Causal Impact Dashboard (Streamlit)

Interactive dashboard to estimate the causal effect of a treatment (e.g., employment program) on an outcome (e.g., income) using DoWhy. 
Includes optional DAG discovery (PC algorithm via gCastle), OLS sanity check, and multiple ATE estimators (distance matching, propensity methods, DR Learner, CausalForestDML).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `data.csv` in the same folder or upload via the app.

## Deploy (Streamlit Cloud)

1. Push `app.py`, `requirements.txt`, and `data.csv` (optional) to your GitHub repo.
2. Create a new Streamlit app and point it to `app.py`.
3. If `castle` fails to install, the app still runs—PC discovery is optional.

## Notes
- Edit prior-knowledge edges in the app to shape the DAG.
- Select controls (confounders) interactively.
- Run multiple estimators and compare ATE in the bar chart.
