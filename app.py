import os
import io
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Causality / stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression

# DoWhy
from dowhy import CausalModel

# Graph/DAG
import graphviz

# Optional: gCastle for PC algorithm causal discovery
try:
    from castle.algorithms import PC
    from castle.common.priori_knowledge import PrioriKnowledge
    CASTLE_AVAILABLE = True
except Exception:
    CASTLE_AVAILABLE = False

# --------------------
# Streamlit page
# --------------------
st.set_page_config(page_title="Causal Impact Analysis on Employment Programs", layout="wide")
st.title("🧪 Causal Impact Analysis on Employment Programs")
st.caption("Estimate the causal effect of a program (treatment) on income outcomes with DoWhy + optional DAG discovery")

# --------------------
# Helpers
# --------------------
def show_fig(fig, width=720):
    """Render any matplotlib fig at a fixed width so it always fits without scroll."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    st.image(buf.getvalue(), width=width)
    plt.close(fig)

# Global compact toggle for slightly smaller figures
compact = st.toggle("Compact mode (smaller charts)", value=True)

# --------------------
# Data loading
# --------------------
@st.cache_data
def load_data(default_path: str = "data.csv") -> pd.DataFrame:
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return pd.DataFrame()

uploaded_file = st.file_uploader("Upload your data.csv (optional if present alongside app.py)", type=["csv"])
data = pd.read_csv(uploaded_file) if uploaded_file is not None else load_data()

if data.empty:
    st.warning("No data loaded. Place **data.csv** next to app.py or upload it above.")
    st.stop()

st.success(f"Loaded dataset with shape: {data.shape}")
st.dataframe(data.head(), height=220, use_container_width=True)

with st.expander("Dataset info & summary", expanded=False):
    buf = io.StringIO()
    data.info(buf=buf)
    st.code(buf.getvalue(), language="text")
    st.dataframe(data.describe(include="all").T, use_container_width=True)

# --------------------
# EDA: Correlation matrix (fixed-width rendering)
# --------------------
st.subheader("📈 Correlation matrix")

corr = data.corr(numeric_only=True)
n = len(corr.columns)

# balanced size + fonts; we'll still render via show_fig() so it fits the page width
if n <= 8:
    w, h, fs = 5.6, 4.4, 9
elif n <= 12:
    w, h, fs = 6.2, 5.0, 8.5
else:
    w, h, fs = 6.8, 5.6, 8

fig_corr, ax = plt.subplots(figsize=(w, h))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(corr.columns, rotation=35, ha='right', fontsize=fs)
ax.set_yticklabels(corr.columns, fontsize=fs)
cbar = fig_corr.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=fs-1)

# numeric annotations (keeps readability on smaller figs)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=fs-2)

fig_corr.tight_layout()
show_fig(fig_corr, width=720)  # 👈 always fits; change 720 if you want narrower/wider

# --------------------
# DAG: Prior knowledge + PC discovery (optional)
# --------------------
st.subheader("🕸️ Causal Graph (DAG)")

cols = list(data.columns)
default_treatment = "treat" if "treat" in cols else st.selectbox("Select treatment column", cols, index=0)
default_outcome = "re22" if "re22" in cols else st.selectbox("Select outcome column", cols, index=min(1, len(cols)-1))
treat_col, outcome_col = default_treatment, default_outcome

st.markdown(f"**Treatment:** `{treat_col}`  |  **Outcome:** `{outcome_col}`")

required_edges_default = [("educ", outcome_col), ("age", outcome_col), ("nodegr", outcome_col)]
forbidden_edges_default = [
    ("treat", "re19"), ("treat", "re20"),
    ("black", "hisp"), ("black", "age"), ("age", "black"),
    ("hisp", "black"), ("educ", "black"), ("educ", "hisp"),
    (outcome_col, "re19"), (outcome_col, "re20"), ("re20", "re19"),
    ("treat", "nodegr")
]

with st.expander("Edit prior-knowledge edges (optional)"):
    st.caption("Add/remove edges as comma-separated pairs 'src->dst'. Leave blank to use defaults.")
    req_text = st.text_area("Required edges", value="\n".join([f"{a}->{b}" for a,b in required_edges_default]), height=80)
    forb_text = st.text_area("Forbidden edges", value="\n".join([f"{a}->{b}" for a,b in forbidden_edges_default]), height=120)

def parse_edges(text):
    edges = []
    for line in text.strip().splitlines():
        if "->" in line:
            a,b = [s.strip() for s in line.split("->", 1)]
            if a and b:
                edges.append((a,b))
    return edges

required_edges = parse_edges(req_text) if req_text.strip() else required_edges_default
forbidden_edges = parse_edges(forb_text) if forb_text.strip() else forbidden_edges_default

def discover_dag(df: pd.DataFrame, required_edges, forbidden_edges):
    dot = graphviz.Digraph()
    for c in df.columns:
        dot.node(c)
    if CASTLE_AVAILABLE:
        mapping = {i: c for i, c in enumerate(df.columns)}
        inv = {v: k for k, v in mapping.items()}
        try:
            knowledge = PrioriKnowledge(len(df.columns))
            req_idx = [(inv[a], inv[b]) for a,b in required_edges if a in inv and b in inv]
            forb_idx = [(inv[a], inv[b]) for a,b in forbidden_edges if a in inv and b in inv]
            if req_idx:
                knowledge.add_required_edges(req_idx)
            if forb_idx:
                knowledge.add_forbidden_edges(forb_idx)
            pc = PC(priori_knowledge=knowledge)
            pc.learn(df)
            mat = pc.causal_matrix
            import networkx as nx
            g = nx.DiGraph(mat)
            g = nx.relabel_nodes(g, mapping, copy=True)
            edges = list(g.edges())
        except Exception as e:
            st.info(f"PC discovery failed ({e}); showing required edges only.")
            edges = required_edges
    else:
        edges = required_edges

    for a,b in edges:
        if a in df.columns and b in df.columns:
            dot.edge(a,b)
    return dot

dag = discover_dag(data, required_edges, forbidden_edges)
st.graphviz_chart(dag, width="stretch")

# --------------------
# Confounders selection
# --------------------
st.subheader("🎛️ Controls / Confounders")
candidate_controls = [c for c in data.columns if c not in [treat_col, outcome_col]]
controls = st.multiselect(
    "Select control variables",
    candidate_controls,
    default=[c for c in candidate_controls if c in ["age","educ","nodegr","black","hisp","married"]],
)
st.caption(f"Using controls: {controls}")

# --------------------
# OLS as sanity check
# --------------------
st.subheader("📐 OLS sanity check")
X = data[[treat_col] + controls].copy()
X = sm.add_constant(X)
y = data[outcome_col]
model_ols = sm.OLS(y, X).fit()
st.code(model_ols.summary().as_text(), language="text")

# --------------------
# DoWhy causal model
# --------------------
st.subheader("🧭 DoWhy: ATE Estimation")

def dag_to_dot(edges):
    lines = ["digraph {"]
    for a,b in edges:
        lines.append(f'  "{a}" -> "{b}";')
    lines.append("}")
    return "\n".join(lines)

# controls -> outcome; treatment -> outcome
edges_for_dot = list(set(required_edges))
for c in controls:
    if (c, outcome_col) not in edges_for_dot:
        edges_for_dot.append((c, outcome_col))
if (treat_col, outcome_col) not in edges_for_dot:
    edges_for_dot.append((treat_col, outcome_col))

dot_str = dag_to_dot(edges_for_dot)

# Robust model creation: fall back to common_causes if DOT parsing fails
try:
    model = CausalModel(data=data, treatment=treat_col, outcome=outcome_col, graph=dot_str)
except Exception as e:
    st.warning(f"Graph parsing/Graphviz issue detected; falling back to common_causes. Details: {e}")
    model = CausalModel(data=data, treatment=treat_col, outcome=outcome_col, common_causes=controls or None)

# Identify effect
try:
    estimand = model.identify_effect()
except Exception as e:
    st.error(f"Could not identify effect automatically. Try adding/removing controls. Details: {e}")
    st.stop()

def estimate_method(name, method_name, **kwargs):
    try:
        est = model.estimate_effect(
            identified_estimand=estimand,
            method_name=method_name,
            target_units="ate",
            method_params=kwargs.get("method_params", {})
        )
        return float(est.value)
    except Exception as e:
        st.warning(f"{name} failed: {e}")
        return np.nan

results = {}

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Run Distance Matching"):
        results["Distance Matching"] = estimate_method("Distance Matching", "backdoor.distance_matching")
with col2:
    if st.button("Run Propensity (Stratification/Matching/Weighting)"):
        results["Propensity Stratification"] = estimate_method("PS Stratification", "backdoor.propensity_score_stratification")
        results["Propensity Matching"] = estimate_method("PS Matching", "backdoor.propensity_score_matching")
        results["Propensity Weighting"] = estimate_method("PS Weighting", "backdoor.propensity_score_weighting")
with col3:
    if st.button("Run DR Learner & Causal Forest"):
        results["Doubly Robust (LinearDRLearner)"] = estimate_method(
            "DR Learner",
            "backdoor.econml.dr.LinearDRLearner",
            method_params={"init_params": {"model_propensity": LogisticRegression(),
                                           "model_regression": LinearRegression()},
                           "fit_params": {}}
        )
        results["CausalForestDML"] = estimate_method(
            "CausalForestDML",
            "backdoor.econml.dml.CausalForestDML",
            method_params={"init_params": {"model_y": LinearRegression(),
                                           "model_t": LogisticRegression(),
                                           "discrete_treatment": True,
                                           "cv": 5},
                           "fit_params": {}}
        )

if results:
    st.subheader("📊 ATE Results (by method)")
    res_df = pd.DataFrame({"Method": list(results.keys()), "ATE": list(results.values())})
    st.dataframe(res_df, height=220, use_container_width=True)

    # fixed-width bar chart
    fig_bar, axb = plt.subplots(figsize=(5.6 if compact else 6.2, 3.0))
    axb.bar(res_df["Method"], res_df["ATE"])
    axb.set_ylabel("ATE")
    axb.set_title("Estimated Average Treatment Effect by Method")
    axb.tick_params(axis='x', rotation=45, labelsize=9 if compact else 10)
    fig_bar.tight_layout()
    show_fig(fig_bar, width=720)

    # Download button
    csv = res_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download ATE Results as CSV", data=csv, file_name="ATE_results.csv", mime="text/csv")

st.info("Tip: Adjust **controls** and **prior edges** to see robustness. Use OLS as a sanity check; ATE should be directionally consistent across methods.")
