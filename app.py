import os
import tempfile
import zipfile
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.profiler import load_dataset, get_sqlite_tables, profile_dataset, recommend_kpis_and_targets
from src.ml_engine import (
    preprocess_and_split, train_baselines, get_best_model,
    extract_feature_importances, generate_evaluation_plots,
    train_clustering, get_cluster_profiles, generate_clustering_diagnostic_plots
)
from src.validator import run_validation_checks
from src.reporter import generate_markdown_report, generate_reproducible_code, create_static_plots, generate_pdf_report

st.set_page_config(page_title="Data2Business Agent", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: #F0F4FF; color: #1E293B; }
    html,body,[class*="css"] { font-family:"Inter",-apple-system,BlinkMacSystemFont,sans-serif; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#1E1B4B 0%,#312E81 100%) !important;
        border-right: 1px solid #4338CA;
    }
    section[data-testid="stSidebar"] * { color:#E0E7FF !important; }
    .card {
        background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px;
        padding:1.5rem 1.75rem; margin-bottom:1.2rem;
        box-shadow:0 2px 12px rgba(99,102,241,.08);
        transition:box-shadow .2s ease,transform .2s ease;
    }
    .card:hover { box-shadow:0 6px 24px rgba(99,102,241,.15); transform:translateY(-2px); }
    h1,h2,h3,h4,h5,h6 { color:#1E293B !important; font-weight:700; }
    .main-title {
        background:linear-gradient(90deg,#4F46E5 0%,#7C3AED 50%,#DB2777 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        font-size:2.4rem; font-weight:800; line-height:1.2; margin-bottom:.2rem;
    }
    .subtitle { color:#475569; font-size:1.05rem; margin-top:0; }
    .stButton>button {
        background:linear-gradient(90deg,#4F46E5 0%,#7C3AED 100%) !important;
        color:#FFFFFF !important; border:none !important; border-radius:8px !important;
        padding:.55rem 1.6rem !important; font-weight:600 !important;
        box-shadow:0 3px 10px rgba(79,70,229,.35) !important; transition:all .2s ease !important;
    }
    .stButton>button:hover { transform:scale(1.03); box-shadow:0 5px 18px rgba(79,70,229,.5) !important; }
    .stDownloadButton>button {
        background:linear-gradient(90deg,#059669 0%,#0284C7 100%) !important;
        color:#FFFFFF !important; border:none !important; border-radius:8px !important;
        padding:.55rem 1.4rem !important; font-weight:600 !important;
    }
    div[data-testid="metric-container"] {
        background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px;
        padding:1rem 1.2rem; box-shadow:0 2px 8px rgba(0,0,0,.06);
    }
    div[data-testid="metric-container"] label {
        color:#64748B !important; font-size:.82rem !important;
        font-weight:600 !important; text-transform:uppercase; letter-spacing:.04em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color:#1E293B !important; font-size:1.6rem !important; font-weight:700 !important;
    }
    .status-badge { padding:3px 9px; border-radius:5px; font-size:.78rem; font-weight:700; }
    .status-critical { background:#FEE2E2; color:#B91C1C; border:1px solid #FECACA; }
    .status-warning  { background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; }
    .status-info     { background:#D1FAE5; color:#065F46; border:1px solid #A7F3D0; }
    .log-card {
        background:#FFFFFF; border-radius:10px; padding:1rem 1.2rem;
        margin-bottom:.8rem; border:1px solid #E2E8F0; box-shadow:0 1px 4px rgba(0,0,0,.05);
    }
    .log-title { font-weight:700; font-size:1rem; color:#1E293B; }
    .log-msg   { font-weight:600; font-size:.95rem; color:#334155; margin-top:.3rem; }
    .log-det   { font-size:.88rem; color:#64748B; margin-bottom:0; }
    button[data-baseweb="tab"] { font-weight:600 !important; color:#475569 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color:#4F46E5 !important; border-bottom:3px solid #4F46E5 !important; }
    .stDataFrame thead th { background:#EEF2FF !important; color:#3730A3 !important; font-weight:700 !important; }
    details>summary { font-weight:600; color:#4F46E5; }
    .stCodeBlock pre { background:#1E1B4B !important; color:#E0E7FF !important; border-radius:10px !important; }
    .stAlert { border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

_DEFAULTS = {
    "uploaded_file": None, "df": None, "file_name": "",
    "sqlite_tables": [], "selected_table": None,
    "profile": None, "kpi_recs": None,
    "target_col": None, "feature_cols": [],
    "task_type": "Classification", "n_clusters": 3,
    "ml_results": None, "best_model_name": None, "best_model_info": None,
    "cluster_results": None, "best_cluster_name": None, "cluster_labels": None,
    "validation_logs": None, "markdown_report": None,
    "reproducible_code": None, "pdf_report": None,
    "business_question": "Predict customer churn based on transaction history and account profiles",
    "target_audience": "manager",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_state():
    skip = {"uploaded_file", "file_name", "sqlite_tables", "selected_table"}
    for k, v in _DEFAULTS.items():
        if k not in skip:
            st.session_state[k] = v


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='font-size:1.7rem;margin-bottom:0;color:#FFFFFF'>🤖 Data2Business</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#A5B4FC;font-size:.88rem;margin-top:0;'>Business Analytics & ML Assistant</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Workflow Progress")
    steps = ["1. Upload & Profile", "2. Goal & Targets", "3. Visual Insights",
             "4. Baseline ML", "5. Trust Validation", "6. Report Export"]
    curr = 0
    if st.session_state.df is not None: curr = 1
    if st.session_state.target_col is not None or st.session_state.task_type == "Clustering": curr = 2
    if st.session_state.ml_results is not None or st.session_state.cluster_results is not None: curr = 5
    for i, s in enumerate(steps):
        if i < curr:   st.markdown(f"🟢 **{s}**")
        elif i == curr: st.markdown(f"🔵 **{s}** *(active)*")
        else:           st.markdown(f"⚪ {s}")
    st.markdown("---")
    if st.session_state.df is not None:
        st.markdown("### 📈 Active Dataset")
        st.markdown(f"- **File:** `{st.session_state.file_name}`")
        if st.session_state.selected_table:
            st.markdown(f"- **Table:** `{st.session_state.selected_table}`")
        st.markdown(f"- **Rows:** {st.session_state.df.shape[0]:,}")
        st.markdown(f"- **Cols:** {st.session_state.df.shape[1]}")
        tlbl = {"Classification": "🏷️ Classification", "Regression": "📉 Regression", "Clustering": "🔵 Clustering"}
        st.markdown(f"- **Task:** {tlbl.get(st.session_state.task_type, st.session_state.task_type)}")
        if st.button("🗑️ Clear / Reset", key="clear_btn"):
            reset_state(); st.session_state.uploaded_file = None; st.rerun()

# ── Header ───────────────────────────────────────────────────
st.markdown("<h1 class='main-title'>Data2Business Agent</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Transform raw data into trustworthy insights, KPIs, and baseline models.</p>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STEP 1 – UPLOAD
# ════════════════════════════════════════════════════════════
if st.session_state.df is None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📁 Step 1: Upload your dataset")
    st.markdown("Supported: **CSV · Excel · JSON · Parquet · SQLite**")
    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a dataset file",
            type=["csv", "xlsx", "xls", "json", "parquet", "db", "sqlite", "sqlite3"]
        )
    with col_sample:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if os.path.exists("iris.csv"):
            if st.button("📂 Load Sample Iris", use_container_width=True):
                with st.spinner("Loading sample dataset..."):
                    st.session_state.df = pd.read_csv("iris.csv")
                    st.session_state.file_name = "iris.csv"
                    st.session_state.selected_table = None
                    st.rerun()
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.file_name = uploaded_file.name
        _, ext = os.path.splitext(uploaded_file.name.lower())
        if ext in [".db", ".sqlite", ".sqlite3"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_file.getbuffer()); tmp_path = tmp.name
            tables = get_sqlite_tables(tmp_path)
            st.session_state.sqlite_tables = tables
            sel_tbl = st.selectbox("Select table", tables)
            if st.button("Load Table"):
                with st.spinner("Loading…"):
                    st.session_state.df = load_dataset(tmp_path, file_type="sqlite", table_name=sel_tbl)
                    st.session_state.selected_table = sel_tbl
                    try: os.unlink(tmp_path)
                    except: pass
                    st.rerun()
        else:
            with st.spinner("Parsing…"):
                try:
                    st.session_state.df = load_dataset(uploaded_file)
                    st.session_state.selected_table = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top:2rem;background:#FFFFFF;border-radius:14px;padding:1.5rem;
                border:1px solid #E2E8F0;box-shadow:0 2px 12px rgba(99,102,241,.08);'>
        <h4 style='color:#4F46E5;margin-top:0;'>💡 How it works</h4>
        <ol style='color:#475569;line-height:2;'>
            <li><b>Upload Data</b> – Profile columns, missing values, data health.</li>
            <li><b>Set Goal</b> – Supervised prediction <i>or</i> unsupervised clustering.</li>
            <li><b>Explore Charts</b> – Auto-generated distributions and correlations.</li>
            <li><b>Train Models</b> – Compare three baselines; pick the best performer.</li>
            <li><b>Validate</b> – Check leakage, imbalance, and cluster quality.</li>
            <li><b>Export</b> – PDF/Markdown report, reproducible code, ZIP bundle.</li>
        </ol>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MAIN FLOW
# ════════════════════════════════════════════════════════════
else:
    if st.session_state.profile is None:
        with st.spinner("Profiling dataset…"):
            st.session_state.profile = profile_dataset(st.session_state.df)
            st.session_state.kpi_recs = recommend_kpis_and_targets(st.session_state.profile)

    tab_profile, tab_goal, tab_viz, tab_ml, tab_validation, tab_export = st.tabs([
        "📊 Data Profile", "🎯 Business Goal & Target", "📈 Visual Insights",
        "🤖 Baseline ML / Clustering", "🛡️ Trust Validation", "📄 Report Export"
    ])

    # ── TAB 1: Profile ────────────────────────────────────
    with tab_profile:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{st.session_state.profile['num_rows']:,}")
        c2.metric("Columns", st.session_state.profile['num_cols'])
        c3.metric("Duplicate Rows", st.session_state.profile['num_duplicates'])
        total_c = st.session_state.df.size
        null_c  = st.session_state.df.isnull().sum().sum()
        c4.metric("Completeness", f"{((total_c - null_c) / total_c * 100) if total_c > 0 else 0:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        st.subheader("Column Profiles")
        col_list = []
        for ci in st.session_state.profile['columns']:
            ss = ""
            if ci['semantic_type'] == 'Numeric' and 'stats' in ci:
                s = ci['stats']
                ss = f"Mean:{s['mean']:.2f} | [{s['min']:.2f},{s['max']:.2f}]"
            elif ci['semantic_type'] == 'Categorical' and 'stats' in ci and 'top_values' in ci['stats']:
                tv = ci['stats']['top_values']
                ss = f"Top:'{tv[0]['value']}' ({tv[0]['count']} rows)" if tv else ""
            elif ci['semantic_type'] == 'Datetime' and 'stats' in ci and 'min' in ci['stats']:
                ss = f"Range:{ci['stats']['min']} → {ci['stats']['max']}"
            col_list.append({
                "Column": ci['column_name'], "Type": ci['data_type'],
                "Semantic": ci['semantic_type'], "Missing%": f"{ci['missing_percentage']}%",
                "Unique": ci['unique_count'], "Stats": ss
            })
        st.table(pd.DataFrame(col_list))

    # ── TAB 2: Goal ───────────────────────────────────────
    with tab_goal:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Define Business Context")
        bq = st.text_input("Business question", value=st.session_state.business_question)
        st.session_state.business_question = bq
        ta_opts = ["technical", "manager", "executive"]
        ta = st.selectbox("Target audience", ta_opts,
                          index=ta_opts.index(st.session_state.target_audience),
                          format_func=str.capitalize)
        st.session_state.target_audience = ta
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.kpi_recs:
            recs = st.session_state.kpi_recs['recommended_targets']
            if recs:
                st.subheader("💡 Recommended Targets")
                st.table(pd.DataFrame([{
                    "Column": r['column'], "Type": r['type'],
                    "Task": r['task_suggestion'],
                    "Confidence": "⭐⭐⭐ High" if r['score'] >= 5 else "⭐⭐ Medium"
                } for r in recs]))

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Configure Pipeline")

        task_type = st.radio(
            "Task Type",
            ["Classification", "Regression", "Clustering"],
            index=["Classification", "Regression", "Clustering"].index(st.session_state.task_type),
            help="**Classification** – predict a category.  **Regression** – predict a number.  "
                 "**Clustering** – discover natural groups (no target needed).",
            horizontal=True
        )

        all_cols = [c['column_name'] for c in st.session_state.profile['columns']]

        if task_type == "Clustering":
            st.info("🔵 **Clustering mode** – no target column needed. Select features and cluster count.")
            def_feat = [c['column_name'] for c in st.session_state.profile['columns']
                        if c['semantic_type'] not in ['ID', 'Text'] and st.session_state.df[c['column_name']].notnull().any()]
            feature_cols = st.multiselect(
                "Feature Columns (for grouping)", all_cols,
                default=st.session_state.feature_cols if st.session_state.feature_cols else def_feat
            )
            n_clusters = st.slider("Number of Clusters (K)", 2, 12, st.session_state.n_clusters)
            if st.button("💾 Save & Apply", key="save_c"):
                valid_features = [col for col in feature_cols if st.session_state.df[col].notnull().any()]
                if not valid_features:
                    st.error("❌ Please select at least one feature column that contains valid (non-missing) data.")
                else:
                    ignored = list(set(feature_cols) - set(valid_features))
                    st.session_state.task_type = "Clustering"
                    st.session_state.target_col = None
                    st.session_state.feature_cols = valid_features
                    st.session_state.n_clusters = n_clusters
                    for k in ["ml_results", "cluster_results", "cluster_labels", "best_cluster_name",
                               "validation_logs", "markdown_report", "pdf_report"]:
                        st.session_state[k] = None
                    st.success(f"Clustering: K={n_clusters}, {len(valid_features)} features.")
                    if ignored:
                        st.warning(f"⚠️ Ignored {len(ignored)} completely empty feature(s): {ignored}")
                    st.rerun()
        else:
            def_idx = 0
            if st.session_state.target_col is None and st.session_state.kpi_recs and st.session_state.kpi_recs['recommended_targets']:
                rc = st.session_state.kpi_recs['recommended_targets'][0]['column']
                if rc in all_cols: def_idx = all_cols.index(rc)
            elif st.session_state.target_col in all_cols:
                def_idx = all_cols.index(st.session_state.target_col)
            target_col = st.selectbox("Target Variable", all_cols, index=def_idx)
            def_feat = [c['column_name'] for c in st.session_state.profile['columns']
                        if c['column_name'] != target_col and c['semantic_type'] not in ['ID', 'Text'] and st.session_state.df[c['column_name']].notnull().any()]
            feature_cols = st.multiselect(
                "Feature Columns (predictors)", all_cols,
                default=st.session_state.feature_cols if st.session_state.feature_cols else def_feat
            )
            if st.button("💾 Save & Apply", key="save_s"):
                valid_features = [col for col in feature_cols if st.session_state.df[col].notnull().any()]
                if not valid_features:
                    st.error("❌ Please select at least one feature column that contains valid (non-missing) data.")
                else:
                    ignored = list(set(feature_cols) - set(valid_features))
                    st.session_state.target_col = target_col
                    st.session_state.task_type = task_type
                    st.session_state.feature_cols = valid_features
                    for k in ["ml_results", "best_model_name", "best_model_info", "cluster_results",
                               "cluster_labels", "validation_logs", "markdown_report", "pdf_report"]:
                        st.session_state[k] = None
                    st.success(f"Predicting `{target_col}` ({task_type}) with {len(valid_features)} features.")
                    if ignored:
                        st.warning(f"⚠️ Ignored {len(ignored)} completely empty feature(s): {ignored}")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: Viz ────────────────────────────────────────
    with tab_viz:
        ready = (
            (st.session_state.task_type == "Clustering" and bool(st.session_state.feature_cols)) or
            (st.session_state.task_type != "Clustering" and st.session_state.target_col is not None)
        )
        if not ready:
            st.warning("Configure settings in the **Business Goal & Target** tab first.")
        else:
            num_cols_all = [c['column_name'] for c in st.session_state.profile['columns'] if c['semantic_type'] == 'Numeric']

            if st.session_state.task_type == "Clustering":
                st.subheader("📈 Feature Distributions for Clustering")
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("1. Feature Distribution")
                num_f = [f for f in st.session_state.feature_cols if f in num_cols_all]
                if num_f:
                    sf = st.selectbox("Feature", num_f, key="cvf")
                    fig = px.histogram(st.session_state.df, x=sf, nbins=40,
                                       title=f"Distribution of {sf}", color_discrete_sequence=["#6366F1"])
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1E293B"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numeric features selected.")
                st.markdown("</div>", unsafe_allow_html=True)
                if len(num_f) >= 2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("2. Correlation Heatmap")
                    corr = st.session_state.df[num_f].corr()
                    fig2 = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f")
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1E293B"))
                    st.plotly_chart(fig2, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                if st.session_state.cluster_results is not None:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("3. Cluster Projection (PCA 2D)")
                    br = st.session_state.cluster_results[st.session_state.best_cluster_name]
                    dp = generate_clustering_diagnostic_plots(
                        st.session_state.best_cluster_name, br['X_preprocessed'],
                        st.session_state.cluster_labels, st.session_state.n_clusters)
                    for _, pfig in dp.items():
                        st.plotly_chart(pfig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                tc = st.session_state.target_col
                st.subheader(f"📈 Visual Insights: Predicting `{tc}`")
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader(f"1. Target Distribution: `{tc}`")
                if st.session_state.task_type == "Classification":
                    fig = px.histogram(st.session_state.df, x=tc, color=tc,
                                       color_discrete_sequence=px.colors.qualitative.Plotly)
                else:
                    fig = px.histogram(st.session_state.df, x=tc, nbins=40, color_discrete_sequence=["#8B5CF6"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1E293B"))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                if len(num_cols_all) > 1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("2. Correlation Heatmap")
                    corr = st.session_state.df[num_cols_all].corr()
                    fig2 = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f")
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1E293B"))
                    st.plotly_chart(fig2, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("3. Feature vs Target")
                nf = [f for f in st.session_state.feature_cols if f in num_cols_all]
                if nf:
                    sf2 = st.selectbox("Feature", nf, key="svf")
                    if st.session_state.task_type == "Classification":
                        fig3 = px.box(st.session_state.df, x=tc, y=sf2, color=tc)
                    else:
                        fig3 = px.scatter(st.session_state.df, x=sf2, y=tc, opacity=.6,
                                          trendline="ols" if len(st.session_state.df) < 5000 else None,
                                          color_discrete_sequence=["#4F46E5"])
                    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1E293B"))
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No numeric features for relationship charts.")
                st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 4: ML / Clustering ────────────────────────────
    with tab_ml:
        conf = (
            (st.session_state.task_type == "Clustering" and bool(st.session_state.feature_cols)) or
            (st.session_state.task_type != "Clustering" and st.session_state.target_col is not None)
        )
        if not conf:
            st.warning("Configure settings in the **Business Goal & Target** tab first.")
        elif st.session_state.task_type == "Clustering":
            st.subheader("🔵 Unsupervised Clustering Pipeline")
            st.markdown(f"Fitting **K-Means** & **Gaussian Mixture Model** with K={st.session_state.n_clusters}.")
            if st.button("🚀 Run Clustering"):
                with st.spinner("Clustering…"):
                    try:
                        cr = train_clustering(
                            st.session_state.df, st.session_state.feature_cols,
                            n_clusters=st.session_state.n_clusters)
                        bcn = max(cr, key=lambda m: cr[m]['metrics'].get("Silhouette Score", -1))
                        bl  = cr[bcn]['labels']
                        st.session_state.feature_cols      = cr[bcn]['valid_features']
                        st.session_state.cluster_results   = cr
                        st.session_state.best_cluster_name = bcn
                        st.session_state.cluster_labels    = bl
                        st.session_state.validation_logs = run_validation_checks(
                            st.session_state.df, None, st.session_state.feature_cols,
                            "Clustering", None, None, cr[bcn]['metrics'])
                        st.session_state.markdown_report = generate_markdown_report(
                            st.session_state.profile, None, st.session_state.feature_cols,
                            "Clustering", bcn, cr[bcn]['metrics'],
                            st.session_state.validation_logs,
                            st.session_state.business_question, st.session_state.target_audience)
                        st.session_state.reproducible_code = generate_reproducible_code(
                            st.session_state.file_name, None, st.session_state.feature_cols,
                            "Clustering", bcn, n_clusters=st.session_state.n_clusters)
                        with tempfile.TemporaryDirectory() as td:
                            sc = create_static_plots(
                                st.session_state.df, None, st.session_state.feature_cols,
                                "Clustering", None, None, td, cluster_labels=bl)
                            pp = os.path.join(td, "report.pdf")
                            generate_pdf_report(
                                st.session_state.profile, None, st.session_state.feature_cols,
                                "Clustering", bcn, cr[bcn]['metrics'],
                                st.session_state.validation_logs,
                                st.session_state.business_question, st.session_state.target_audience,
                                sc, pp)
                            with open(pp, "rb") as f:
                                st.session_state.pdf_report = f.read()
                        st.success("Clustering complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback; st.code(traceback.format_exc())

            if st.session_state.cluster_results is not None:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("🏆 Model Comparison")
                rows = [{"Model": m, **{k: f"{v:.4f}" for k, v in r['metrics'].items()}}
                        for m, r in st.session_state.cluster_results.items()]
                st.table(pd.DataFrame(rows))
                st.markdown(f"⭐ Best (Silhouette): **{st.session_state.best_cluster_name}**")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📋 Cluster Profiles")
                st.dataframe(get_cluster_profiles(
                    st.session_state.df, st.session_state.feature_cols,
                    st.session_state.cluster_labels), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                br2 = st.session_state.cluster_results[st.session_state.best_cluster_name]
                dp2 = generate_clustering_diagnostic_plots(
                    st.session_state.best_cluster_name, br2['X_preprocessed'],
                    st.session_state.cluster_labels, st.session_state.n_clusters)
                cl2, cr2 = st.columns(2)
                for idx, (pn, pf) in enumerate(dp2.items()):
                    with (cl2 if idx % 2 == 0 else cr2):
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.plotly_chart(pf, use_container_width=True, key=f"clustering_chart_{pn}")
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.subheader("🤖 Baseline Machine Learning Pipeline")
            st.markdown("80/20 split · imputation · scaling · encoding · three models compared.")
            if st.button("🚀 Train & Evaluate Models"):
                with st.spinner("Training…"):
                    try:
                        X_tr, X_te, y_tr, y_te, pre, valid_feats = preprocess_and_split(
                            st.session_state.df, st.session_state.target_col,
                            st.session_state.feature_cols, st.session_state.task_type)
                        st.session_state.feature_cols    = valid_feats
                        res = train_baselines(X_tr, y_tr, X_te, y_te, pre, st.session_state.task_type)
                        bn, bi = get_best_model(res, st.session_state.task_type)
                        st.session_state.ml_results      = res
                        st.session_state.best_model_name = bn
                        st.session_state.best_model_info = bi
                        tp = bi['pipeline'].predict(X_tr)
                        if st.session_state.task_type == "Classification":
                            from sklearn.metrics import f1_score as f1e
                            avg = "binary" if len(np.unique(y_tr)) == 2 else "weighted"
                            trs = f1e(y_tr, tp, average=avg, zero_division=0)
                            tes = bi['metrics']['F1-Score']
                        else:
                            from sklearn.metrics import r2_score as r2e
                            trs = r2e(y_tr, tp); tes = bi['metrics']['R²']
                        st.session_state.validation_logs = run_validation_checks(
                            st.session_state.df, st.session_state.target_col,
                            st.session_state.feature_cols, st.session_state.task_type,
                            trs, tes, bi['metrics'])
                        st.session_state.markdown_report = generate_markdown_report(
                            st.session_state.profile, st.session_state.target_col,
                            st.session_state.feature_cols, st.session_state.task_type,
                            bn, bi['metrics'], st.session_state.validation_logs,
                            st.session_state.business_question, st.session_state.target_audience)
                        st.session_state.reproducible_code = generate_reproducible_code(
                            st.session_state.file_name, st.session_state.target_col,
                            st.session_state.feature_cols, st.session_state.task_type, bn)
                        with tempfile.TemporaryDirectory() as td:
                            sc = create_static_plots(
                                st.session_state.df, st.session_state.target_col,
                                st.session_state.feature_cols, st.session_state.task_type,
                                y_te, bi['y_pred'], td)
                            pp = os.path.join(td, "report.pdf")
                            generate_pdf_report(
                                st.session_state.profile, st.session_state.target_col,
                                st.session_state.feature_cols, st.session_state.task_type,
                                bn, bi['metrics'], st.session_state.validation_logs,
                                st.session_state.business_question, st.session_state.target_audience,
                                sc, pp)
                            with open(pp, "rb") as f: st.session_state.pdf_report = f.read()
                        st.success("Models trained!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback; st.code(traceback.format_exc())

            if st.session_state.ml_results is not None:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("🏆 Model Comparison")
                rows = [{"Model": m, **{k: f"{v:.4f}" for k, v in r['metrics'].items()}}
                        for m, r in st.session_state.ml_results.items()]
                st.table(pd.DataFrame(rows))
                st.markdown(f"⭐ Best: **{st.session_state.best_model_name}**")
                st.markdown("</div>", unsafe_allow_html=True)
                cl3, cr3 = st.columns(2)
                with cl3:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Feature Importances")
                    idf = extract_feature_importances(
                        st.session_state.best_model_name,
                        st.session_state.best_model_info['pipeline'],
                        st.session_state.feature_cols)
                    if idf is not None:
                        fig_i = px.bar(idf.head(15), x="Importance", y="Feature", orientation="h",
                                       title="Top Drivers", color="Importance", color_continuous_scale="Purples")
                        fig_i.update_layout(yaxis={"categoryorder": "total ascending"},
                                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                            font=dict(color="#1E293B"))
                        st.plotly_chart(fig_i, use_container_width=True)
                    else:
                        st.info("Importances not available for this model.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with cr3:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Diagnostic Charts")
                    _, _, _, y_t2, _, _ = preprocess_and_split(
                        st.session_state.df, st.session_state.target_col,
                        st.session_state.feature_cols, st.session_state.task_type)
                    ep = generate_evaluation_plots(
                        st.session_state.best_model_name, y_t2,
                        st.session_state.best_model_info['y_pred'],
                        st.session_state.best_model_info['y_prob'],
                        st.session_state.task_type)
                    if ep:
                        sp = st.selectbox("Plot", list(ep.keys()))
                        st.plotly_chart(ep[sp], use_container_width=True)
                    else:
                        st.info("No diagnostics for this configuration.")
                    st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 5: Validation ─────────────────────────────────
    with tab_validation:
        if st.session_state.validation_logs is None:
            st.warning("Run the pipeline in the **Baseline ML / Clustering** tab first.")
        else:
            st.subheader("🛡️ Automated Trust & Validation Log")
            for log in st.session_state.validation_logs:
                s = log['status']
                if s == 'CRITICAL': cls, icon, bd = "status-critical", "🔴", "#B91C1C"
                elif s == 'WARNING': cls, icon, bd = "status-warning",  "🟡", "#92400E"
                else:               cls, icon, bd = "status-info",     "🟢", "#065F46"
                st.markdown(f"""
                <div class="log-card" style="border-left:5px solid {bd};">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="log-title">{icon} {log['check_name']}</span>
                        <span class="status-badge {cls}">{s}</span>
                    </div>
                    <p class="log-msg">{log['message']}</p>
                    <p class="log-det">{log['details']}</p>
                </div>""", unsafe_allow_html=True)

    # ── TAB 6: Export ─────────────────────────────────────
    with tab_export:
        if st.session_state.markdown_report is None:
            st.warning("Run the pipeline in the **Baseline ML / Clustering** tab first.")
        else:
            st.subheader("📄 Export Artifacts")
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📑 Reports")
                if st.session_state.pdf_report:
                    st.download_button("📥 PDF Report", st.session_state.pdf_report,
                                       "report.pdf", "application/pdf")
                st.download_button("📝 Markdown Report", st.session_state.markdown_report,
                                   "report.md", "text/markdown")
                with st.expander("Preview Markdown"):
                    st.markdown(st.session_state.markdown_report)
                st.markdown("</div>", unsafe_allow_html=True)
            with ec2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("🐍 Code & Bundle")
                st.download_button("🐍 Python Script", st.session_state.reproducible_code,
                                   "analysis.py", "text/plain")
                zb = io.BytesIO()
                with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("report.md", st.session_state.markdown_report)
                    zf.writestr("analysis.py", st.session_state.reproducible_code)
                    if st.session_state.pdf_report:
                        zf.writestr("report.pdf", st.session_state.pdf_report)
                    try:
                        if not st.session_state.file_name.endswith((".db", ".sqlite", ".sqlite3")):
                            zf.writestr(f"dataset_{st.session_state.file_name}",
                                        st.session_state.df.to_csv(index=False))
                    except: pass
                st.download_button("📦 Analysis ZIP", zb.getvalue(),
                                   "analysis_package.zip", "application/zip")
                with st.expander("Preview Python Code"):
                    st.code(st.session_state.reproducible_code, language="python")
                st.markdown("</div>", unsafe_allow_html=True)
