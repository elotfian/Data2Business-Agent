import os
import tempfile
import zipfile
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import local modules
from src.profiler import load_dataset, get_sqlite_tables, profile_dataset, recommend_kpis_and_targets
from src.ml_engine import preprocess_and_split, train_baselines, get_best_model, extract_feature_importances, generate_evaluation_plots
from src.validator import run_validation_checks
from src.reporter import generate_markdown_report, generate_reproducible_code, create_static_plots, generate_pdf_report

# Page Config
st.set_page_config(
    page_title="Data2Business Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphism & Sleek Dark Mode Styling
st.markdown("""
<style>
    /* Dark Theme backgrounds */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        color: #F8FAFC;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Cards and Glassmorphism */
    div.element-container:has(div.card) {
        margin-bottom: 1rem;
    }
    .card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .main-title {
        background: linear-gradient(90deg, #A78BFA 0%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    /* Status indicators */
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-critical {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-info {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6D28D9 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(109, 40, 217, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 18px rgba(109, 40, 217, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""
if 'sqlite_tables' not in st.session_state:
    st.session_state.sqlite_tables = []
if 'selected_table' not in st.session_state:
    st.session_state.selected_table = None
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'kpi_recs' not in st.session_state:
    st.session_state.kpi_recs = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'task_type' not in st.session_state:
    st.session_state.task_type = "Classification"
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'best_model_info' not in st.session_state:
    st.session_state.best_model_info = None
if 'validation_logs' not in st.session_state:
    st.session_state.validation_logs = None
if 'markdown_report' not in st.session_state:
    st.session_state.markdown_report = None
if 'reproducible_code' not in st.session_state:
    st.session_state.reproducible_code = None
if 'pdf_report' not in st.session_state:
    st.session_state.pdf_report = None

def reset_state():
    st.session_state.df = None
    st.session_state.profile = None
    st.session_state.kpi_recs = None
    st.session_state.target_col = None
    st.session_state.feature_cols = []
    st.session_state.ml_results = None
    st.session_state.best_model_name = None
    st.session_state.best_model_info = None
    st.session_state.validation_logs = None
    st.session_state.markdown_report = None
    st.session_state.reproducible_code = None
    st.session_state.pdf_report = None

# Sidebar Content
with st.sidebar:
    st.markdown("<h1 style='font-size: 1.8rem; margin-bottom: 0;'>🤖 Data2Business</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Business Analytics & ML Assistant</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📊 Workflow Progress")
    steps = ["1. Upload & Profile", "2. Goal & Targets", "3. Visual Insights", "4. Baseline ML Model", "5. Trust Validation", "6. Report Export"]
    
    # Progress Calculation
    curr_step = 0
    if st.session_state.df is not None:
        curr_step = 1
    if st.session_state.target_col is not None:
        curr_step = 2
    if st.session_state.ml_results is not None:
        curr_step = 5
        
    for i, step in enumerate(steps):
        if i < curr_step:
            st.markdown(f"🟢 **{step}**")
        elif i == curr_step:
            st.markdown(f"🔵 **{step}** (Active)")
        else:
            st.markdown(f"⚪ {step}")
            
    st.markdown("---")
    
    # Sidebar stats
    if st.session_state.df is not None:
        st.markdown("### 📈 Active Dataset")
        st.markdown(f"- **File:** `{st.session_state.file_name}`")
        if st.session_state.selected_table:
            st.markdown(f"- **Table:** `{st.session_state.selected_table}`")
        st.markdown(f"- **Rows:** {st.session_state.df.shape[0]}")
        st.markdown(f"- **Cols:** {st.session_state.df.shape[1]}")
        
        if st.button("Clear / Reset", key="clear_btn"):
            reset_state()
            st.session_state.uploaded_file = None
            st.rerun()

# Main Title Header
st.markdown("<h1 class='main-title'>Data2Business Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #94A3B8; margin-top:-5px;'>Transform raw data into trustworthy insights, KPIs, and baseline models.</p>", unsafe_allow_html=True)

# ----------------- STEP 1: UPLOAD & PROFILE -----------------
if st.session_state.df is None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📁 Step 1: Upload your dataset")
    st.markdown("Supported formats: CSV, Excel (`.xlsx`, `.xls`), JSON, Parquet, or SQLite databases.")
    
    uploaded_file = st.file_uploader("Choose a dataset file", type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'db', 'sqlite', 'sqlite3'])
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.file_name = uploaded_file.name
        
        # Check if SQLite
        _, ext = os.path.splitext(uploaded_file.name.lower())
        if ext in ['.db', '.sqlite', '.sqlite3']:
            # SQLite requires a physical file path. We write uploaded buffer to a temporary file.
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            tables = get_sqlite_tables(tmp_path)
            st.session_state.sqlite_tables = tables
            
            selected_table = st.selectbox("Select table to analyze", tables)
            if st.button("Load Table"):
                with st.spinner("Loading SQLite database table..."):
                    df = load_dataset(tmp_path, file_type='sqlite', table_name=selected_table)
                    st.session_state.df = df
                    st.session_state.selected_table = selected_table
                    # Cleanup tmp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    st.rerun()
        else:
            with st.spinner("Parsing dataset..."):
                try:
                    df = load_dataset(uploaded_file)
                    st.session_state.df = df
                    st.session_state.selected_table = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Showcase a friendly welcome guide if empty
    st.markdown("""
    <div style='margin-top: 2rem; border-left: 4px solid #8B5CF6; padding-left: 1rem;'>
        <h4>💡 How it works</h4>
        <p>1. <b>Upload Data:</b> Profile and examine column types, missingness, and general health metrics.<br>
        2. <b>Select Target Outcome:</b> Specify what you want to predict (regression or classification) and key features.<br>
        3. <b>Explore Visualizations:</b> View automatically generated charts mapping trends and class distributions.<br>
        4. <b>Train Baseline ML:</b> Train three different models in parallel, compare scores, and extract driver importance.<br>
        5. <b>Validate & Trust:</b> Run quality checks for leakage, class imbalance, and multicollinearity.<br>
        6. <b>Generate Report:</b> Download executive, manager, or technical-tailored reports, charts, and clean executable code.</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------- MAIN FLOW (WHEN DATA IS LOADED) -----------------
else:
    # Always profile on first load
    if st.session_state.profile is None:
        with st.spinner("Profiling dataset..."):
            st.session_state.profile = profile_dataset(st.session_state.df)
            st.session_state.kpi_recs = recommend_kpis_and_targets(st.session_state.profile)

    # Show profiling dashboard tabs
    tab_profile, tab_goal, tab_viz, tab_ml, tab_validation, tab_export = st.tabs([
        "📊 Data Profile", "🎯 Business Goal & Target", "📈 Visual Insights", "🤖 Baseline ML", "🛡️ Trust Validation", "📄 Report Export"
    ])
    
    # ----------------- TAB: DATA PROFILE -----------------
    with tab_profile:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Data profiling overview")
        
        # Summary KPI cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Row Count", st.session_state.profile['num_rows'])
        col2.metric("Column Count", st.session_state.profile['num_cols'])
        col3.metric("Duplicate Rows", st.session_state.profile['num_duplicates'])
        
        # Calculate overall completeness
        total_cells = st.session_state.df.size
        null_cells = st.session_state.df.isnull().sum().sum()
        completeness = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
        col4.metric("Data Completeness", f"{completeness:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        st.subheader("Column Profiles")
        
        # Table of columns
        col_list = []
        for col_info in st.session_state.profile['columns']:
            stats_str = ""
            if col_info['semantic_type'] == 'Numeric' and 'stats' in col_info:
                stats_str = f"Mean: {col_info['stats']['mean']:.2f} | Range: [{col_info['stats']['min']:.2f}, {col_info['stats']['max']:.2f}]"
            elif col_info['semantic_type'] == 'Categorical' and 'stats' in col_info and 'top_values' in col_info['stats']:
                top_v = col_info['stats']['top_values']
                if top_v:
                    stats_str = f"Top: '{top_v[0]['value']}' ({top_v[0]['count']} rows)"
            elif col_info['semantic_type'] == 'Datetime' and 'stats' in col_info and 'min' in col_info['stats']:
                stats_str = f"Range: {col_info['stats']['min']} to {col_info['stats']['max']}"
                
            col_list.append({
                "Column Name": col_info['column_name'],
                "Data Type": col_info['data_type'],
                "Semantic Type": col_info['semantic_type'],
                "Missing %": f"{col_info['missing_percentage']}%",
                "Unique Count": col_info['unique_count'],
                "Summary Statistics": stats_str
            })
            
        st.table(pd.DataFrame(col_list))
        
    # ----------------- TAB: GOAL & TARGETS -----------------
    with tab_goal:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Define business context & variables")
        
        business_question = st.text_input(
            "What business or analytics question are you trying to answer?",
            value="Predict customer churn based on transaction history and account profiles"
        )
        
        target_audience = st.selectbox(
            "Who is the target audience for the final report?",
            options=["technical", "manager", "executive"],
            index=1,
            format_func=lambda x: x.capitalize()
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display recommendations
        if st.session_state.kpi_recs:
            st.subheader("💡 Recommended Target Outcomes")
            st.markdown("The agent has analyzed columns based on variance, cardinality, and naming patterns:")
            
            recs = st.session_state.kpi_recs['recommended_targets']
            if recs:
                rec_df = pd.DataFrame([
                    {
                        "Column": r['column'],
                        "Semantic Type": r['type'],
                        "Suggested Task": r['task_suggestion'],
                        "Confidence Score": "High ⭐⭐⭐" if r['score'] >= 5 else "Medium ⭐⭐"
                    } for r in recs
                ])
                st.table(rec_df)
            else:
                st.info("No clear target variables could be recommended automatically. Please select manually below.")
                
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Configure Modeling Pipeline")
        
        # Target Selector
        all_cols = [c['column_name'] for c in st.session_state.profile['columns']]
        
        # Auto-select target from recommendations if none selected yet
        default_target_idx = 0
        if st.session_state.target_col is None and st.session_state.kpi_recs['recommended_targets']:
            rec_target = st.session_state.kpi_recs['recommended_targets'][0]['column']
            if rec_target in all_cols:
                default_target_idx = all_cols.index(rec_target)
        elif st.session_state.target_col in all_cols:
            default_target_idx = all_cols.index(st.session_state.target_col)
            
        target_col = st.selectbox("Select Target Variable (Outcome to predict)", all_cols, index=default_target_idx)
        
        # Update task type based on target semantic type
        target_sem_type = next((c['semantic_type'] for c in st.session_state.profile['columns'] if c['column_name'] == target_col), "Categorical")
        suggested_task = "Classification" if target_sem_type == "Categorical" else "Regression"
        
        task_type = st.radio("Analytics Task Type", ["Classification", "Regression"], 
                             index=0 if suggested_task == "Classification" else 1,
                             help="Classification for category labels, Regression for numerical quantities.")
        
        # Feature Selector (Default to all cols except target and ID/Text cols)
        default_features = []
        for col_info in st.session_state.profile['columns']:
            name = col_info['column_name']
            sem = col_info['semantic_type']
            if name != target_col and sem not in ['ID', 'Text']:
                default_features.append(name)
                
        feature_cols = st.multiselect("Select Feature Columns (Predictors)", all_cols, default=default_features)
        
        # Save button to update state
        if st.button("Save Settings & Apply"):
            st.session_state.target_col = target_col
            st.session_state.task_type = task_type
            st.session_state.feature_cols = feature_cols
            
            # Reset down-stream metrics to force recalculation
            st.session_state.ml_results = None
            st.session_state.best_model_name = None
            st.session_state.best_model_info = None
            st.session_state.validation_logs = None
            st.session_state.markdown_report = None
            st.session_state.pdf_report = None
            
            st.success(f"Configured: predicting `{target_col}` ({task_type}) using {len(feature_cols)} features.")
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    # ----------------- TAB: VISUAL INSIGHTS -----------------
    with tab_viz:
        if st.session_state.target_col is None:
            st.warning("Please configure and save your target variable in the 'Business Goal & Target' tab first.")
        else:
            st.subheader(f"📈 Visual Insights: Analyzing predictors for outcome `{st.session_state.target_col}`")
            
            # Row 1: Target distribution
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader(f"1. Target Distribution: `{st.session_state.target_col}`")
            if st.session_state.task_type == 'Classification':
                fig_target = px.histogram(
                    st.session_state.df, 
                    x=st.session_state.target_col, 
                    color=st.session_state.target_col,
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
            else:
                fig_target = px.histogram(
                    st.session_state.df, 
                    x=st.session_state.target_col, 
                    kde=True,
                    color_discrete_sequence=['#8B5CF6']
                )
            fig_target.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E2E8F0")
            )
            st.plotly_chart(fig_target, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Row 2: Correlation Heatmap (for Numeric Columns)
            num_cols = [c['column_name'] for c in st.session_state.profile['columns'] if c['semantic_type'] == 'Numeric']
            if len(num_cols) > 1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("2. Correlation Heatmap (Numeric Columns)")
                corr_matrix = st.session_state.df[num_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix, 
                    color_continuous_scale='RdBu_r', 
                    zmin=-1, zmax=1,
                    text_auto=".2f"
                )
                fig_corr.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#E2E8F0")
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Row 3: Feature vs Target relationships
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("3. Feature Relationships vs Target")
            
            # Pick a numeric feature to plot vs target
            numeric_features = [f for f in st.session_state.feature_cols if f in num_cols]
            if numeric_features:
                selected_feat = st.selectbox("Select feature to plot vs Target", numeric_features)
                
                if st.session_state.task_type == 'Classification':
                    # Box plot
                    fig_rel = px.box(
                        st.session_state.df, 
                        x=st.session_state.target_col, 
                        y=selected_feat, 
                        color=st.session_state.target_col
                    )
                else:
                    # Scatter plot
                    fig_rel = px.scatter(
                        st.session_state.df, 
                        x=selected_feat, 
                        y=st.session_state.target_col, 
                        opacity=0.6,
                        trendline="ols" if len(st.session_state.df) < 5000 else None
                    )
                    
                fig_rel.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#E2E8F0")
                )
                st.plotly_chart(fig_rel, use_container_width=True)
            else:
                st.info("No numeric features found to plot relationship charts.")
            st.markdown("</div>", unsafe_allow_html=True)
            
    # ----------------- TAB: BASELINE ML -----------------
    with tab_ml:
        if st.session_state.target_col is None:
            st.warning("Please configure your target variable in the 'Business Goal & Target' tab first.")
        else:
            st.subheader("🤖 Run Baseline Machine Learning Pipeline")
            st.markdown("The agent will automatically partition data (80/20 train/test), impute missing values, scale variables, encode categoricals, and evaluate three models.")
            
            if st.button("🚀 Train & Evaluate Models"):
                with st.spinner("Executing pipeline, preprocessing data, training classifiers/regressors..."):
                    try:
                        # 1. Preprocess & Split
                        X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(
                            st.session_state.df, 
                            st.session_state.target_col, 
                            st.session_state.feature_cols, 
                            st.session_state.task_type
                        )
                        
                        # 2. Train baseline models
                        results = train_baselines(
                            X_train, y_train, X_test, y_test, 
                            preprocessor, st.session_state.task_type
                        )
                        
                        # 3. Select best model
                        best_name, best_info = get_best_model(results, st.session_state.task_type)
                        
                        # Save to session state
                        st.session_state.ml_results = results
                        st.session_state.best_model_name = best_name
                        st.session_state.best_model_info = best_info
                        
                        # Extract train score to pass to validator (for overfitting checks)
                        train_pred = best_info['pipeline'].predict(X_train)
                        metric_name = 'F1-Score' if st.session_state.task_type == 'Classification' else 'R²'
                        if st.session_state.task_type == 'Classification':
                            from sklearn.metrics import f1_score as f1_eval
                            is_binary = len(np.unique(y_train)) == 2
                            avg = 'binary' if is_binary else 'weighted'
                            train_score = f1_eval(y_train, train_pred, average=avg, zero_division=0)
                            test_score = best_info['metrics']['F1-Score']
                        else:
                            from sklearn.metrics import r2_score as r2_eval
                            train_score = r2_eval(y_train, train_pred)
                            test_score = best_info['metrics']['R²']
                            
                        # 4. Run automated validation checks
                        st.session_state.validation_logs = run_validation_checks(
                            st.session_state.df, 
                            st.session_state.target_col, 
                            st.session_state.feature_cols, 
                            st.session_state.task_type,
                            train_score, 
                            test_score,
                            best_info['metrics']
                        )
                        
                        # 5. Generate Markdown report content
                        st.session_state.markdown_report = generate_markdown_report(
                            st.session_state.profile, 
                            st.session_state.target_col, 
                            st.session_state.feature_cols, 
                            st.session_state.task_type, 
                            best_name, 
                            best_info['metrics'], 
                            st.session_state.validation_logs, 
                            business_question, 
                            target_audience
                        )
                        
                        # 6. Generate Reproducible script
                        st.session_state.reproducible_code = generate_reproducible_code(
                            st.session_state.file_name, 
                            st.session_state.target_col, 
                            st.session_state.feature_cols, 
                            st.session_state.task_type, 
                            best_name
                        )
                        
                        # 7. Generate PDF report using static matplotlib plots
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            static_charts = create_static_plots(
                                st.session_state.df, 
                                st.session_state.target_col, 
                                st.session_state.feature_cols, 
                                st.session_state.task_type, 
                                y_test, 
                                best_info['y_pred'], 
                                tmp_dir
                            )
                            
                            pdf_buf_path = os.path.join(tmp_dir, "report.pdf")
                            generate_pdf_report(
                                st.session_state.profile,
                                st.session_state.target_col,
                                st.session_state.feature_cols,
                                st.session_state.task_type,
                                best_name,
                                best_info['metrics'],
                                st.session_state.validation_logs,
                                business_question,
                                target_audience,
                                static_charts,
                                pdf_buf_path
                            )
                            
                            # Read PDF into bytes for downloadable buffer
                            with open(pdf_buf_path, "rb") as f:
                                st.session_state.pdf_report = f.read()
                                
                        st.success("Baseline models trained successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error executing ML Pipeline: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        
            # Show results if available
            if st.session_state.ml_results is not None:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("🏆 Baseline Comparisons")
                
                # Metrics comparison table
                comparison_data = []
                for name, res in st.session_state.ml_results.items():
                    row = {"Model": name}
                    row.update({m: f"{v:.4f}" for m, v in res['metrics'].items()})
                    comparison_data.append(row)
                    
                st.table(pd.DataFrame(comparison_data))
                
                st.markdown(f"⭐ Best performing model based on metrics: **{st.session_state.best_model_name}**")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display Feature Importance and Evaluation Plots
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Feature Importances / Drivers")
                    
                    # Extract importances
                    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(
                        st.session_state.df, 
                        st.session_state.target_col, 
                        st.session_state.feature_cols, 
                        st.session_state.task_type
                    )
                    
                    imp_df = extract_feature_importances(
                        st.session_state.best_model_name, 
                        st.session_state.best_model_info['pipeline'], 
                        st.session_state.feature_cols
                    )
                    
                    if imp_df is not None:
                        fig_imp = px.bar(
                            imp_df.head(15), 
                            x='Importance', 
                            y='Feature', 
                            orientation='h',
                            title=f"Top Features driving prediction",
                            color='Importance',
                            color_continuous_scale='Purples'
                        )
                        fig_imp.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color="#E2E8F0")
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.info("Feature importances could not be directly extracted from the best model (e.g. Gradient Boosting without permutation feature weights). Standard linear/tree coefficient shapes are needed.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col_right:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Model Diagnostic Charts")
                    
                    # Evaluation Plots
                    eval_plots = generate_evaluation_plots(
                        st.session_state.best_model_name, 
                        y_test, 
                        st.session_state.best_model_info['y_pred'], 
                        st.session_state.best_model_info['y_prob'], 
                        st.session_state.task_type
                    )
                    
                    if eval_plots:
                        selected_plot = st.selectbox("Select diagnostic plot", list(eval_plots.keys()))
                        st.plotly_chart(eval_plots[selected_plot], use_container_width=True)
                    else:
                        st.info("No diagnostics plots generated for this configuration.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
    # ----------------- TAB: TRUST VALIDATION -----------------
    with tab_validation:
        if st.session_state.validation_logs is None:
            st.warning("Please train the machine learning models first in the 'Baseline ML' tab.")
        else:
            st.subheader("🛡️ Automated Trust & Validation Log")
            st.markdown("This panel runs checks for statistical health, data quality risks, and potential leakage:")
            
            for log in st.session_state.validation_logs:
                status = log['status']
                
                # HTML template for alerts
                if status == 'CRITICAL':
                    status_class = "status-critical"
                    icon = "🔴"
                elif status == 'WARNING':
                    status_class = "status-warning"
                    icon = "🟡"
                else:
                    status_class = "status-info"
                    icon = "🟢"
                    
                st.markdown(f"""
                <div class="card" style="border-left: 5px solid {'#EF4444' if status=='CRITICAL' else ('#F59E0B' if status=='WARNING' else '#10B981')};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight: 800; font-size:1.1rem;">{icon} {log['check_name']}</span>
                        <span class="status-badge {status_class}">{status}</span>
                    </div>
                    <p style="margin-top: 0.5rem; font-weight: 600; color: #F1F5F9;">{log['message']}</p>
                    <p style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 0;">{log['details']}</p>
                </div>
                """, unsafe_allow_html=True)

    # ----------------- TAB: REPORT EXPORT -----------------
    with tab_export:
        if st.session_state.markdown_report is None:
            st.warning("Please train the machine learning models first in the 'Baseline ML' tab.")
        else:
            st.subheader("📄 Export Artifacts & Analysis Package")
            st.markdown("Download printable reports, clean reproducible python code, and packages for deployment:")
            
            # Layout
            col_report, col_code = st.columns(2)
            
            with col_report:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Download Generated Reports")
                st.markdown("Download documents tailored to your chosen target audience:")
                
                # PDF Download button
                if st.session_state.pdf_report:
                    st.download_button(
                        label="📥 Download PDF Business Report",
                        data=st.session_state.pdf_report,
                        file_name="business_analytics_report.pdf",
                        mime="application/pdf"
                    )
                    
                # Markdown Download
                st.download_button(
                    label="📝 Download Markdown Report",
                    data=st.session_state.markdown_report,
                    file_name="business_analytics_report.md",
                    mime="text/markdown"
                )
                
                # Display markdown report inside Streamlit
                with st.expander("Preview Generated Report (Markdown)"):
                    st.markdown(st.session_state.markdown_report)
                    
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_code:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Reproducible Code & Packages")
                st.markdown("Get standalone, fully commented code that can run outside this dashboard:")
                
                st.download_button(
                    label="🐍 Download Reproducible Python Script",
                    data=st.session_state.reproducible_code,
                    file_name="reproducible_analysis.py",
                    mime="text/plain"
                )
                
                # Package creation: ZIP containing report, code, and (if size is okay) dataset
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    # Write markdown report
                    zip_file.writestr("business_analytics_report.md", st.session_state.markdown_report)
                    # Write code
                    zip_file.writestr("reproducible_analysis.py", st.session_state.reproducible_code)
                    # Write PDF if available
                    if st.session_state.pdf_report:
                        zip_file.writestr("business_analytics_report.pdf", st.session_state.pdf_report)
                    # Write dataset (if uploaded file is small/memory buffer)
                    try:
                        if not st.session_state.file_name.endswith(('.db', '.sqlite', '.sqlite3')):
                            # CSV string
                            csv_data = st.session_state.df.to_csv(index=False)
                            zip_file.writestr(f"dataset_{st.session_state.file_name}", csv_data)
                    except:
                        pass
                        
                st.download_button(
                    label="📦 Download Complete Analysis Zip Package",
                    data=zip_buffer.getvalue(),
                    file_name="data2business_analysis_package.zip",
                    mime="application/zip"
                )
                
                with st.expander("Preview Reproducible Python Code"):
                    st.code(st.session_state.reproducible_code, language="python")
                    
                st.markdown("</div>", unsafe_allow_html=True)
