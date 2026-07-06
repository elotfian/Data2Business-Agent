import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from fpdf import FPDF

def generate_markdown_report(profile, target, features, task_type, best_model_name, metrics, validation_logs, business_question, target_audience):
    """
    Generates a markdown report tailored to the target audience.
    """
    lines = []
    
    # Title Section
    lines.append(f"# Data2Business Analytics Report")
    lines.append(f"**Business Question:** {business_question}")
    lines.append(f"**Target Audience:** {target_audience.capitalize()}")
    lines.append(f"**Analysis Mode:** Predictive Modeling & ML Baseline")
    lines.append("")
    
    # 1. Executive Summary (Always present, but detail varies)
    lines.append("## 1. Executive Summary")
    
    # High-level business interpretation of the ML outcome
    if task_type == 'Classification':
        accuracy = metrics.get('Accuracy', 0)
        f1 = metrics.get('F1-Score', 0)
        performance_desc = f"with an F1-Score of {f1:.2%} (Accuracy: {accuracy:.2%})"
    elif task_type == 'Regression':
        r2 = metrics.get('R²', 0)
        rmse = metrics.get('RMSE', 0)
        performance_desc = f"with an R² of {r2:.2f} (RMSE: {rmse:.4f})"
    else: # Clustering
        silhouette = metrics.get('Silhouette Score', 0)
        performance_desc = f"with a Silhouette Score of {silhouette:.3f}"
        
    if target_audience == 'executive':
        lines.append("### Key Findings & Actionable Recommendations")
        if task_type == 'Clustering':
            lines.append(f"- **Unsupervised Segmentation:** The `{best_model_name}` algorithm analyzed `{profile['num_rows']}` records across `{len(features)}` features to identify latent subgroups.")
            lines.append(f"- **Segmentation Quality:** The model separated the data into distinct segments {performance_desc} (scores >0.25 show valid clusters).")
            lines.append("- **Business Utility:** These clusters define natural market segments or behavior profiles. Tailoring strategies to these individual groups will improve conversion, retention, and target resource allocation.")
        else:
            lines.append(f"- **Primary Target Variable:** `{target}` represents the main business outcome we are analyzing.")
            lines.append(f"- **Baseline Predictability:** We successfully trained a `{best_model_name}` model that predicts `{target}` {performance_desc}.")
            
            # Simple impact language
            if task_type == 'Classification':
                lines.append(f"- **Decision Support:** This model can automatically categorize and forecast `{target}` outcomes, helping optimize business decisions and target resources effectively.")
            else:
                lines.append(f"- **Forecasting Ability:** The model can estimate numerical `{target}` outcomes, providing key budget/revenue/pricing insights.")
            
            # Top 3 features
            lines.append("- **Core Drivers:** The primary factors influencing this business metric include the top features identified in the technical analysis (see details below).")
        
        # Validation warning check
        criticals = [log for log in validation_logs if log['status'] == 'CRITICAL']
        warnings = [log for log in validation_logs if log['status'] == 'WARNING']
        if criticals:
            lines.append(f"- **Important Data Quality Alert:** We detected {len(criticals)} critical anomalies (e.g., target leakage or severe imbalance) that should be addressed before deploying these insights.")
        elif warnings:
            lines.append(f"- **Data Quality Note:** {len(warnings)} observations were flagged (e.g., multicollinearity, missing data, or uneven segments).")
        else:
            lines.append("- **Reliability status:** The source data passed all sanity checks and is highly reliable for decision making.")
            
    elif target_audience == 'manager':
        lines.append("### Business Insights & Takeaways")
        if task_type == 'Clustering':
            lines.append(f"We analyzed `{profile['num_rows']}` records using `{len(features)}` features for behavioral clustering.")
            lines.append(f"The best performing baseline segmentation is **{best_model_name}** {performance_desc}.")
            lines.append("")
            lines.append("#### Key Insights:")
            lines.append(f"1. **Latent Subgroups:** The `{best_model_name}` baseline automatically partitions the database. These cohorts represent distinct user behaviors or product families.")
            lines.append("2. **Targeted Strategies:** Operational managers should analyze the cluster profiles table to align products/promotions with each cluster's specific averages and preferences.")
            lines.append("3. **Cluster Health:** Review the Validation Log below to make sure the partitions are reasonably balanced.")
        else:
            lines.append(f"We analyzed `{profile['num_rows']}` rows with `{len(features)}` feature columns to predict `{target}`.")
            lines.append(f"The best performing baseline model is **{best_model_name}** {performance_desc}.")
            lines.append("")
            lines.append("#### Key Insights:")
            lines.append(f"1. **Predictive Capability:** The `{best_model_name}` baseline represents the most robust starter model. It can be integrated into regular operational workflows to automate `{target}` checks.")
            lines.append("2. **Feature Impact:** Operational managers should prioritize monitoring the most important features (as shown in the charts) since they account for the vast majority of variance in outcomes.")
            lines.append("3. **Operational Risks:** Check the Validation Log section below to ensure data collection pipelines are stable.")
        
    else: # Technical
        lines.append("### Technical Overview")
        lines.append(f"- **Dataset dimensions:** {profile['num_rows']} rows, {profile['num_cols']} columns.")
        lines.append(f"- **Task type:** {task_type}")
        if task_type == 'Clustering':
            lines.append(f"- **Best pipeline:** Preprocessing (imputation + standard scaling) followed by a `{best_model_name}` partition model.")
            lines.append("- **Model evaluation criteria:** Silhouette Score (cohesion and separation).")
        else:
            lines.append(f"- **Best pipeline:** Preprocessing (scaling + imputing + encoding) followed by a `{best_model_name}` estimator.")
            lines.append(f"- **Model evaluation criteria:** Optimised on {'F1-Score' if task_type == 'Classification' else 'R²'}.")
            lines.append("- **Holdout validation:** 20% test partition, stratified where applicable.")
        
    lines.append("")
    
    # 2. Data Profile & Profiling Log
    if target_audience in ['manager', 'technical']:
        lines.append("## 2. Dataset Profile Summary")
        lines.append(f"- Total Rows: **{profile['num_rows']}**")
        lines.append(f"- Total Columns: **{profile['num_cols']}**")
        lines.append(f"- Duplicated Rows: **{profile['num_duplicates']}**")
        lines.append("")
        lines.append("| Column Name | Inferred Semantic Type | Missing % | Unique Count |")
        lines.append("| --- | --- | --- | --- |")
        for col in profile['columns']:
            lines.append(f"| `{col['column_name']}` | {col['semantic_type']} | {col['missing_percentage']}% | {col['unique_count']} |")
        lines.append("")
        
    # 3. Model Performance Section
    lines.append("## 3. Model Performance & Evaluation")
    lines.append(f"Selected Champion Model: **{best_model_name}**")
    lines.append("")
    lines.append("| Evaluation Metric | Value | Interpretation |")
    lines.append("| --- | --- | --- |")
    for m_name, m_val in metrics.items():
        # Human readable interpretation
        if m_name == 'Accuracy':
            desc = "Percentage of correct predictions overall."
        elif m_name == 'F1-Score':
            desc = "Balanced measure of precision and recall (target score for imbalanced classes)."
        elif m_name == 'ROC-AUC':
            desc = "Ability of the model to distinguish between classes (1.0 is perfect, 0.5 is random)."
        elif m_name == 'R²':
            desc = "Proportion of variance in target explained by features (1.0 is perfect)."
        elif m_name == 'RMSE':
            desc = "Standard deviation of prediction errors (lower is better)."
        elif m_name == 'MAE':
            desc = "Average absolute prediction error (lower is better)."
        elif m_name == 'Silhouette Score':
            desc = "Cohesion & separation of clusters (-1 to 1; higher is better, >0.25 is structured)."
        elif m_name == 'Davies-Bouldin Index':
            desc = "Average similarity between clusters (lower is better, 0.0 is perfect)."
        elif m_name == 'Calinski-Harabasz Index':
            desc = "Ratio of between-cluster variance to within-cluster variance (higher is better)."
        elif m_name == 'Inertia (Within Cluster Sum)':
            desc = "Sum of squared distances of samples to their closest cluster center (lower is better)."
        elif m_name == 'BIC (Bayesian Info Criterion)':
            desc = "Model complexity penalization metric (lower is better)."
        else:
            desc = "Model performance metric."
            
        lines.append(f"| **{m_name}** | {m_val:.4f} | {desc} |")
    lines.append("")
    
    # 4. Data Validation and Trust Log
    lines.append("## 4. Automated Data Validation & Trust Log")
    lines.append("This log flags anomalies, target leakage, class imbalance, and overfitting to ensure business trust.")
    lines.append("")
    
    for log in validation_logs:
        status_emoji = "🔴" if log['status'] == 'CRITICAL' else ("🟡" if log['status'] == 'WARNING' else "🟢")
        lines.append(f"### {status_emoji} {log['check_name']} (`{log['status']}`)")
        lines.append(f"- **Message:** {log['message']}")
        lines.append(f"- **Details:** {log['details']}")
        lines.append("")
        
    return "\n".join(lines)

def generate_reproducible_code(file_path_or_name, target_col, feature_cols, task_type, best_model_name, n_clusters=3):
    """
    Generates standalone executable python code that reproduces the analysis.
    """
    # Map model name to scikit-learn constructors
    if task_type == 'Classification':
        if best_model_name == 'Logistic Regression':
            model_import = "from sklearn.linear_model import LogisticRegression"
            model_inst = "model = LogisticRegression(max_iter=1000, random_state=42)"
        elif best_model_name == 'Random Forest':
            model_import = "from sklearn.ensemble import RandomForestClassifier"
            model_inst = "model = RandomForestClassifier(n_estimators=100, random_state=42)"
        else: # Gradient Boosting
            model_import = "from sklearn.ensemble import HistGradientBoostingClassifier"
            model_inst = "model = HistGradientBoostingClassifier(random_state=42)"
    elif task_type == 'Regression':
        if best_model_name == 'Ridge Regression':
            model_import = "from sklearn.linear_model import Ridge"
            model_inst = "model = Ridge(random_state=42)"
        elif best_model_name == 'Random Forest':
            model_import = "from sklearn.ensemble import RandomForestRegressor"
            model_inst = "model = RandomForestRegressor(n_estimators=100, random_state=42)"
        else: # Gradient Boosting
            model_import = "from sklearn.ensemble import HistGradientBoostingRegressor"
            model_inst = "model = HistGradientBoostingRegressor(random_state=42)"
    else: # Clustering
        if best_model_name == 'K-Means':
            model_import = "from sklearn.cluster import KMeans"
            model_inst = f"model = KMeans(n_clusters={n_clusters}, random_state=42, n_init='auto')"
        else: # GMM
            model_import = "from sklearn.mixture import GaussianMixture"
            model_inst = f"model = GaussianMixture(n_components={n_clusters}, random_state=42)"
            
    is_sqlite = str(file_path_or_name).endswith(('.db', '.sqlite', '.sqlite3'))
    
    code = f"""# ==============================================================================
# REPRODUCIBLE DATA ANALYSIS & MACHINE LEARNING PIPELINE
# Generated by Data2Business Agent
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
{model_import}
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
"""

    if task_type in ['Classification', 'Regression']:
        code += "from sklearn.model_selection import train_test_split\n"
        code += "from sklearn.metrics import classification_report, mean_squared_error, r2_score\n"

    code += """
# 1. LOAD DATA
print("Loading dataset...")
"""

    if is_sqlite:
        code += f"""import sqlite3
conn = sqlite3.connect("{file_path_or_name}")
df = pd.read_sql_query("SELECT * FROM sqlite_master WHERE type='table';", conn)
table_name = df.iloc[0]['name'] if len(df) > 0 else 'table'
df = pd.read_sql_query(f"SELECT * FROM [{{table_name}}]", conn)
conn.close()
"""
    else:
        _, ext = os.path.splitext(str(file_path_or_name).lower())
        if ext == '.csv':
            code += f'df = pd.read_csv("{file_path_or_name}")\n'
        elif ext in ['.xlsx', '.xls']:
            code += f'df = pd.read_excel("{file_path_or_name}")\n'
        elif ext == '.json':
            code += f'df = pd.read_json("{file_path_or_name}")\n'
        elif ext in ['.parquet', '.pq']:
            code += f'df = pd.read_parquet("{file_path_or_name}")\n'
        else:
            code += f'df = pd.read_csv("{file_path_or_name}")  # Inferred CSV\n'

    code += f"""
# 2. DEFINE FEATURES
feature_cols = {feature_cols}
X = df[feature_cols].copy()
"""

    if task_type in ['Classification', 'Regression']:
        code += f"""
target_col = "{target_col}"
y = df[target_col].copy()

# 3. SPLIT DATA
print("Splitting data into train and test sets...")
"""
        if task_type == 'Classification':
            code += """class_counts = y.value_counts()
stratify = y if (class_counts >= 2).all() else None
"""
        else:
            code += "stratify = None\n"

        code += """X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)
"""

    code += """
# PREPROCESSING PIPELINE
numeric_features = []
categorical_features = []
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 10:
        numeric_features.append(col)
    else:
        categorical_features.append(col)
        X[col] = X[col].astype(str)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

low_card_features = [col for col in categorical_features if X[col].nunique() <= 15]
high_card_features = [col for col in categorical_features if X[col].nunique() > 15]

transformers = []
if numeric_features:
    transformers.append(('num', numeric_transformer, numeric_features))
if low_card_features:
    transformers.append(('cat_ohe', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), low_card_features))
if high_card_features:
    transformers.append(('cat_ord', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]), high_card_features))

preprocessor = ColumnTransformer(transformers=transformers)
"""

    if task_type == 'Clustering':
        code += f"""
# 4. FIT CLUSTERING MODEL
print("Fitting {best_model_name} clustering model...")
X_preprocessed = preprocessor.fit_transform(X)
model = {model_inst.split(' = ')[1]}
"""
        if best_model_name == 'K-Means':
            code += """model.fit(X_preprocessed)
labels = model.labels_
"""
        else:
            code += """labels = model.fit_predict(X_preprocessed)
"""
        code += """
# 5. EVALUATE CLUSTERING
if len(set(labels)) > 1:
    sil = silhouette_score(X_preprocessed, labels)
    print(f"Silhouette Score: {sil:.4f}")

# 6. PCA VISUALISATION
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_preprocessed)
df_pca = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Cluster': labels})

plt.figure(figsize=(8, 6))
for cl in sorted(df_pca['Cluster'].unique()):
    subset = df_pca[df_pca['Cluster'] == cl]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=f'Cluster {cl}', alpha=0.7)
plt.legend()
plt.title('PCA 2D Cluster Visualisation')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.tight_layout()
plt.show()

# 7. CLUSTER PROFILES
df_out = df[feature_cols].copy()
df_out['Cluster'] = [f'Cluster {l}' for l in labels]
print(df_out.groupby('Cluster').mean(numeric_only=True))

print("Workflow complete!")
"""
    else:
        code += f"""
# 4. FIT SUPERVISED MODEL PIPELINE
print("Fitting {best_model_name} model pipeline...")
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', {model_inst.split(' = ')[1]})
])

pipeline.fit(X_train, y_train)

# 6. EVALUATE
print("Evaluating model...")
y_pred = pipeline.predict(X_test)

"""

    if task_type == 'Classification':
        code += """print("Classification Report:")
print(classification_report(y_test, y_pred))
"""
    else:
        code += """rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2 Score: {r2:.4f}")
"""

    code += """
# 7. VISUALIZE FEATURE IMPORTANCES (If available)
model_step = pipeline.named_steps['model']
preprocessor_step = pipeline.named_steps['preprocessor']

feature_names = []
for trans_name, transformer, columns in preprocessor_step.transformers_:
    if trans_name == 'num':
        feature_names.extend(columns)
    elif trans_name == 'cat_ohe':
        if hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
            ohe = transformer.named_steps['onehot']
            names = ohe.get_feature_names_out(columns)
            feature_names.extend(names)
        else:
            feature_names.extend(columns)
    elif trans_name == 'cat_ord':
        feature_names.extend(columns)

importances = None
if hasattr(model_step, 'feature_importances_'):
    importances = model_step.feature_importances_
elif hasattr(model_step, 'coef_'):
    coef = model_step.coef_
    importances = np.mean(np.abs(coef), axis=0) if len(coef.shape) > 1 else np.abs(coef)

if importances is not None and len(importances) == len(feature_names):
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.show()
else:
    print("Feature importances not easily extracted or mismatched dimensions.")

print("Workflow complete!")
"""
    return code

def create_static_plots(df, target_col, feature_cols, task_type, y_test, y_pred, output_dir, cluster_labels=None):
    """
    Generates and saves static PNG charts for inclusion in the PDF report.
    
    Returns a dict of chart name -> file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = {}
    
    # Set styling
    sns.set_theme(style="whitegrid")
    
    # 1. Target/Cluster Distribution Plot
    plt.figure(figsize=(6, 4))
    if task_type == 'Clustering' and cluster_labels is not None:
        import pandas as pd_local
        label_series = pd_local.Series([f"Cluster {l}" for l in cluster_labels])
        label_series.value_counts().sort_index().plot(kind='bar', color='steelblue')
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
    elif task_type == 'Classification' or (target_col is not None and df[target_col].nunique() <= 10):
        sns.countplot(data=df, x=target_col, hue=target_col, palette="pastel", legend=False)
        plt.title(f"Distribution of Target: {target_col}")
    else:
        sns.histplot(data=df, x=target_col, kde=True, color="purple")
        plt.title(f"Distribution of Target: {target_col}")
    plt.tight_layout()
    target_dist_path = os.path.join(output_dir, "target_distribution.png")
    plt.savefig(target_dist_path, dpi=150)
    plt.close()
    chart_paths['target_distribution'] = target_dist_path
    
    # 2. Performance Evaluation Chart
    plt.figure(figsize=(6, 4))
    if task_type == 'Clustering' and cluster_labels is not None:
        # PCA 2D scatter plot
        try:
            from sklearn.decomposition import PCA
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            import numpy as _np
            X_num = df[[c for c in feature_cols if df[c].dtype.kind in 'biufc']].copy()
            if X_num.shape[1] >= 2:
                X_imp = SimpleImputer(strategy='median').fit_transform(X_num)
                X_sc = StandardScaler().fit_transform(X_imp)
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(X_sc)
                scatter_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
                for idx, cl in enumerate(sorted(set(cluster_labels))):
                    mask = _np.array(cluster_labels) == cl
                    plt.scatter(coords[mask, 0], coords[mask, 1], label=f'Cluster {cl}', color=scatter_colors[idx % len(scatter_colors)], alpha=0.7)
                plt.legend(fontsize=8)
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                plt.title("PCA 2D Cluster Projection")
        except Exception:
            plt.text(0.5, 0.5, 'PCA projection unavailable', ha='center', va='center')
    elif task_type == 'Classification':
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
    else:
        # Prediction vs Actual
        plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        
    plt.tight_layout()
    eval_chart_path = os.path.join(output_dir, "model_evaluation.png")
    plt.savefig(eval_chart_path, dpi=150)
    plt.close()
    chart_paths['model_evaluation'] = eval_chart_path
    
    return chart_paths


class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        # Enable auto page breaks
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        self.set_fill_color(31, 41, 55) # Dark charcoal background
        self.rect(0, 0, 210, 15, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 10)
        self.cell(0, -5, 'Data2Business Agent - Analytics Report', 0, 0, 'L')
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def generate_pdf_report(profile, target, features, task_type, best_model_name, metrics, validation_logs, business_question, target_audience, chart_paths, output_path):
    """
    Generates a beautifully formatted PDF report including tables, logs, and static charts.
    """
    pdf = PDFReport()
    pdf.add_page()
    
    # Title Page/Header
    pdf.set_y(25)
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(79, 70, 229) # Purple indigo theme
    pdf.cell(0, 10, 'Data2Business Report', 0, 1, 'L')
    
    # Metadata block
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f"Business Question: {business_question}", 0, 1, 'L')
    pdf.cell(0, 6, f"Target Audience: {target_audience.capitalize()}", 0, 1, 'L')
    pdf.cell(0, 6, f"Dataset Size: {profile['num_rows']} rows, {profile['num_cols']} columns", 0, 1, 'L')
    pdf.ln(5)
    
    # 1. Executive Summary
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 10, '1. Executive Summary', 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(55, 65, 81)
    
    if task_type == 'Classification':
        accuracy = metrics.get('Accuracy', 0)
        f1 = metrics.get('F1-Score', 0)
        perf_text = f"with an F1-Score of {f1:.2%} (Accuracy: {accuracy:.2%})"
    else:
        r2 = metrics.get('R²', 0)
        rmse = metrics.get('RMSE', 0)
        perf_text = f"with an R² of {r2:.2f} (RMSE: {rmse:.4f})"
        
    summary_p = f"We have conducted a predictive baseline analysis to address the question: '{business_question}'. " \
                f"Using advanced preprocessing and automated feature parsing, a {best_model_name} model was selected as the optimal baseline " \
                f"{perf_text}."
    pdf.multi_cell(0, 6, summary_p)
    pdf.ln(5)
    
    # Embed Target Distribution Chart
    if chart_paths and 'target_distribution' in chart_paths:
        y_pos = pdf.get_y()
        # Draw target distribution chart on the left, model evaluation on the right
        pdf.image(chart_paths['target_distribution'], x=15, y=y_pos, w=85)
        if 'model_evaluation' in chart_paths:
            pdf.image(chart_paths['model_evaluation'], x=110, y=y_pos, w=85)
        pdf.ln(65) # Leave vertical space for the charts
        
    # 2. Performance Table
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 10, '2. Model Performance Metrics', 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    # Draw metrics table
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(243, 244, 246)
    pdf.cell(70, 8, 'Metric', 1, 0, 'L', True)
    pdf.cell(40, 8, 'Value', 1, 0, 'R', True)
    pdf.cell(80, 8, 'Explanation', 1, 1, 'L', True)
    
    pdf.set_font('Arial', '', 10)
    for m_name, m_val in metrics.items():
        if m_name == 'Accuracy':
            desc = "Correct classification rate overall"
        elif m_name == 'F1-Score':
            desc = "Balance of Precision and Recall"
        elif m_name == 'ROC-AUC':
            desc = "Class separation score"
        elif m_name == 'R²':
            desc = "Variance in target explained"
        elif m_name == 'RMSE':
            desc = "Root mean squared error"
        elif m_name == 'MAE':
            desc = "Mean absolute error"
        else:
            desc = "Performance metric"
            
        pdf.cell(70, 8, str(m_name), 1, 0, 'L')
        pdf.cell(40, 8, f"{m_val:.4f}", 1, 0, 'R')
        pdf.cell(80, 8, desc, 1, 1, 'L')
        
    pdf.ln(8)
    
    # 3. Validation & Trust Log
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 10, '3. Automated Data Validation & Trust Log', 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font('Arial', '', 10)
    for log in validation_logs:
        # Determine color for severity
        if log['status'] == 'CRITICAL':
            pdf.set_fill_color(254, 226, 226) # Light red
            pdf.set_text_color(153, 27, 27)   # Dark red
        elif log['status'] == 'WARNING':
            pdf.set_fill_color(254, 243, 199) # Light amber
            pdf.set_text_color(146, 64, 14)   # Dark amber
        else:
            pdf.set_fill_color(220, 252, 231) # Light green
            pdf.set_text_color(21, 128, 61)   # Dark green
            
        # Draw status box
        pdf.cell(35, 7, f"[{log['status']}]", 1, 0, 'C', True)
        pdf.set_text_color(31, 41, 55)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 7, f" {log['check_name']}", 0, 1, 'L')
        
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(55, 65, 81)
        pdf.multi_cell(0, 5, f"Message: {log['message']}\nDetails: {log['details']}")
        pdf.ln(2)
        
    # Output file
    pdf.output(output_path)
