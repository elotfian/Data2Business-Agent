import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

def preprocess_and_split(df, target_col, feature_cols, task_type, test_size=0.2, random_state=42):
    """
    Splits the dataframe and constructs a Scikit-Learn preprocessing pipeline.
    
    Returns:
    - X_train, X_test, y_train, y_test
    - preprocessor: fitted ColumnTransformer or pipeline
    """
    # Drop rows where target is null
    df_clean = df.dropna(subset=[target_col])
    if len(df_clean) == 0:
        raise ValueError(f"Target column '{target_col}' contains only missing values. Cannot train supervised models.")
        
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()
    
    # Train-test split
    stratify = y if task_type == 'Classification' and y.nunique() >= 2 else None
    
    # If a class is too rare (only 1 occurrence), stratify might fail. Check class counts:
    if stratify is not None:
        class_counts = y.value_counts()
        if (class_counts < 2).any():
            stratify = None
            
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # Filter out features that are completely null in X_train
    valid_features = [col for col in feature_cols if X_train[col].notnull().any()]
    if not valid_features:
        raise ValueError("All selected feature columns contain only missing values in the training split. Please select columns with valid data.")
        
    X_train = X_train[valid_features].copy()
    X_test = X_test[valid_features].copy()
    
    # Identify numeric and categorical features
    numeric_features = []
    categorical_features = []
    
    for col in valid_features:
        if pd.api.types.is_numeric_dtype(X_train[col]) and X_train[col].nunique() > 10:
            numeric_features.append(col)
        else:
            categorical_features.append(col)
            # Ensure categorical is string
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
            
    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # For categorical features, split by cardinality
    low_card_features = [col for col in categorical_features if X_train[col].nunique() <= 15]
    high_card_features = [col for col in categorical_features if X_train[col].nunique() > 15]
    
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
    
    return X_train, X_test, y_train, y_test, preprocessor, valid_features

def train_baselines(X_train, y_train, X_test, y_test, preprocessor, task_type):
    """
    Trains multiple baseline models and evaluates them.
    
    Returns:
    - results: dict of model name -> {pipeline, metrics, predictions}
    """
    results = {}
    
    if task_type == 'Classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': HistGradientBoostingClassifier(random_state=42)
        }
    else:
        models = {
            'Ridge Regression': Ridge(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': HistGradientBoostingRegressor(random_state=42)
        }
        
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        if task_type == 'Classification':
            unique_classes = np.unique(y_train)
            is_binary = len(unique_classes) == 2
            
            acc = accuracy_score(y_test, y_pred)
            
            # Use appropriate average for multi-class
            avg = 'binary' if is_binary else 'weighted'
            prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
            
            try:
                if is_binary:
                    # Probabilities for class 1
                    y_prob = pipeline.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                else:
                    y_prob = pipeline.predict_proba(X_test)
                    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            except Exception:
                auc = 0.0
                y_prob = None
                
            metrics = {
                'Accuracy': float(acc),
                'Precision': float(prec),
                'Recall': float(rec),
                'F1-Score': float(f1),
                'ROC-AUC': float(auc)
            }
        else:
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'MAE': float(mae),
                'MSE': float(mse),
                'RMSE': float(rmse),
                'R²': float(r2)
            }
            y_prob = None
            
        results[name] = {
            'pipeline': pipeline,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
    return results

def get_best_model(results, task_type):
    """
    Selects the best model based on F1-Score (Classification) or R² (Regression).
    """
    best_name = None
    best_score = -float('inf')
    metric_to_optimize = 'F1-Score' if task_type == 'Classification' else 'R²'
    
    for name, res in results.items():
        score = res['metrics'][metric_to_optimize]
        if score > best_score:
            best_score = score
            best_name = name
            
    return best_name, results[best_name]

def extract_feature_importances(best_model_name, fitted_pipeline, feature_cols):
    """
    Attempts to extract and align feature importances or model coefficients.
    """
    model = fitted_pipeline.named_steps['model']
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    
    # Retrieve feature names after preprocessing transformation
    feature_names = []
    
    # Go through transformers to collect names
    for trans_name, transformer, columns in preprocessor.transformers_:
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
        elif trans_name == 'remainder' and columns:
            # If there are remainder columns passed through
            if isinstance(columns, list):
                feature_names.extend(columns)
            elif isinstance(columns, np.ndarray):
                # Handle indices
                feature_names.extend([feature_cols[i] for i in columns])
                
    # Safeguard feature names length matches model coefficients/importances
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For multi-class logistic regression, coef_ is 2D. Take mean absolute weight or class 0.
        coef = model.coef_
        if len(coef.shape) > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
            
    if importances is not None and len(importances) == len(feature_names):
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        return df_imp
        
    # If lengths don't match, return simple importance mapped to raw features if possible
    # e.g. HistGradientBoosting doesn't natively expose feature_importances_ easily without permutation
    # So we return None or compute a simplified version
    return None

def generate_evaluation_plots(best_model_name, y_test, y_pred, y_prob, task_type):
    """
    Generates interactive Plotly plots for model evaluation.
    
    Returns:
    - plots: dict of name -> Plotly figure object
    """
    plots = {}
    
    if task_type == 'Classification':
        # 1. Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        labels = np.unique(y_test)
        
        # Convert numeric labels to strings if necessary
        labels_str = [str(l) for l in labels]
        
        fig_cm = px.imshow(
            cm,
            x=labels_str,
            y=labels_str,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            color_continuous_scale='Purples',
            text_auto=True,
            title=f"Confusion Matrix - {best_model_name}"
        )
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#E2E8F0"),
            title_font=dict(color="#F8FAFC", size=16)
        )
        plots['confusion_matrix'] = fig_cm
        
        # 2. ROC Curve Plot (Binary Classification)
        if len(labels) == 2 and y_prob is not None:
            # We need binary labels (0 and 1) or target encoding
            try:
                # Find positive class
                pos_class = labels[1]
                binary_y_test = (y_test == pos_class).astype(int)
                
                from sklearn.metrics import roc_curve
                fpr, tpr, thresholds = roc_curve(binary_y_test, y_prob)
                auc_val = roc_auc_score(binary_y_test, y_prob)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc_val:.3f})', line=dict(color='#8B5CF6', width=3)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='#64748B', dash='dash')))
                
                fig_roc.update_layout(
                    title=f"ROC Curve - {best_model_name}",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#E2E8F0"),
                    title_font=dict(color="#F8FAFC", size=16),
                    legend=dict(font=dict(size=10))
                )
                fig_roc.update_xaxes(showgrid=True, gridcolor='#334155')
                fig_roc.update_yaxes(showgrid=True, gridcolor='#334155')
                plots['roc_curve'] = fig_roc
            except Exception as e:
                # Silently skip ROC curve plot if calculations fail
                pass
    else:
        # Regression Plots
        # 1. Prediction vs Actual Plot
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            marker=dict(color='#8B5CF6', opacity=0.6),
            name='Predictions'
        ))
        
        # Perfect line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_pv.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='#EF4444', dash='dash'),
            name='Perfect Prediction'
        ))
        
        fig_pv.update_layout(
            title=f"Actual vs. Predicted - {best_model_name}",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#E2E8F0"),
            title_font=dict(color="#F8FAFC", size=16)
        )
        fig_pv.update_xaxes(showgrid=True, gridcolor='#334155')
        fig_pv.update_yaxes(showgrid=True, gridcolor='#334155')
        plots['prediction_vs_actual'] = fig_pv
        
        # 2. Residual Plot
        residuals = y_test - y_pred
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(color='#EC4899', opacity=0.6),
            name='Residuals'
        ))
        fig_res.add_trace(go.Scatter(
            x=[y_pred.min(), y_pred.max()],
            y=[0, 0],
            mode='lines',
            line=dict(color='#64748B', dash='dash'),
            name='Zero Residual'
        ))
        
        fig_res.update_layout(
            title=f"Residual Plot - {best_model_name}",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals (Actual - Predicted)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#E2E8F0"),
            title_font=dict(color="#F8FAFC", size=16)
        )
        fig_res.update_xaxes(showgrid=True, gridcolor='#334155')
        fig_res.update_yaxes(showgrid=True, gridcolor='#334155')
        plots['residuals'] = fig_res
        
    return plots

def train_clustering(df, feature_cols, n_clusters=3, random_state=42):
    """
    Fits baseline clustering models (K-Means and GMM) on features.
    
    Returns:
    - results: dict of model name -> {model, preprocessor, labels, metrics, X_preprocessed}
    """
    # Filter out feature columns that are completely empty (100% missing values)
    valid_features = [col for col in feature_cols if df[col].notnull().any()]
    if not valid_features:
        raise ValueError("All selected feature columns contain only missing values. Please select columns with valid data.")
        
    X = df[valid_features].copy()
    
    # 1. Preprocessing Pipeline
    numeric_features = []
    categorical_features = []
    
    for col in valid_features:
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
    
    # Preprocess features
    X_preprocessed = preprocessor.fit_transform(X)
    
    results = {}
    
    # A. KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(X_preprocessed)
    km_labels = kmeans.labels_
    
    # Metrics
    if len(np.unique(km_labels)) > 1:
        km_sil = float(silhouette_score(X_preprocessed, km_labels))
        km_db = float(davies_bouldin_score(X_preprocessed, km_labels))
        km_ch = float(calinski_harabasz_score(X_preprocessed, km_labels))
    else:
        km_sil, km_db, km_ch = 0.0, 0.0, 0.0
        
    results['K-Means'] = {
        'model': kmeans,
        'preprocessor': preprocessor,
        'labels': km_labels,
        'metrics': {
            'Silhouette Score': km_sil,
            'Davies-Bouldin Index': km_db,
            'Calinski-Harabasz Index': km_ch,
            'Inertia (Within Cluster Sum)': float(kmeans.inertia_)
        },
        'X_preprocessed': X_preprocessed,
        'valid_features': valid_features
    }
    
    # B. Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gmm_labels = gmm.fit_predict(X_preprocessed)
    
    if len(np.unique(gmm_labels)) > 1:
        gmm_sil = float(silhouette_score(X_preprocessed, gmm_labels))
        gmm_db = float(davies_bouldin_score(X_preprocessed, gmm_labels))
        gmm_ch = float(calinski_harabasz_score(X_preprocessed, gmm_labels))
    else:
        gmm_sil, gmm_db, gmm_ch = 0.0, 0.0, 0.0
        
    results['Gaussian Mixture Model'] = {
        'model': gmm,
        'preprocessor': preprocessor,
        'labels': gmm_labels,
        'metrics': {
            'Silhouette Score': gmm_sil,
            'Davies-Bouldin Index': gmm_db,
            'Calinski-Harabasz Index': gmm_ch,
            'BIC (Bayesian Info Criterion)': float(gmm.bic(X_preprocessed))
        },
        'X_preprocessed': X_preprocessed,
        'valid_features': valid_features
    }
    
    return results

def get_cluster_profiles(df, feature_cols, labels):
    """
    Creates a summary of cluster characteristics.
    """
    df_copy = df[feature_cols].copy()
    df_copy['Cluster'] = [f"Cluster {l}" for l in labels]
    
    profiles = []
    total_size = len(df_copy)
    
    for cluster_name in sorted(df_copy['Cluster'].unique()):
        clust_df = df_copy[df_copy['Cluster'] == cluster_name]
        cnt = len(clust_df)
        pct = (cnt / total_size) * 100 if total_size > 0 else 0
        
        profile = {
            'Cluster': cluster_name,
            'Size (Rows)': cnt,
            'Percentage (%)': round(pct, 2)
        }
        
        # Profile features
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df_copy[col]) and df_copy[col].nunique() > 10:
                profile[f"{col} (Mean)"] = round(float(clust_df[col].mean()), 2)
            else:
                # Mode
                mode_vals = clust_df[col].mode()
                profile[f"{col} (Top)"] = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else 'N/A'
                
        profiles.append(profile)
        
    return pd.DataFrame(profiles)

def generate_clustering_diagnostic_plots(best_model_name, X_preprocessed, labels, n_clusters):
    """
    Generates PCA 2D Scatter plot and Elbow/Silhouette plot.
    """
    plots = {}
    
    # 1. PCA 2D Scatter Plot
    pca = PCA(n_components=2, random_state=42)
    try:
        # If dataset has fewer columns than 2, pad with zeros
        X_dense = X_preprocessed.toarray() if hasattr(X_preprocessed, 'toarray') else X_preprocessed
        if X_dense.shape[1] < 2:
            padding = np.zeros((X_dense.shape[0], 2 - X_dense.shape[1]))
            X_dense = np.hstack((X_dense, padding))
            
        X_pca = pca.fit_transform(X_dense)
        evr = pca.explained_variance_ratio_
        
        df_pca = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'Cluster': [f"Cluster {l}" for l in labels]
        })
        
        fig_pca = px.scatter(
            df_pca, x='PCA1', y='PCA2', color='Cluster',
            title=f"2D Cluster Projection (PCA) - {best_model_name}",
            labels={
                'PCA1': f'PCA Component 1 ({evr[0]:.1%})',
                'PCA2': f'PCA Component 2 ({evr[1]:.1%})'
            },
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig_pca.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#E2E8F0"),
            title_font=dict(color="#F8FAFC", size=16)
        )
        fig_pca.update_xaxes(showgrid=True, gridcolor='#334155')
        fig_pca.update_yaxes(showgrid=True, gridcolor='#334155')
        
        plots['pca_scatter'] = fig_pca
    except Exception:
        pass
        
    # 2. Silhouette Plot for K values (2 to 8)
    k_vals = list(range(2, min(9, len(X_preprocessed))))
    if len(k_vals) >= 2:
        try:
            silhouettes = []
            for k in k_vals:
                km = KMeans(n_clusters=k, random_state=42, n_init='auto')
                km.fit(X_preprocessed)
                silhouettes.append(silhouette_score(X_preprocessed, km.labels_))
                
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=k_vals, y=silhouettes, mode='lines+markers',
                name='Silhouette Score', line=dict(color='#8B5CF6', width=3)
            ))
            
            fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="#EF4444", 
                                 annotation_text=f"K={n_clusters}", annotation_position="top")
            
            fig_elbow.update_layout(
                title="Optimal Cluster Evaluation (K-Means)",
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Silhouette Score (Higher is Better)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E2E8F0"),
                title_font=dict(color="#F8FAFC", size=16)
            )
            fig_elbow.update_xaxes(showgrid=True, gridcolor='#334155')
            fig_elbow.update_yaxes(showgrid=True, gridcolor='#334155')
            plots['elbow_curve'] = fig_elbow
        except Exception:
            pass
            
    return plots
