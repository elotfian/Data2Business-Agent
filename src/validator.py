import numpy as np
import pandas as pd

def run_validation_checks(df, target_col, feature_cols, task_type, train_score, test_score, best_model_metrics=None, cluster_labels=None):
    """
    Runs automated checks on the dataset and the trained model/clusters.
    
    Returns a list of dicts:
    - { 'status': 'INFO'|'WARNING'|'CRITICAL', 'check_name': str, 'message': str, 'details': str }
    """
    logs = []
    num_rows = len(df)
    
    # 1. Sample size check
    if num_rows < 100:
        logs.append({
            'status': 'WARNING',
            'check_name': 'Small Sample Size',
            'message': f"Dataset has only {num_rows} rows.",
            'details': "Machine learning models require sufficient data to generalize. Results may be unstable or overfitted. Consider gathering more data or using simpler statistical models."
        })
    else:
        logs.append({
            'status': 'INFO',
            'check_name': 'Sample Size',
            'message': f"Dataset size is sufficient ({num_rows} rows).",
            'details': "The row count is adequate for running baseline machine learning algorithms."
        })
        
    # 2. Missing values check
    missing_pcts = df[feature_cols].isnull().mean() * 100
    high_missing = missing_pcts[missing_pcts > 20]
    crit_missing = missing_pcts[missing_pcts > 50]
    
    if not crit_missing.empty:
        logs.append({
            'status': 'CRITICAL',
            'check_name': 'Extreme Missing Data',
            'message': f"Columns {list(crit_missing.index)} have over 50% missing values.",
            'details': "Imputing more than 50% of a column introduces significant synthetic bias. Consider removing these columns from your features."
        })
    elif not high_missing.empty:
        logs.append({
            'status': 'WARNING',
            'check_name': 'High Missing Data',
            'message': f"Columns {list(high_missing.index)} have 20-50% missing values.",
            'details': "These columns contain substantial missing data. The agent will impute them using median/mode, but this might introduce noise."
        })
    else:
        logs.append({
            'status': 'INFO',
            'check_name': 'Missing Data Check',
            'message': "No features have high missingness (< 20%).",
            'details': "All selected feature columns contain complete or mostly complete data."
        })
        
    # 3. Target Leakage (Only for Classification/Regression)
    if task_type in ['Classification', 'Regression'] and target_col is not None:
        leakage_detected = False
        if pd.api.types.is_numeric_dtype(df[target_col]):
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    correlation = df[col].corr(df[target_col])
                    if abs(correlation) > 0.95:
                        logs.append({
                            'status': 'CRITICAL',
                            'check_name': 'Potential Target Leakage',
                            'message': f"Feature '{col}' has a correlation of {correlation:.3f} with target '{target_col}'.",
                            'details': "An extremely high correlation suggests that this feature contains information about the target that wouldn't be available at prediction time. Using it will result in an over-optimistic model that fails in production."
                        })
                        leakage_detected = True
                        
        # Target Leakage based on model performance
        if not leakage_detected and test_score is not None:
            if test_score > 0.999:
                logs.append({
                    'status': 'WARNING',
                    'check_name': 'Suspiciously High Model Score',
                    'message': f"Model achieved a score of {test_score:.4f}.",
                    'details': "Perfect or near-perfect performance often indicates target leakage, where a feature directly exposes the target outcome. Review the selected features."
                })
                
    # 4. Class Imbalance (Classification only)
    if task_type == 'Classification' and target_col is not None:
        class_counts = df[target_col].value_counts()
        if len(class_counts) >= 2:
            majority_class = class_counts.index[0]
            majority_count = class_counts.iloc[0]
            minority_count = class_counts.iloc[-1]
            imbalance_ratio = majority_count / minority_count
            
            minority_pct = (minority_count / num_rows) * 100
            
            if minority_pct < 5.0:
                logs.append({
                    'status': 'CRITICAL',
                    'check_name': 'Severe Class Imbalance',
                    'message': f"The minority class represents only {minority_pct:.2f}% of the target.",
                    'details': f"Severe class imbalance (ratio {imbalance_ratio:.1f}:1). The model will likely predict the majority class ('{majority_class}') for almost all cases. Standard metrics like Accuracy will be misleading; focus on F1-Score or Precision/Recall."
                })
            elif minority_pct < 20.0:
                logs.append({
                    'status': 'WARNING',
                    'check_name': 'Moderate Class Imbalance',
                    'message': f"The minority class represents {minority_pct:.2f}% of the target.",
                    'details': f"Class imbalance ratio is {imbalance_ratio:.1f}:1. Model baseline might be biased. Stratified split and F1 evaluation have been automatically applied."
                })
            else:
                logs.append({
                    'status': 'INFO',
                    'check_name': 'Class Balance Check',
                    'message': f"Classes are reasonably balanced. Minority class is {minority_pct:.1f}%.",
                    'details': "Target classes have sufficient representation for standard training and evaluation."
                })
                
    # 5. Multicollinearity (Feature redundancy)
    high_corr_features = []
    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_features) > 1:
        corr_matrix = df[numeric_features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        for col in upper_tri.columns:
            high_corrs = upper_tri[col][upper_tri[col] > 0.85]
            for idx, val in high_corrs.items():
                high_corr_features.append((idx, col, val))
                
    if high_corr_features:
        details_str = "High correlation found between: " + ", ".join([f"({f1} & {f2}: r={v:.2f})" for f1, f2, v in high_corr_features])
        logs.append({
            'status': 'WARNING',
            'check_name': 'Multicollinearity (Feature Redundancy)',
            'message': f"Detected {len(high_corr_features)} pairs of highly correlated features.",
            'details': f"{details_str}. Highly correlated features are redundant and can destabilize linear models or make feature importance metrics less reliable."
        })
    else:
        logs.append({
            'status': 'INFO',
            'check_name': 'Multicollinearity Check',
            'message': "No high correlation detected between numeric features.",
            'details': "All numeric features have low mutual correlations (r < 0.85)."
        })
        
    # 6. Overfitting Risk (Classification/Regression only)
    if task_type in ['Classification', 'Regression'] and train_score is not None and test_score is not None:
        gap = train_score - test_score
        metric_name = "F1-Score" if task_type == 'Classification' else "R²"
        
        if gap > 0.15:
            logs.append({
                'status': 'WARNING',
                'check_name': 'Overfitting Risk',
                'message': f"Training {metric_name} is {train_score:.3f} while test score is {test_score:.3f} (gap = {gap:.3f}).",
                'details': "The model performs substantially better on training data than on unseen test data. This indicates overfitting. You may need to simplify the model, tune regularisation parameters, or gather more training samples."
            })
        else:
            logs.append({
                'status': 'INFO',
                'check_name': 'Generalization Check',
                'message': f"Training and test scores are close (gap = {gap:.3f}).",
                'details': f"The model shows good generalization. Train {metric_name}: {train_score:.3f}, Test {metric_name}: {test_score:.3f}."
            })
            
    # 7. High Cardinality Categorical Features
    high_cardinality = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            card = df[col].nunique()
            if card > 30:
                high_cardinality.append((col, card))
                
    if high_cardinality:
        details_str = ", ".join([f"'{c}' ({card} categories)" for c, card in high_cardinality])
        logs.append({
            'status': 'WARNING',
            'check_name': 'High Cardinality Categoricals',
            'message': f"Categorical features with very high cardinality detected: {details_str}.",
            'details': "One-hot encoding high-cardinality features causes an explosion of feature space (sparsity), which degrades performance. The agent will use Ordinal/Label encoding for these, but reducing categories or grouping them is recommended."
        })
        
    # 8. Clustering Specific Checks
    if task_type == 'Clustering' and best_model_metrics is not None:
        sil_score = best_model_metrics.get('Silhouette Score')
        if sil_score is not None:
            if sil_score < 0.25:
                logs.append({
                    'status': 'WARNING',
                    'check_name': 'Weak Cluster Structure (Low Silhouette Score)',
                    'message': f"The best clustering model has a Silhouette Score of {sil_score:.3f}.",
                    'details': "A Silhouette Score below 0.25 indicates that the clusters overlap significantly or that the data lacks clear group boundaries. Consider using different features, adjusting the cluster count K, or scaling numeric columns."
                })
            elif sil_score >= 0.50:
                logs.append({
                    'status': 'INFO',
                    'check_name': 'Strong Cluster Structure',
                    'message': f"Clustering model achieved a Silhouette Score of {sil_score:.3f}.",
                    'details': "A score above 0.50 suggests well-defined, dense, and well-separated data segments."
                })
            else:
                logs.append({
                    'status': 'INFO',
                    'check_name': 'Moderate Cluster Structure',
                    'message': f"Clustering model achieved a Silhouette Score of {sil_score:.3f}.",
                    'details': "A score between 0.25 and 0.50 indicates reasonable cluster structure, though boundaries are somewhat soft."
                })
                
        if cluster_labels is not None:
            unique, counts = np.unique(cluster_labels, return_counts=True)
            sizes = dict(zip(unique, counts))
            
            for cluster_id, size in sizes.items():
                pct = (size / num_rows) * 100
                if pct < 5.0:
                    logs.append({
                        'status': 'WARNING',
                        'check_name': 'Small Cluster Segment',
                        'message': f"Cluster {cluster_id} contains only {size} rows ({pct:.2f}% of the dataset).",
                        'details': "Clusters representing less than 5% of the data may be capturing noise, anomalies, or very small sub-populations. Verify if this segment is actionable for business decisions."
                    })
                elif pct > 80.0:
                    logs.append({
                        'status': 'WARNING',
                        'check_name': 'Dominated Cluster Size',
                        'message': f"Cluster {cluster_id} contains {size} rows ({pct:.2f}% of the dataset).",
                        'details': "A single cluster dominates over 80% of the dataset. This indicates that the clustering model did not partition the data evenly. Try choosing other features or increasing K."
                    })
                    
    return logs
