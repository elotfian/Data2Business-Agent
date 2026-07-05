import os
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil

# Import modules from src
from src.profiler import load_dataset, profile_dataset, recommend_kpis_and_targets, infer_column_types
from src.ml_engine import preprocess_and_split, train_baselines, get_best_model, extract_feature_importances
from src.validator import run_validation_checks
from src.reporter import generate_markdown_report, generate_reproducible_code, create_static_plots, generate_pdf_report

class TestData2BusinessAgent(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Generate synthetic classification dataset
        np.random.seed(42)
        n_samples = 150
        
        # Features
        cls.num_feat = np.random.randn(n_samples) * 10 + 50
        cls.num_feat_correlated = cls.num_feat * 0.9 + np.random.randn(n_samples) # correlated
        cls.cat_feat_low = np.random.choice(['A', 'B', 'C'], size=n_samples)
        cls.cat_feat_high = [f"This is a long description of the occupation for client number {i} to simulate a text field." for i in range(n_samples)]
        cls.id_col = [f"ID_{i:04d}" for i in range(n_samples)]
        
        # Target for classification (binary)
        cls.target_class = np.random.choice([0, 1], size=n_samples)
        # Target for regression
        cls.target_reg = cls.num_feat * 2.5 + np.random.randn(n_samples) * 5
        
        cls.df_class = pd.DataFrame({
            'cust_id': cls.id_col,
            'age': cls.num_feat,
            'income': cls.num_feat_correlated,
            'region': cls.cat_feat_low,
            'occupation': cls.cat_feat_high,
            'churn': cls.target_class
        })
        
        cls.df_reg = pd.DataFrame({
            'product_id': cls.id_col,
            'marketing_spend': cls.num_feat,
            'sales_rep': cls.cat_feat_low,
            'sales': cls.target_reg
        })
        
        cls.tmp_dir = tempfile.mkdtemp()
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_profiler_types(self):
        # Test semantic type inference
        inferred = infer_column_types(self.df_class)
        self.assertEqual(inferred['cust_id'], 'ID')
        self.assertEqual(inferred['age'], 'Numeric')
        self.assertEqual(inferred['region'], 'Categorical')
        self.assertEqual(inferred['occupation'], 'Text') # due to high cardinality & string
        self.assertEqual(inferred['churn'], 'Categorical') # low cardinality integer (0 or 1)

    def test_profiler_profile(self):
        # Profile dataset
        profile = profile_dataset(self.df_class)
        self.assertEqual(profile['num_rows'], 150)
        self.assertEqual(profile['num_cols'], 6)
        self.assertEqual(profile['num_duplicates'], 0)
        
        # Check column names inside profile
        col_names = [c['column_name'] for c in profile['columns']]
        self.assertIn('age', col_names)
        self.assertIn('churn', col_names)
        
        # Target/KPI recommendation
        recs = recommend_kpis_and_targets(profile)
        self.assertTrue(len(recs['recommended_targets']) > 0)
        target_names = [r['column'] for r in recs['recommended_targets']]
        self.assertIn('churn', target_names)

    def test_ml_pipeline_classification(self):
        # Setup settings
        target_col = 'churn'
        feature_cols = ['age', 'income', 'region', 'occupation']
        task_type = 'Classification'
        
        # 1. Preprocess & split
        X_train, X_test, y_train, y_test, preprocessor, _ = preprocess_and_split(
            self.df_class, target_col, feature_cols, task_type
        )
        self.assertEqual(len(X_train), 120)
        self.assertEqual(len(X_test), 30)
        
        # 2. Train baseline models
        results = train_baselines(X_train, y_train, X_test, y_test, preprocessor, task_type)
        self.assertIn('Logistic Regression', results)
        self.assertIn('Random Forest', results)
        self.assertIn('Gradient Boosting', results)
        
        # 3. Get best model
        best_name, best_info = get_best_model(results, task_type)
        self.assertIsNotNone(best_name)
        self.assertIn('Accuracy', best_info['metrics'])
        self.assertIn('F1-Score', best_info['metrics'])
        
        # 4. Feature importances
        imp_df = extract_feature_importances(best_name, best_info['pipeline'], feature_cols)
        # Importances might be extracted for random forest/logistic, verify format if returned
        if imp_df is not None:
            self.assertIn('Feature', imp_df.columns)
            self.assertIn('Importance', imp_df.columns)

    def test_ml_pipeline_regression(self):
        # Setup settings
        target_col = 'sales'
        feature_cols = ['marketing_spend', 'sales_rep']
        task_type = 'Regression'
        
        # 1. Preprocess & split
        X_train, X_test, y_train, y_test, preprocessor, _ = preprocess_and_split(
            self.df_reg, target_col, feature_cols, task_type
        )
        self.assertEqual(len(X_train), 120)
        self.assertEqual(len(X_test), 30)
        
        # 2. Train baseline models
        results = train_baselines(X_train, y_train, X_test, y_test, preprocessor, task_type)
        self.assertIn('Ridge Regression', results)
        self.assertIn('Random Forest', results)
        
        # 3. Get best model
        best_name, best_info = get_best_model(results, task_type)
        self.assertIsNotNone(best_name)
        self.assertIn('R²', best_info['metrics'])
        self.assertIn('RMSE', best_info['metrics'])

    def test_validation_and_reporting(self):
        profile = profile_dataset(self.df_class)
        target_col = 'churn'
        feature_cols = ['age', 'income', 'region', 'occupation']
        task_type = 'Classification'
        
        # Train baseline
        X_train, X_test, y_train, y_test, preprocessor, _ = preprocess_and_split(
            self.df_class, target_col, feature_cols, task_type
        )
        results = train_baselines(X_train, y_train, X_test, y_test, preprocessor, task_type)
        best_name, best_info = get_best_model(results, task_type)
        
        # Run validations
        logs = run_validation_checks(
            self.df_class, target_col, feature_cols, task_type,
            train_score=0.85, test_score=best_info['metrics']['F1-Score'],
            best_model_metrics=best_info['metrics']
        )
        self.assertTrue(len(logs) > 0)
        
        # Ensure there is a Multicollinearity warning since age and income are highly correlated (0.9+)
        check_names = [log['check_name'] for log in logs]
        self.assertIn('Multicollinearity (Feature Redundancy)', check_names)
        
        # Generate reports
        markdown_rep = generate_markdown_report(
            profile, target_col, feature_cols, task_type,
            best_name, best_info['metrics'], logs,
            "Predict customer churn", "manager"
        )
        self.assertIn("Data2Business", markdown_rep)
        self.assertIn("Executive Summary", markdown_rep)
        
        reproducible_code = generate_reproducible_code(
            "dummy_dataset.csv", target_col, feature_cols, task_type, best_name
        )
        self.assertIn("REPRODUCIBLE DATA ANALYSIS", reproducible_code)
        self.assertIn("train_test_split", reproducible_code)
        
        # Static plots and PDF
        static_charts = create_static_plots(
            self.df_class, target_col, feature_cols, task_type,
            y_test, best_info['y_pred'], self.tmp_dir
        )
        self.assertIn('target_distribution', static_charts)
        self.assertTrue(os.path.exists(static_charts['target_distribution']))
        
        pdf_path = os.path.join(self.tmp_dir, "test_report.pdf")
        generate_pdf_report(
            profile, target_col, feature_cols, task_type,
            best_name, best_info['metrics'], logs,
            "Predict customer churn", "manager", static_charts, pdf_path
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertTrue(os.path.getsize(pdf_path) > 0)

    def test_load_dataset_buffers(self):
        import io
        
        # Test CSV buffer with .name
        csv_data = "a,b\n1,2\n3,4"
        csv_buffer = io.StringIO(csv_data)
        csv_buffer.name = "test.csv"
        df = load_dataset(csv_buffer)
        self.assertEqual(df.shape, (2, 2))
        
        # Test Excel buffer with .name
        excel_buffer = io.BytesIO()
        df_dummy = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df_dummy.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        excel_buffer.name = "test.xlsx"
        df_excel = load_dataset(excel_buffer)
        self.assertEqual(df_excel.shape, (2, 2))

    def test_missing_targets_drop(self):
        # Create dataset with NaNs in target
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': ['A', 'B', 'A', 'B', 'A'],
            'target': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        # 1. Verify preprocess_and_split drops rows (should split remaining 3 rows)
        X_train, X_test, y_train, y_test, preprocessor, _ = preprocess_and_split(
            df, 'target', ['feature1', 'feature2'], 'Regression'
        )
        self.assertEqual(len(X_train) + len(X_test), 3)
        self.assertFalse(y_train.isnull().any())
        self.assertFalse(y_test.isnull().any())
        
        # 2. Verify run_validation_checks flags target missingness
        logs = run_validation_checks(
            df, 'target', ['feature1', 'feature2'], 'Regression',
            train_score=0.8, test_score=0.7
        )
        check_names = [log['check_name'] for log in logs]
        self.assertIn('Missing Values in Target', check_names)

if __name__ == '__main__':
    unittest.main()
