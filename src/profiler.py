import os
import sqlite3
import pandas as pd
import numpy as np

def load_dataset(file_path_or_buffer, file_type=None, table_name=None):
    """
    Loads a dataset from various formats: CSV, Excel, JSON, Parquet, SQLite.
    
    Parameters:
    - file_path_or_buffer: Path to file, or a file-like object (buffer).
    - file_type: 'csv', 'excel', 'json', 'parquet', or 'sqlite'. If None, infers from extension.
    - table_name: Required for SQLite, specifies which table to load.
    
    Returns:
    - df: pandas DataFrame, or connection object for SQLite if no table specified.
    """
    if file_type is None and isinstance(file_path_or_buffer, str):
        _, ext = os.path.splitext(file_path_or_buffer.lower())
        if ext == '.csv':
            file_type = 'csv'
        elif ext in ['.xlsx', '.xls']:
            file_type = 'excel'
        elif ext == '.json':
            file_type = 'json'
        elif ext in ['.parquet', '.pq']:
            file_type = 'parquet'
        elif ext in ['.db', '.sqlite', '.sqlite3']:
            file_type = 'sqlite'
        else:
            raise ValueError(f"Could not infer file type for extension {ext}")

    if file_type == 'csv':
        return pd.read_csv(file_path_or_buffer)
    elif file_type == 'excel':
        return pd.read_excel(file_path_or_buffer)
    elif file_type == 'json':
        return pd.read_json(file_path_or_buffer)
    elif file_type == 'parquet':
        return pd.read_parquet(file_path_or_buffer)
    elif file_type == 'sqlite':
        if isinstance(file_path_or_buffer, str):
            conn = sqlite3.connect(file_path_or_buffer)
        else:
            # For stream-based buffers, SQLite requires writing to a temp file
            # We handle this in streamlit by writing the buffer to a temp file and passing the path.
            raise ValueError("SQLite loading requires a file path string.")
        
        if table_name:
            df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
            conn.close()
            return df
        else:
            return conn
    else:
        # Fallback load as CSV
        try:
            return pd.read_csv(file_path_or_buffer)
        except Exception as e:
            raise ValueError(f"Unsupported file type: {file_type}. Failed to load: {str(e)}")

def get_sqlite_tables(db_path):
    """Returns a list of table names in a SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def infer_column_types(df):
    """
    Infers the semantic type of each column in the DataFrame.
    
    Types returned: 'Numeric', 'Categorical', 'Datetime', 'ID', 'Text'
    """
    inferred_types = {}
    num_rows = len(df)
    
    for col in df.columns:
        # Drop missing values to inspect contents
        non_null_series = df[col].dropna()
        if len(non_null_series) == 0:
            inferred_types[col] = 'Categorical' # Fallback
            continue
            
        col_type = df[col].dtype
        num_unique = non_null_series.nunique()
        unique_ratio = num_unique / num_rows if num_rows > 0 else 0
        
        # Check for ID patterns
        col_name_lower = str(col).lower()
        is_id_name = any(x in col_name_lower for x in ['id', 'key', 'code', 'uuid', 'pk', 'number', 'no'])
        
        # 1. ID check: Only integer or string types can be IDs. String columns with long values are Text, not IDs.
        is_id_type = pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or col_type == 'object'
        avg_str_len = non_null_series.astype(str).str.len().mean() if (col_type == 'object' or pd.api.types.is_string_dtype(df[col])) else 0
        if is_id_type and avg_str_len <= 20 and ((num_unique == num_rows) or (is_id_name and unique_ratio > 0.8 and num_unique > 5)):
            inferred_types[col] = 'ID'
            continue
            
        # 2. Datetime check
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            inferred_types[col] = 'Datetime'
            continue
            
        # Try converting object columns with date-like names to datetime
        if col_type == 'object' or pd.api.types.is_string_dtype(df[col]):
            is_date_name = any(x in col_name_lower for x in ['date', 'time', 'timestamp', 'created', 'updated', 'year', 'month'])
            if is_date_name:
                try:
                    pd.to_datetime(non_null_series.head(50), errors='raise')
                    inferred_types[col] = 'Datetime'
                    continue
                except:
                    pass
        
        # 3. Categorical check
        # Explicit categorical, boolean, or low cardinality columns
        if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            inferred_types[col] = 'Categorical'
            continue
            
        if num_unique <= 15:
            inferred_types[col] = 'Categorical'
            continue
            
        # 4. Numeric check
        if pd.api.types.is_numeric_dtype(df[col]):
            # If it is numeric but has very few values (already covered), otherwise Numeric
            inferred_types[col] = 'Numeric'
            continue
            
        # 5. Text vs Categorical check
        # High cardinality object/string columns
        if col_type == 'object' or pd.api.types.is_string_dtype(df[col]):
            # Check average length of strings
            avg_len = non_null_series.astype(str).str.len().mean()
            if avg_len > 30 and unique_ratio > 0.5:
                inferred_types[col] = 'Text'
            else:
                inferred_types[col] = 'Categorical'
            continue
            
        # Fallback
        inferred_types[col] = 'Categorical'
        
    return inferred_types

def profile_dataset(df):
    """
    Profiles a DataFrame and returns detailed statistics.
    """
    num_rows, num_cols = df.shape
    num_duplicates = df.duplicated().sum()
    
    col_types = infer_column_types(df)
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / num_rows) * 100
    
    columns_summary = []
    
    for col in df.columns:
        col_type = col_types[col]
        missing_cnt = int(missing_counts[col])
        missing_pct = float(missing_percentages[col])
        num_unique = int(df[col].nunique())
        
        summary = {
            'column_name': col,
            'data_type': str(df[col].dtype),
            'semantic_type': col_type,
            'missing_count': missing_cnt,
            'missing_percentage': round(missing_pct, 2),
            'unique_count': num_unique,
            'is_nullable': missing_cnt > 0
        }
        
        non_null_series = df[col].dropna()
        if len(non_null_series) > 0:
            if col_type == 'Numeric':
                summary['stats'] = {
                    'min': float(non_null_series.min()),
                    'max': float(non_null_series.max()),
                    'mean': float(non_null_series.mean()),
                    'median': float(non_null_series.median()),
                    'std': float(non_null_series.std()) if len(non_null_series) > 1 else 0.0
                }
            elif col_type == 'Categorical':
                top_values = non_null_series.value_counts().head(5)
                summary['stats'] = {
                    'top_values': [{ 'value': str(val), 'count': int(cnt), 'percentage': round((cnt/len(non_null_series))*100, 2) } for val, cnt in top_values.items()]
                }
            elif col_type == 'Datetime':
                # Convert to datetime for stats
                dt_series = pd.to_datetime(non_null_series, errors='coerce')
                dt_series = dt_series.dropna()
                if len(dt_series) > 0:
                    summary['stats'] = {
                        'min': str(dt_series.min()),
                        'max': str(dt_series.max()),
                        'range_days': int((dt_series.max() - dt_series.min()).days)
                    }
                else:
                    summary['stats'] = {}
            else:
                summary['stats'] = {}
        else:
            summary['stats'] = {}
            
        columns_summary.append(summary)
        
    return {
        'num_rows': num_rows,
        'num_cols': num_cols,
        'num_duplicates': int(num_duplicates),
        'columns': columns_summary
    }

def recommend_kpis_and_targets(profile):
    """
    Recommends possible key performance indicators (KPIs) and machine learning target columns.
    """
    recommended_targets = []
    recommended_kpis = []
    
    for col_info in profile['columns']:
        col_name = col_info['column_name']
        sem_type = col_info['semantic_type']
        missing_pct = col_info['missing_percentage']
        unique_cnt = col_info['unique_count']
        
        # Don't recommend target/KPI if missing percentage is extremely high
        if missing_pct > 50:
            continue
            
        # Target Candidates: Categorical (2-10 unique) or Numeric (not high missing, not ID, not text)
        col_name_lower = col_name.lower()
        
        # Scoring function for target recommendation
        target_score = 0
        kpi_score = 0
        
        # Text clues
        target_keywords = ['target', 'label', 'class', 'status', 'churn', 'default', 'y', 'outcome', 'purchased', 'clicked', 'sold', 'revenue', 'price', 'profit', 'sales']
        kpi_keywords = ['revenue', 'sales', 'profit', 'cost', 'spend', 'churn', 'conversion', 'satisfaction', 'nps', 'score', 'margin', 'amount', 'total']
        
        if any(kw in col_name_lower for kw in target_keywords):
            target_score += 3
        if any(kw in col_name_lower for kw in kpi_keywords):
            kpi_score += 3
            
        if sem_type == 'Categorical':
            if unique_cnt == 2:
                target_score += 3 # Binary classification is very common
            elif 2 < unique_cnt <= 10:
                target_score += 2 # Multi-class classification
            kpi_score += 1
        elif sem_type == 'Numeric':
            target_score += 1 # Regression target
            kpi_score += 2 # Sums, averages, ranges
            
        # If it is an ID, text, or constant, exclude
        if sem_type in ['ID', 'Text'] or unique_cnt <= 1:
            target_score = 0
            kpi_score = 0
            
        if target_score >= 2:
            recommended_targets.append({
                'column': col_name,
                'type': sem_type,
                'task_suggestion': 'Classification' if sem_type == 'Categorical' else 'Regression',
                'score': target_score
            })
            
        if kpi_score >= 2:
            recommended_kpis.append({
                'column': col_name,
                'type': sem_type,
                'score': kpi_score
            })
            
    # Sort recommendations by score descending
    recommended_targets.sort(key=lambda x: x['score'], reverse=True)
    recommended_kpis.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'recommended_targets': recommended_targets,
        'recommended_kpis': recommended_kpis
    }
