import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {'user_id', 'date', 'metric', 'value'}

def validate_schema(df: pd.DataFrame):
    """
    Checks if the dataframe has the required columns.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Basic type check - ensure value is numeric
    if not pd.api.types.is_numeric_dtype(df['value']):
        # Try to coerce
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        if df['value'].isna().any():
             raise ValueError("Column 'value' contains non-numeric data")

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the dataframe:
    - Converts date to datetime
    - Standardizes metric names
    """
    df = df.copy()
    
    # Parse dates
    # Try to infer format, handle errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Normalize metric names (lowercase, strip)
    df['metric'] = df['metric'].astype(str).str.lower().str.strip()
    df['metric'] = df['metric'].replace({
        'heart rate': 'heart_rate',
        'bpm': 'heart_rate',
        'hr': 'heart_rate',
        'water intake': 'water'
    })
    
    return df

def aggregate_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates data per day and metric.
    """
    # First, ensure we are working with just dates for grouping
    df['day'] = df['date'].dt.date
    
    # Since we're converting from wide format, we already have one value per day per metric
    # Just group by day and metric and take the first value (should be only one anyway)
    result = df.groupby(['day', 'metric'])['value'].first().reset_index()
    
    return result

def detect_anomalies(series: pd.Series, window=7, k=3):
    """
    Detects anomalies using rolling mean and std.
    """
    rolling_mean = series.rolling(window, min_periods=1).mean()
    rolling_std = series.rolling(window, min_periods=1).std().fillna(0.0)
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, 1e-9)
    
    z = (series - rolling_mean).abs() / rolling_std
    
    anomalies = z[z > k]
    
    results = []
    for idx in anomalies.index:
        results.append({
            "date": str(idx),
            "value": float(series.loc[idx]),
            "z_score": float(z.loc[idx]),
            "reason": f"Deviation > {k} sigma (val={series.loc[idx]:.1f}, mean={rolling_mean.loc[idx]:.1f})"
        })
        
    # Domain specific rules fallback
    # e.g. if sample size is small, or just absolute thresholds
    # We can add those here if needed, but z-score is good for now.
    
    return results

def convert_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide format CSV to long format expected by processor
    """
    records = []
    for _, row in df.iterrows():
        date = row['date']
        records.extend([
            {'user_id': 'user1', 'date': date, 'metric': 'heart_rate', 'value': float(row['heart_rate'])},
            {'user_id': 'user1', 'date': date, 'metric': 'steps', 'value': float(row['steps'])},
            {'user_id': 'user1', 'date': date, 'metric': 'sleep', 'value': float(row['sleep_hours'])},
            {'user_id': 'user1', 'date': date, 'metric': 'water', 'value': float(row['water_liters'])},
            {'user_id': 'user1', 'date': date, 'metric': 'calories', 'value': float(row['calories_burned'])}
        ])
    return pd.DataFrame(records)

def get_trends_and_insights(df: pd.DataFrame):
    """
    Main processing function to generate summary, trends, and anomalies.
    """
    # Handle wide format CSV
    if 'heart_rate' in df.columns and 'steps' in df.columns:
        df = convert_wide_to_long(df)
    
    validate_schema(df)
    df = normalize(df)
    
    # Calculate user count before aggregation
    unique_users = df['user_id'].nunique()
    
    daily_df = aggregate_per_day(df)
    
    summary = {
        "total_users": unique_users
    }
    trends = []
    anomalies = []
    
    # Process each metric
    for metric in daily_df['metric'].unique():
        metric_data = daily_df[daily_df['metric'] == metric].set_index('day')['value'].sort_index()
        
        # 1. Summary (Last 7 days avg)
        last_7 = metric_data.tail(7)
        if not last_7.empty:
            avg_7d = last_7.mean()
            summary[f"{metric}_avg_7d"] = round(avg_7d, 2)
        
        # 2. Anomalies
        metric_anomalies = detect_anomalies(metric_data)
        for a in metric_anomalies:
            a['metric'] = metric
            anomalies.append(a)
            
        # 3. Trends (Compare first vs last day, or recent vs previous)
        if len(metric_data) >= 2:
            if len(metric_data) >= 6:
                # Use original logic for longer datasets
                recent = metric_data.iloc[-3:].mean()
                prev = metric_data.iloc[-6:-3].mean()
            else:
                # For shorter datasets, compare last vs first
                recent = metric_data.iloc[-1]
                prev = metric_data.iloc[0]
            
            if prev > 0:
                change = ((recent - prev) / prev) * 100
                direction = "up" if change > 5 else "down" if change < -5 else "stable"
                trends.append({
                    "metric": metric,
                    "trend": direction,
                    "change_percent": round(change, 1)
                })
    
    # Domain specific checks (as requested)
    # Resting HR > 100
    hr_data = daily_df[daily_df['metric'] == 'heart_rate']
    for _, row in hr_data.iterrows():
        if row['value'] > 100:
            anomalies.append({
                "date": str(row['day']),
                "metric": "heart_rate",
                "value": row['value'],
                "reason": "Urgent: Resting HR > 100"
            })
            
    # Sleep < 4 hours
    sleep_data = daily_df[daily_df['metric'] == 'sleep']
    for _, row in sleep_data.iterrows():
        if row['value'] < 4:
             anomalies.append({
                "date": str(row['day']),
                "metric": "sleep",
                "value": row['value'],
                "reason": "Fatigue warning: Sleep < 4h"
            })

    # Ensure all values are properly converted to standard Python types
    timeseries_data = daily_df.to_dict(orient='records')
    for record in timeseries_data:
        record['value'] = float(record['value'])
        record['day'] = str(record['day'])
    
    return {
        "summary": summary,
        "trends": trends,
        "anomalies": anomalies,
        "timeseries": timeseries_data # For frontend graphs
    }
