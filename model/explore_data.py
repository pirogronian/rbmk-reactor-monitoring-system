"""
Data exploration script to understand RBMK reactor monitoring data structure.
Run this first to understand your data before training the model.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

def explore_data(filepath):
    """
    Load and analyze the parquet file to understand data structure.

    Returns a summary of:
    - Data shape and columns
    - Data types and missing values
    - Statistical properties (mean, std, min, max)
    - Time-based information if available
    """
    print("=" * 80)
    print("RBMK REACTOR DATA EXPLORATION")
    print("=" * 80)

    # Load data
    df = pd.read_parquet(filepath)

    print(f"\n📊 DATA SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📋 COLUMNS ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col} ({df[col].dtype})")

    print(f"\n❓ MISSING VALUES:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values detected")
    else:
        print(missing[missing > 0])

    print(f"\n📈 STATISTICAL SUMMARY:")
    print(df.describe().to_string())

    print(f"\n🔢 NUMERIC COLUMNS ONLY:")
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"  Total: {len(numeric_df.columns)}")
    print(f"  Columns: {numeric_df.columns.tolist()}")

    # Check for timestamp
    if 'time' in df.columns or 'timestamp' in df.columns:
        time_col = 'time' if 'time' in df.columns else 'timestamp'
        print(f"\n⏱️  TIME INFORMATION (Column: {time_col}):")
        print(f"  First timestamp: {df[time_col].min()}")
        print(f"  Last timestamp: {df[time_col].max()}")

    print("\n" + "=" * 80)
    print("Sample data (first 5 rows):")
    print("=" * 80)
    print(df.head().to_string())

    return df

if __name__ == "__main__":
    data_path = Path("data/Influx_RBML_data.parquet")
    if data_path.exists():
        df = explore_data(data_path)
    else:
        print(f"❌ Data file not found at {data_path}")
        print("   Make sure the parquet file is in the data folder")
