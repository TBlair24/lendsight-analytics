# =============================================================
# LendSight Analytics — ClickHouse Data Loader
# Reads CSVs and loads them into ClickHouse tables
# =============================================================

from clickhouse_driver import Client
import pandas as pd
import os

DATA_DIR = r"C:\Projects\lendsight-analytics\data"

# Connect to ClickHouse (running in WSL)
# Find your WSL IP by running `hostname -I` in Ubuntu terminal
# Or use 'localhost' if it works
client = Client(
    host='localhost',
    port=9000,
    database='lendsight',
    user='default',
    password=''
)

def load_table(table_name: str, csv_file: str):
    print(f"  → Loading {table_name}...")
    df = pd.read_csv(f"{DATA_DIR}/{csv_file}")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Detect date columns: columns containing "date", ending with "_month", or "month_start"
    date_cols = [c for c in df.columns if 
                 'date' in c.lower() or
                 c.lower().endswith('_month') or
                 c.lower() == 'month_start']
    print(f"    Date columns: {date_cols}")
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Fill nulls for non-date columns with 0
    for col in df.columns:
        if col not in date_cols:
            df[col] = df[col].fillna(0)

    # Build rows with proper date objects
    rows = []
    for idx, row in df.iterrows():
        row_values = []
        for col_name, val in row.items():
            if col_name in date_cols:
                # Convert to date object (not string)
                if pd.isna(val):
                    row_values.append(None)
                else:
                    # val is a Timestamp, convert to date
                    date_obj = val.date()
                    row_values.append(date_obj)
            else:
                row_values.append(val)
        rows.append(tuple(row_values))
    
    client.execute(f"INSERT INTO {table_name} VALUES", rows)
    print(f"     ✓ {len(df):,} rows loaded into {table_name}")

print("📦 Loading data into ClickHouse...")

load_table("dim_date",       "dim_date.csv")
load_table("dim_customer",   "dim_customer.csv")
load_table("fact_loans",     "fact_loans.csv")
load_table("fact_payments",  "fact_payments.csv")
load_table("fact_ml_scores", "fact_ml_scores.csv")

# Quick validation
for table in ["dim_date","dim_customer","fact_loans","fact_payments","fact_ml_scores"]:
    count = client.execute(f"SELECT count() FROM {table}")[0][0]
    print(f"  ✓ {table}: {count:,} rows")

print("\n✅ All tables loaded successfully!")