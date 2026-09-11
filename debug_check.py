import pandas as pd
from pathlib import Path

DATA_FILE = Path(__file__).parent / 'data' / 'network_metrics.csv'
TARGET_METRICS = [
    "bandwidth_utilization_pct",
    "latency_ms",
    "packet_loss_pct",
    "jitter_ms",
]

df = pd.read_csv(DATA_FILE)
print('Total rows:', len(df))

# parse timestamp
try:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
except Exception as e:
    print('timestamp parse error', e)

# estimate median seconds
timestamp_diff = df['timestamp'].diff().dt.total_seconds()
positive_diffs = timestamp_diff[timestamp_diff > 0]
if positive_diffs.empty:
    median_seconds = 5.0
else:
    median_seconds = positive_diffs.median()
if pd.isna(median_seconds) or median_seconds <= 0:
    median_seconds = 5.0
horizon_seconds = 5 * 60
steps = int(round(horizon_seconds / median_seconds))
steps = max(1, steps)
print('median_seconds:', median_seconds)
print('steps:', steps)

# create future columns
for column in TARGET_METRICS:
    if column in df.columns:
        df[f'future_{column}'] = df[column].shift(-steps)

future_columns = [f for f in df.columns if f.startswith('future_')]
print('future_columns:', future_columns)
print('Rows before future filtering:', len(df))
for col in future_columns:
    print(col, 'valid=', df[col].notna().sum(), 'missing=', df[col].isna().sum())

# Drop rows where all future cols are NaN
before = len(df)
df2 = df.dropna(subset=future_columns, how='all')
print('Rows after dropna how=all:', len(df2), 'removed=', before - len(df2))

print('\nSample head:')
print(df.head())
print('\nSample tail:')
print(df.tail())
