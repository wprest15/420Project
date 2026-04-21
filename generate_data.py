"""
Generate a synthetic financial fraud dataset for testing.
Run: python generate_data.py
Produces: account_data.csv  (the default path skeleton.py expects)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000
FRAUD_RATE = 0.08  # 8% fraud — realistic imbalance

n_fraud = int(N * FRAUD_RATE)
n_legit = N - n_fraud

def make_legit(n):
    return pd.DataFrame({
        'revenue_growth':        np.random.normal(0.05, 0.10, n),
        'accounts_receivable':   np.random.normal(0.30, 0.08, n),
        'gross_margin':          np.random.normal(0.40, 0.10, n),
        'debt_to_equity':        np.random.normal(0.50, 0.20, n),
        'asset_turnover':        np.random.normal(1.00, 0.25, n),
        'current_ratio':         np.random.normal(2.00, 0.50, n),
        'operating_cash_flow':   np.random.normal(0.10, 0.05, n),
        'inventory_days':        np.random.normal(45.0, 10.0, n),
        'days_sales_outstanding':np.random.normal(35.0, 8.0,  n),
        'return_on_assets':      np.random.normal(0.08, 0.04, n),
    })

def make_fraud(n):
    return pd.DataFrame({
        'revenue_growth':        np.random.normal(0.25, 0.15, n),   # inflated
        'accounts_receivable':   np.random.normal(0.55, 0.10, n),   # inflated
        'gross_margin':          np.random.normal(0.55, 0.12, n),   # inflated
        'debt_to_equity':        np.random.normal(1.20, 0.40, n),   # high leverage
        'asset_turnover':        np.random.normal(0.70, 0.20, n),   # low
        'current_ratio':         np.random.normal(1.20, 0.30, n),   # tight liquidity
        'operating_cash_flow':   np.random.normal(0.02, 0.04, n),   # low vs. reported profit
        'inventory_days':        np.random.normal(80.0, 20.0, n),   # slow-moving
        'days_sales_outstanding':np.random.normal(65.0, 15.0, n),   # slow collections
        'return_on_assets':      np.random.normal(0.18, 0.06, n),   # suspiciously high
    })

legit = make_legit(n_legit)
legit['fraud_label'] = 0

fraud = make_fraud(n_fraud)
fraud['fraud_label'] = 1

df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('account_data.csv', index=False)

print(f"Saved account_data.csv  ({N} rows, {n_fraud} fraud [{FRAUD_RATE*100:.0f}%])")
print(f"Columns: {list(df.columns)}")
