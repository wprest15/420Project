"""
Generates three synthetic fraud datasets with different characteristics.
Run: python generate_datasets.py

Produces:
  financial_easy.csv   - financial fraud, clear signal (good baseline)
  financial_hard.csv   - financial fraud, noisy/subtle signal (harder to detect)
  insurance_fraud.csv  - completely different domain (insurance claims)
"""
import numpy as np
import pandas as pd

np.random.seed(42)


def save(df, path):
    df.to_csv(path, index=False)
    n_fraud = df['fraud_label'].sum()
    pct = 100 * n_fraud / len(df)
    print(f"Saved {path}  ({len(df)} rows, {n_fraud} fraud [{pct:.1f}%])")


# ── Dataset 1: Financial fraud, clear signal ─────────────────────────────────
# Fraud companies have clearly inflated revenue and receivables.
# This should produce high-fitness rules quickly.

def financial_easy(n=2000, fraud_rate=0.08):
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    legit = pd.DataFrame({
        'revenue_growth':         np.random.normal(0.05, 0.08, n_legit),
        'accounts_receivable':    np.random.normal(0.30, 0.06, n_legit),
        'gross_margin':           np.random.normal(0.40, 0.08, n_legit),
        'debt_to_equity':         np.random.normal(0.50, 0.15, n_legit),
        'asset_turnover':         np.random.normal(1.00, 0.20, n_legit),
        'current_ratio':          np.random.normal(2.00, 0.40, n_legit),
        'operating_cash_flow':    np.random.normal(0.10, 0.04, n_legit),
        'inventory_days':         np.random.normal(45.0,  8.0, n_legit),
        'days_sales_outstanding': np.random.normal(35.0,  6.0, n_legit),
        'return_on_assets':       np.random.normal(0.08, 0.03, n_legit),
        'fraud_label': 0,
    })

    fraud = pd.DataFrame({
        'revenue_growth':         np.random.normal(0.30, 0.08, n_fraud),  # very inflated
        'accounts_receivable':    np.random.normal(0.60, 0.07, n_fraud),  # very inflated
        'gross_margin':           np.random.normal(0.60, 0.08, n_fraud),
        'debt_to_equity':         np.random.normal(1.40, 0.25, n_fraud),
        'asset_turnover':         np.random.normal(0.60, 0.15, n_fraud),
        'current_ratio':          np.random.normal(1.10, 0.20, n_fraud),
        'operating_cash_flow':    np.random.normal(0.01, 0.03, n_fraud),  # low vs profit
        'inventory_days':         np.random.normal(85.0, 12.0, n_fraud),
        'days_sales_outstanding': np.random.normal(70.0, 10.0, n_fraud),
        'return_on_assets':       np.random.normal(0.22, 0.05, n_fraud),
        'fraud_label': 1,
    })

    df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    save(df, 'financial_easy.csv')


# ── Dataset 2: Financial fraud, hard/noisy signal ────────────────────────────
# Fraud companies look much more like legitimate ones — overlapping distributions.
# Expect lower fitness scores and more complex rules needed.

def financial_hard(n=2000, fraud_rate=0.08):
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    legit = pd.DataFrame({
        'revenue_growth':         np.random.normal(0.05, 0.12, n_legit),
        'accounts_receivable':    np.random.normal(0.30, 0.12, n_legit),
        'gross_margin':           np.random.normal(0.40, 0.12, n_legit),
        'debt_to_equity':         np.random.normal(0.50, 0.25, n_legit),
        'asset_turnover':         np.random.normal(1.00, 0.30, n_legit),
        'current_ratio':          np.random.normal(2.00, 0.60, n_legit),
        'operating_cash_flow':    np.random.normal(0.10, 0.07, n_legit),
        'inventory_days':         np.random.normal(45.0, 15.0, n_legit),
        'days_sales_outstanding': np.random.normal(35.0, 12.0, n_legit),
        'return_on_assets':       np.random.normal(0.08, 0.06, n_legit),
        'fraud_label': 0,
    })

    fraud = pd.DataFrame({
        'revenue_growth':         np.random.normal(0.12, 0.12, n_fraud),  # subtle
        'accounts_receivable':    np.random.normal(0.40, 0.12, n_fraud),  # subtle
        'gross_margin':           np.random.normal(0.48, 0.12, n_fraud),
        'debt_to_equity':         np.random.normal(0.80, 0.25, n_fraud),
        'asset_turnover':         np.random.normal(0.80, 0.30, n_fraud),
        'current_ratio':          np.random.normal(1.60, 0.50, n_fraud),
        'operating_cash_flow':    np.random.normal(0.05, 0.07, n_fraud),
        'inventory_days':         np.random.normal(58.0, 15.0, n_fraud),
        'days_sales_outstanding': np.random.normal(48.0, 12.0, n_fraud),
        'return_on_assets':       np.random.normal(0.13, 0.06, n_fraud),
        'fraud_label': 1,
    })

    df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    save(df, 'financial_hard.csv')


# ── Dataset 3: Insurance claim fraud ─────────────────────────────────────────
# Completely different domain — shows the system generalizes.
# Features: claim amounts, injury counts, policy age, etc.

def insurance_fraud(n=2000, fraud_rate=0.10):
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    legit = pd.DataFrame({
        'claim_amount':        np.random.normal(5000,  2000, n_legit),
        'num_injuries':        np.random.poisson(1.2,         n_legit).astype(float),
        'policy_age_years':    np.random.normal(6.0,   3.0,  n_legit),
        'prior_claims':        np.random.poisson(0.5,         n_legit).astype(float),
        'days_to_report':      np.random.normal(3.0,   2.0,  n_legit),
        'medical_cost_ratio':  np.random.normal(0.40,  0.10, n_legit),
        'witness_count':       np.random.poisson(1.5,         n_legit).astype(float),
        'attorney_involved':   np.random.binomial(1, 0.15,    n_legit).astype(float),
        'vehicle_age_years':   np.random.normal(5.0,   3.0,  n_legit),
        'deductible_ratio':    np.random.normal(0.10,  0.05, n_legit),
        'fraud_label': 0,
    })

    fraud = pd.DataFrame({
        'claim_amount':        np.random.normal(9000,  3000, n_fraud),   # inflated
        'num_injuries':        np.random.poisson(3.5,         n_fraud).astype(float),  # more injuries
        'policy_age_years':    np.random.normal(1.5,   1.0,  n_fraud),   # new policy
        'prior_claims':        np.random.poisson(2.5,         n_fraud).astype(float),  # repeat claims
        'days_to_report':      np.random.normal(1.0,   0.8,  n_fraud),   # reported fast
        'medical_cost_ratio':  np.random.normal(0.70,  0.12, n_fraud),   # high medical
        'witness_count':       np.random.poisson(0.3,         n_fraud).astype(float),  # few witnesses
        'attorney_involved':   np.random.binomial(1, 0.70,    n_fraud).astype(float),  # lawyer involved
        'vehicle_age_years':   np.random.normal(12.0,  4.0,  n_fraud),   # old vehicle
        'deductible_ratio':    np.random.normal(0.03,  0.02, n_fraud),   # low deductible
        'fraud_label': 1,
    })

    df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    save(df, 'insurance_fraud.csv')


financial_easy()
financial_hard()
insurance_fraud()
