# CS 420 Project

**Walton Prest, Peter Wraith**

Evolutionary Algorithm for Fraud Detection using ElasticNet + K-Means Clustering

---

## Requirements

Install dependencies before running anything:

```bash
pip install scikit-learn pandas numpy
```

---

## Dataset Options

### Option A — Synthetic Data (quick testing)

Generates a 2000-row financial fraud dataset with realistic features:

```bash
python generate_data.py
```

This creates `account_data.csv` in the project folder.

### Option B — Kaggle Credit Card Fraud Dataset

1. Download `creditcard.csv` from Kaggle and place it in the project folder
2. Prep it (renames the target column):

```bash
python -c "
import pandas as pd
df = pd.read_csv('creditcard.csv')
df = df.rename(columns={'Class': 'fraud_label'})
df.to_csv('creditcard_prepped.csv', index=False)
print(f'Done: {len(df)} rows, {df.fraud_label.sum()} fraud cases')
"
```

### Option C — Custom Dataset

Your CSV must have a column named `fraud_label` (0 = legitimate, 1 = fraud). All other numeric columns are used as features automatically.

---

## Running the Project

```bash
python skeleton.py --data_path <your_csv>
```

**Synthetic data (fast):**
```bash
python skeleton.py --data_path account_data.csv
```

**Credit card dataset (large — use reduced settings):**
```bash
python skeleton.py --data_path creditcard_prepped.csv --pop_size 50 --generations 20 --k_clusters 5 --patience 10
```

**Full run with all options:**
```bash
python skeleton.py \
  --data_path account_data.csv \
  --pop_size 100 \
  --generations 50 \
  --k_clusters 5 \
  --mutation_rate 0.3 \
  --tournament_size 3 \
  --patience 10
```

---

## Parameters

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `account_data.csv` | Path to CSV file (must have `fraud_label` column) |
| `--pop_size` | `100` | Number of rules per generation — larger = more diverse but slower |
| `--generations` | `50` | Maximum number of EA iterations |
| `--k_clusters` | `5` | Number of k-means diversity clusters |
| `--mutation_rate` | `0.3` | Probability a child rule gets randomly mutated |
| `--tournament_size` | `3` | Number of rules that compete in each selection round |
| `--n_elites` | auto | Top rules carried directly to next generation (default: 10% of pop) |
| `--patience` | `10` | Stop early if no improvement for this many generations |

---

## Output

Each generation prints best and average fitness. After evolution completes:

- **Top 5 fraud rules** printed in human-readable form (e.g. `debt_to_equity > 1.36`)
- **Performance metrics** for the best rule: Precision, Recall, F1
- **Top-k hit rates** at 10% and 20% — what fraction of real fraud is captured in the top-flagged accounts

---

## Project Structure

```
420Project/
├── skeleton.py          # Main EA implementation
├── generate_data.py     # Synthetic dataset generator
├── account_data.csv     # Synthetic dataset (after running generate_data.py)
├── creditcard.csv       # Kaggle dataset (if downloaded)
├── creditcard_prepped.csv  # Kaggle dataset after renaming target column
└── README.md
```
