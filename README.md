# CS 420 Project: Evolutionary Fraud Rule Discovery for Financial Statement Audits

**Authors:** Walton Prest, Peter Wraith  
**Course:** CS 420

## Project Descriptor (Presentation-Ready)

This project builds an **interpretable fraud detection pipeline** that combines:

1. **ElasticNet** (a baseline statistical risk model), and
2. an **Evolutionary Algorithm (EA)** that discovers human-readable fraud rules.

The main objective is to show that we can move beyond black-box scoring and produce a ranked set of **auditable, threshold-based rules** (for example: `debt_to_equity > 1.36`) that still achieve strong fraud triage performance.

In practical audit settings, investigators often only have time to review a small fraction of accounts. This system is designed for that reality: prioritize the most suspicious accounts first, while keeping the decision logic transparent enough for governance, documentation, and review.

---

## Problem Statement and Motivation

Fraud detection systems in auditing face a tradeoff:

- High-performing models can be difficult to interpret.
- Fully interpretable rules are easier to defend but are often weaker if hand-crafted.

This project addresses that tension by **evolving simple rule sets automatically** and evaluating whether they can compete with, or complement, a baseline machine learning model.

### Research Question

> Can evolutionary algorithms discover interpretable fraud detection rules that are competitive with a standalone ElasticNet baseline across datasets of different difficulty?

---

## End-to-End Approach

The pipeline is:

1. **Load and preprocess data**
   - Read CSV input.
   - Require a `fraud_label` target column.
   - Keep numeric features, fill missing values with medians, and standardize.

2. **Train baseline ElasticNet model**
   - Use `ElasticNetCV` to generate continuous fraud risk scores.

3. **Initialize EA population of candidate rules**
   - Each rule is 1–3 conditions over scaled features.
   - Condition format: `(feature, operator, threshold)`.

4. **Evaluate candidate fitness**
   - Metrics combined in fitness: recall, precision, top-k fraud capture proxy, and agreement with ElasticNet scores.
   - Penalizes over-complex rules and poor coverage extremes.

5. **Preserve diversity with K-Means clustering**
   - Candidate rules are vectorized and clustered.
   - Selection occurs within clusters to avoid collapsing to one rule type too early.

6. **Evolve via tournament selection, crossover, and mutation**
   - Iterate over generations.
   - Use patience-based early stopping.

7. **Return top-k interpretable rules and audit-centric metrics**
   - Precision, Recall, F1
   - Top-10% and Top-20% hit rates

---

## What Makes This Project Distinct

- **Interpretability-first modeling**: produces explicit rule logic, not only probability scores.
- **Hybrid intelligence**: combines regression risk signal with evolutionary search.
- **Audit workflow alignment**: focuses on top-k investigation performance, which maps directly to limited review capacity.
- **Diversity-aware evolution**: uses clustering to maintain broad rule exploration.

---

## Repository Contents

```text
420Project/
├── README.md
├── skeleton.py                 # Core pipeline + EA engine (CLI entry point)
├── experiment.py               # Original experiment runner/plot generator
├── experiment2.py              # Train/test split experiment runner
├── make_diagram.py             # Pipeline figure generation utility
├── generate_data.py            # Synthetic single-dataset generator
├── generate_datasets.py        # Multi-dataset generator (easy/hard/insurance)
├── poster_24x36.html           # Presentation/poster artifact
├── account_data.csv            # Generated synthetic dataset
├── financial_easy.csv          # Easier financial dataset
├── financial_hard.csv          # Harder financial dataset
├── insurance_fraud.csv         # Insurance-style fraud dataset
├── creditcard.csv              # Optional Kaggle credit card dataset
├── creditcard_prepped.csv      # Renamed target-ready Kaggle dataset
└── results/                    # Generated figures from experiment scripts
```

---

## Installation

Install required Python packages:

```bash
pip install scikit-learn pandas numpy matplotlib
```

---

## Data Options

### Option A — Quick Synthetic Dataset

```bash
python generate_data.py
```

Creates `account_data.csv`.

### Option B — Three-Dataset Benchmark Setup

```bash
python generate_datasets.py
```

Creates benchmark-style files:

- `financial_easy.csv`
- `financial_hard.csv`
- `insurance_fraud.csv`

### Option C — Kaggle Credit Card Fraud Data

1. Place `creditcard.csv` in the project root.
2. Rename target column to `fraud_label`:

```bash
python -c "
import pandas as pd

df = pd.read_csv('creditcard.csv')
df = df.rename(columns={'Class': 'fraud_label'})
df.to_csv('creditcard_prepped.csv', index=False)
print(f'Done: {len(df)} rows, {df.fraud_label.sum()} fraud cases')
"
```

### Option D — Your Own Data

Provide a CSV where:

- target column is exactly `fraud_label` (0/1), and
- remaining numeric columns are usable features.

---

## Running the Core Pipeline

### Basic run

```bash
python skeleton.py --data_path <your_csv>
```

### Example (synthetic)

```bash
python skeleton.py --data_path account_data.csv
```

### Example (larger dataset, reduced settings)

```bash
python skeleton.py --data_path creditcard_prepped.csv --pop_size 50 --generations 20 --k_clusters 5 --patience 10
```

### Full custom run

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

## Main CLI Parameters

| Argument | Default | Meaning |
|---|---:|---|
| `--data_path` | `account_data.csv` | Input CSV path with `fraud_label` |
| `--pop_size` | `100` | Candidate rules per generation |
| `--generations` | `50` | Max EA iterations |
| `--k_clusters` | `5` | K-Means clusters for diversity |
| `--mutation_rate` | `0.3` | Mutation probability per offspring |
| `--tournament_size` | `3` | Selection pressure in each cluster |
| `--n_elites` | auto | Top performers copied directly forward |
| `--patience` | `10` | Early stop if no best-fitness gain |

---

## Experiments and Figures for Presentation

Run the experiment scripts to regenerate figures in `results/`.

```bash
python experiment.py
python experiment2.py
```

These scripts evaluate:

1. **Fitness progression over generations**
2. **EA rules vs ElasticNet baseline** on Precision/Recall/F1/Top-10% hit rate
3. **Mutation rate sensitivity**
4. **Top-k fraud capture curves**

The repository already includes generated figures under `results/` for poster/report use.

---

## Key Takeaways (for Slides)

- The EA can produce **interpretable rule-based fraud signals** while maintaining strong triage performance.
- Fraud capture is often high in top-ranked subsets, supporting practical audit prioritization.
- Performance varies by dataset difficulty; harder financial settings remain the main challenge.
- Moderate mutation rates generally provide stable performance, while overly aggressive mutation can reduce consistency.

---

## Limitations and Future Work

Current limitations:

- Fitness is tuned for current metrics and may not directly encode business cost asymmetry.
- Rules are currently simple threshold conjunctions.
- Current evaluation is dataset-based and can be expanded to stronger temporal/drift testing.

Future enhancements:

- Cost-sensitive and utility-aware fitness objectives.
- Richer rule grammar (OR groups, monotonic constraints, domain priors).
- Drift-aware retraining and periodic rule recalibration.
- Model governance outputs (rule cards, change logs, audit trail export).

---


## Presentation Talking Points (Speaker Notes)

Use these points as a guided script while presenting the poster/slides.

### 1) Opening Hook (30–45 seconds)

- Auditors usually cannot review every account manually.
- Most fraud pipelines optimize prediction quality, but audit teams also need **explanations they can defend**.
- This project asks: can we keep strong fraud triage performance **and** output clear, auditable decision rules?

### 2) Why This Problem Matters (business + governance)

- In real audits, teams investigate a ranked subset (for example top 10% or 20% of accounts).
- A slightly better top-k ranking can materially improve fraud recovery and resource allocation.
- Black-box models can be difficult to justify in regulated contexts.
- Interpretable rules improve handoff between data teams, auditors, and compliance stakeholders.

### 3) Core Idea in One Sentence

- We use ElasticNet for baseline risk signal, then evolve simple threshold rules with an EA to produce interpretable, high-priority fraud flags.

### 4) Walkthrough of the Pipeline Figure

When presenting the pipeline diagram, narrate left-to-right:

1. **Data preprocessing**: enforce `fraud_label`, keep numeric features, impute medians, standardize.
2. **ElasticNet baseline**: produces a stable numeric risk signal.
3. **EA initialization**: start with many simple candidate rules (1–3 conditions each).
4. **Fitness scoring**: reward recall, precision, top-k capture proxy, and agreement with risk signal.
5. **Diversity step**: K-Means groups similar rules to avoid premature convergence.
6. **Selection + variation**: tournament selection, crossover, mutation.
7. **Output**: top interpretable rules + metrics for audit triage.

### 5) How to Explain the Fitness Function

Suggested language:

- “The fitness is multi-objective in spirit: it balances finding fraud (recall), avoiding false alarms (precision), capturing fraud early in ranked review (top-k signal), and consistency with baseline risk scoring.”
- “We also penalize overly complex rules and degenerate rules that flag nearly everyone or almost no one.”
- “This pushes the search toward practical, human-reviewable rules.”

### 6) How to Read the Charts

- **EA fitness curves**: show whether evolution is learning over generations (best and average fitness trends).
- **EA vs ElasticNet bars**: compare precision/recall/F1/top-10% hit rate across datasets.
- **Top-k capture plots**: emphasize operational value—how much total fraud is captured when only a fraction of accounts can be investigated.
- **Mutation sensitivity plots**: demonstrate robustness and parameter behavior.

### 7) Dataset-by-Dataset Narrative

Use this quick framing:

- **Easy Financial**: confirms method can exploit clear signal and produce strong interpretable triage.
- **Hard Financial**: stress test; useful for discussing limitations and harder class boundaries.
- **Insurance Fraud**: shows portability of the framework to another fraud domain.

### 8) Key Contributions to Highlight

- A practical **hybrid interpretable framework** (not just a single model benchmark).
- Rule diversity control via clustering, improving exploration quality.
- Evaluation oriented to **audit workflow metrics** (top-k capture), not only generic ML scores.
- Outputs that can be turned into policy, controls, and investigation playbooks.

### 9) Honest Limitations (good for Q&A)

- Current rule grammar is conjunction-focused (threshold AND logic).
- Objective is not yet explicitly cost-sensitive by fraud dollar impact.
- Temporal drift and production retraining cadence are not fully modeled.
- Performance can vary with dataset difficulty and class imbalance.

### 10) Strong Future Work Talking Points

- Add cost-sensitive fitness tied to investigation budget and expected loss.
- Add richer rule structures (OR blocks, grouped features, monotonic constraints).
- Introduce drift monitoring + scheduled retraining.
- Add governance artifacts: rule cards, versioned rule audits, change logs.

### 11) Suggested 60-Second Closing

> “Our project shows that interpretable, evolved fraud rules can be competitive for real triage workflows. Instead of choosing between accuracy and explainability, this hybrid approach uses baseline statistical learning plus evolutionary search to deliver transparent rules that auditors can act on and defend. The strongest practical value is in top-k prioritization—helping teams catch more fraud earlier with limited review capacity.”

---

## One-Slide Verbal Summary

> We built a hybrid fraud detection system for financial audits that combines ElasticNet scoring with an evolutionary search for interpretable rules. Instead of only predicting risk, the system outputs clear threshold-based fraud rules and ranks accounts for investigation. Across multiple datasets, it demonstrates strong top-k fraud capture, making it practical for limited audit resources while preserving transparency and explainability.