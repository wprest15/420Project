"""
Research Question:
    Can evolutionary algorithms discover interpretable fraud detection rules
    that outperform a standalone ElasticNet baseline in identifying financial fraud?

Experiment:
    1. Compare EA-evolved rules vs. ElasticNet-only baseline on three datasets
    2. Show fitness progression across generations (does the EA actually learn?)
    3. Compare performance across datasets of varying difficulty
    4. Compare different mutation rates to show parameter sensitivity

Run: python experiment.py
Produces plots saved to results/ folder.
"""
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # no display needed

from skeleton import (
    load_data, train_regression, predict_risk,
    initialize_population, evaluate_pop, cluster_hypo,
    select_pop, crossover, mutate, get_top_hypo,
    apply_hypothesis, eval_performance, get_topk_hit_rate,
    convert_hypo, evolve
)

os.makedirs('results', exist_ok=True)

DATASETS = {
    'Easy Financial':  'financial_easy.csv',
    'Hard Financial':  'financial_hard.csv',
    'Insurance Fraud': 'insurance_fraud.csv',
}

EA_PARAMS = dict(pop_size=100, generations=50, k_clusters=5,
                 mutation_rate=0.3, tournament_size=3, patience=15)


# ── Helper ────────────────────────────────────────────────────────────────────

def run_elasticnet_baseline(X, y):
    """Evaluate ElasticNet alone as a pure scoring model."""
    model = train_regression(X, y)
    scores = predict_risk(model, X)
    threshold = np.percentile(scores, 90)
    y_hat = (scores >= threshold).astype(int)
    metrics = eval_performance(y, y_hat)
    metrics['top10_hit_rate'] = get_topk_hit_rate(y, scores, 0.10)
    metrics['top20_hit_rate'] = get_topk_hit_rate(y, scores, 0.20)
    return metrics


def run_ea(X, y, **kwargs):
    """Run the full EA and return performance of the best evolved rule."""
    feature_space = range(X.shape[1])
    results = evolve(X, y, feature_space=feature_space, **kwargs)

    best = results['best_individual']
    y_hat = apply_hypothesis(best, X)
    metrics = eval_performance(y, y_hat)

    risk_scores = predict_risk(results['model'], X)
    metrics['top10_hit_rate'] = get_topk_hit_rate(y, risk_scores, 0.10)
    metrics['top20_hit_rate'] = get_topk_hit_rate(y, risk_scores, 0.20)
    metrics['best_fitness_history'] = results['best_fitness_history']
    metrics['avg_fitness_history']  = results['avg_fitness_history']
    metrics['best_individual'] = best
    metrics['model'] = results['model']
    return metrics


# ── Experiment 1: Fitness over generations ────────────────────────────────────
# Shows the EA is actually learning, not random

def exp1_fitness_curves():
    print("\n[Experiment 1] Fitness curves over generations")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle('EA Fitness Progression Across Datasets', fontsize=13, fontweight='bold')

    for ax, (name, path) in zip(axes, DATASETS.items()):
        print(f"  Running {name}...")
        X, y, *_ = load_data(path)
        res = run_ea(X, y, **EA_PARAMS)

        gens = range(1, len(res['best_fitness_history']) + 1)
        ax.plot(gens, res['best_fitness_history'], label='Best', color='steelblue', linewidth=2)
        ax.plot(gens, res['avg_fitness_history'],  label='Average', color='orange',
                linewidth=1.5, linestyle='--')
        ax.set_title(name)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/exp1_fitness_curves.png', dpi=150)
    plt.close()
    print("  Saved results/exp1_fitness_curves.png")


# ── Experiment 2: EA vs ElasticNet baseline comparison ───────────────────────
# Directly answers the research question

def exp2_ea_vs_baseline():
    print("\n[Experiment 2] EA vs ElasticNet baseline")
    metrics_list = []

    for name, path in DATASETS.items():
        print(f"  Running {name}...")
        X, y, *_ = load_data(path)
        en  = run_elasticnet_baseline(X, y)
        ea  = run_ea(X, y, **EA_PARAMS)
        metrics_list.append((name, en, ea))

    metric_keys = ['precision', 'recall', 'f1', 'top10_hit_rate']
    metric_labels = ['Precision', 'Recall', 'F1 Score', 'Top-10% Hit Rate']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('EA-Evolved Rules vs. ElasticNet Baseline', fontsize=13, fontweight='bold')

    x = np.arange(len(DATASETS))
    width = 0.35

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        en_vals = [m[1][key] for m in metrics_list]
        ea_vals = [m[2][key] for m in metrics_list]
        bars1 = ax.bar(x - width/2, en_vals, width, label='ElasticNet', color='coral', alpha=0.85)
        bars2 = ax.bar(x + width/2, ea_vals, width, label='EA Rules',   color='steelblue', alpha=0.85)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in metrics_list], fontsize=7, rotation=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig('results/exp2_ea_vs_baseline.png', dpi=150)
    plt.close()
    print("  Saved results/exp2_ea_vs_baseline.png")


# ── Experiment 3: Mutation rate sensitivity ───────────────────────────────────
# Shows how a key EA parameter affects performance — typical for a research paper

def exp3_mutation_sensitivity():
    print("\n[Experiment 3] Mutation rate sensitivity (financial_easy.csv)")
    X, y, *_ = load_data('financial_easy.csv')

    mutation_rates = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
    best_fitnesses = []
    f1_scores = []

    for rate in mutation_rates:
        print(f"  mutation_rate={rate}")
        params = {**EA_PARAMS, 'mutation_rate': rate}
        res = run_ea(X, y, **params)
        best_fitnesses.append(max(res['best_fitness_history']))
        f1_scores.append(res['f1'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Effect of Mutation Rate on EA Performance', fontsize=13, fontweight='bold')

    ax1.plot(mutation_rates, best_fitnesses, 'o-', color='steelblue', linewidth=2, markersize=7)
    ax1.set_xlabel('Mutation Rate')
    ax1.set_ylabel('Best Fitness Achieved')
    ax1.set_title('Best Fitness vs. Mutation Rate')
    ax1.grid(True, alpha=0.3)

    ax2.plot(mutation_rates, f1_scores, 's-', color='coral', linewidth=2, markersize=7)
    ax2.set_xlabel('Mutation Rate')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs. Mutation Rate')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/exp3_mutation_sensitivity.png', dpi=150)
    plt.close()
    print("  Saved results/exp3_mutation_sensitivity.png")


# ── Experiment 4: Top-k hit rate across datasets ──────────────────────────────
# Most important metric for the auditing use case

def exp4_topk_hit_rates():
    print("\n[Experiment 4] Top-k hit rates")
    k_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    colors = ['steelblue', 'coral', 'seagreen']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle('Fraud Captured in Top-k% of Flagged Accounts (EA Rules)',
                 fontsize=13, fontweight='bold')

    for ax, (name, path), color in zip(axes, DATASETS.items(), colors):
        print(f"  Running {name}...")
        X, y, *_ = load_data(path)
        res = run_ea(X, y, **EA_PARAMS)
        risk_scores = predict_risk(res['model'], X)

        hit_rates = [get_topk_hit_rate(y, risk_scores, k) for k in k_values]
        ax.plot([k*100 for k in k_values], hit_rates, 'o-', color=color, linewidth=2, markersize=7)
        ax.axline((0, 0), slope=1/100, linestyle='--', color='gray', alpha=0.5, label='Random baseline')
        ax.set_title(name)
        ax.set_xlabel('Top k% of Accounts Investigated')
        ax.set_ylabel('Fraction of All Fraud Captured')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/exp4_topk_hit_rates.png', dpi=150)
    plt.close()
    print("  Saved results/exp4_topk_hit_rates.png")


# ── Print summary table ───────────────────────────────────────────────────────

def print_summary():
    print("\n" + "="*65)
    print("SUMMARY TABLE")
    print("="*65)
    print(f"{'Dataset':<20} {'Method':<15} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Top10%':>7}")
    print("-"*65)
    for name, path in DATASETS.items():
        X, y, _, feature_names, _ = load_data(path)
        en  = run_elasticnet_baseline(X, y)
        ea  = run_ea(X, y, **EA_PARAMS)
        print(f"{name:<20} {'ElasticNet':<15} {en['precision']:>6.3f} {en['recall']:>6.3f} "
              f"{en['f1']:>6.3f} {en['top10_hit_rate']:>7.3f}")
        rule_str = convert_hypo(ea['best_individual'], feature_names)
        print(f"{'':<20} {'EA Rules':<15} {ea['precision']:>6.3f} {ea['recall']:>6.3f} "
              f"{ea['f1']:>6.3f} {ea['top10_hit_rate']:>7.3f}")
        print(f"  Best rule: {rule_str}")
        print()


if __name__ == '__main__':
    print("Running all experiments...")
    print("Research Question: Can EAs discover interpretable fraud rules that")
    print("outperform a standalone ElasticNet baseline?\n")

    exp1_fitness_curves()
    exp2_ea_vs_baseline()
    exp3_mutation_sensitivity()
    exp4_topk_hit_rates()
    print_summary()

    print("\nAll plots saved to results/")
    print("Use these figures directly in your paper and poster.")
