import argparse
import copy
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Data Layer
def load_data(path):
    # Inputs account-level data. Scale and filter

    #returns X,y:
        # X = feature matrix
        # y = fraud labels

    df = pd.read_csv(path)

    # Drop rows missing the label
    df = df.dropna(subset=['fraud_label'])
    y = df['fraud_label'].astype(int).values

    # Drop label and any non-numeric/id columns
    X_df = df.drop(columns=['fraud_label'])
    X_df = X_df.select_dtypes(include='number')

    medians = X_df.median()
    X_df = X_df.fillna(medians)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)

    feature_names = list(X_df.columns)

    return X, y, scaler, feature_names, medians

def preprocess_new_data(df, scaler, feature_names, medians):
    # Preprocess new data using same steps as training data
    X_df = df.copy()
    X_df = X_df.select_dtypes(include='number')

    for col in feature_names:
        if col not in X_df.columns:
            X_df[col] = medians[col]

    X_df = X_df[feature_names].copy()
    X_df = X_df.fillna(medians)

    X_new = scaler.transform(X_df)
    return X_new

# Individual Representation
def create_hypothesis(feature_space):
    """
    Select subset and assign conditions
        Example: revenue_growth > threshold

    Returns individual: list of conditions, each a dict:
        {'feature': int index, 'op': '>' or '<', 'threshold': float}
    """
    n_conditions = random.randint(1, 3) # 1 to 3 conditions
    selected = random.sample(list(feature_space), n_conditions)

    individual = []
    for feat_idx in selected:
        individual.append({
            'feature': feat_idx,
            'op': random.choice(['>', '<']),
            'threshold': round(random.uniform(-2.0, 2.0), 4)  # scaled data is ~N(0,1)
        })
    return individual

def apply_hypothesis(individual, X):
    # Apply fraud hypo to data — all conditions must hold (AND logic)
    # Returns binary predictions array (1 = flagged as fraud)
    mask = np.ones(len(X), dtype=bool)
    for cond in individual:
        col = X[:, cond['feature']]
        if cond['op'] == '>':
            mask &= col > cond['threshold']
        else:
            mask &= col < cond['threshold']
    return mask.astype(int)

# Population Management
def initialize_population(pop_size, feature_space):
    # Return list of individuals
    return [create_hypothesis(feature_space) for i in range(pop_size)]

# Clustering
def calc_hypo_representation(individual, X):
    # Convert individual into vector for clustering
    n_features = X.shape[1]
    ind_vector = np.zeros(n_features * 3)  # feature, op, threshold for each feature
    for cond in individual:
        idx = cond['feature'] * 3
        ind_vector[idx] = 1  # feature used
        ind_vector[idx + 1] = 1 if cond['op'] == '>' else 0  # op
        ind_vector[idx + 2] = cond['threshold']  # threshold
    # Returns vector representation
    return ind_vector

def cluster_hypo(population, X, k):
    # Apply k-means clustering to group indivs by similarity
    if len(population) == 0:
        return {}
    representations = np.array([calc_hypo_representation(ind, X) for ind in population])

    k = min(k, len(population))  # can't have more clusters than individuals

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(representations)
    # Returns clusters dict
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    return clusters

# ElasticNet Regression
def train_regression(X, y):
    # Train ElasticNet, tune hyperparameters, use CV
    model = ElasticNetCV(cv=5, random_state=42)
    model.fit(X, y)
    # Returns trained model
    return model

def predict_risk(model, X):
    # Generate fraud risk scores using ElasticNet
    return model.predict(X)

# Fitness
def evaluate_hypo(individual, model, X, y):
    predictions = apply_hypothesis(individual, X)
    n_flagged = predictions.sum()
    coverage = n_flagged / len(predictions)

    # Degenerate rules get zero fitness
    if n_flagged == 0 or n_flagged == len(predictions):
        return 0.0

    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)

    # Fraction of true frauds captured by flagged rows
    total_fraud = y.sum()
    fraud_captured = y[predictions == 1].sum()
    topk_capture = fraud_captured / total_fraud if total_fraud > 0 else 0.0

    # Reward agreement with ElasticNet: flagged rows should have higher risk scores
    risk_scores = predict_risk(model, X)
    mean_risk_flagged = risk_scores[predictions == 1].mean()
    mean_risk_unflagged = risk_scores[predictions == 0].mean()
    risk_agreement = max(0.0, mean_risk_flagged - mean_risk_unflagged)

    # Simplicity penalty: 0 for 1 condition, grows with more
    simplicity_penalty = 0.05 * (len(individual) - 1)

    # Coverage penalty: rules that flag almost nothing or almost everything
    if coverage < 0.01 or coverage > 0.5:
        coverage_penalty = 0.2
    else:
        coverage_penalty = 0.0

    fitness = (0.35 * recall
               + 0.25 * precision
               + 0.25 * topk_capture
               + 0.15 * min(risk_agreement, 1.0)
               - simplicity_penalty
               - coverage_penalty)
    return max(0.0, fitness)

def evaluate_pop(population, model, X, y):
    # Returns dict mapping population index -> fitness
    fitness_dict = {}
    for i, individual in enumerate(population):
        fitness_dict[i] = evaluate_hypo(individual, model, X, y)
    return fitness_dict

# Selection
def tourn_sel(cluster, fitness_dict, tournament_size):
    # cluster is a list of population indices
    size = min(tournament_size, len(cluster))
    contestants = random.sample(cluster, size)
    return max(contestants, key=lambda idx: fitness_dict[idx])

def select_pop(clusters, fitness_dict, tournament_size):
    # Run tournament selection in each cluster, return list of winning indices
    selected = []
    for cluster in clusters.values():
        if len(cluster) > 0:
            winner = tourn_sel(cluster, fitness_dict, tournament_size)
            selected.append(winner)
    return selected

# Genetic Variation
def crossover(p1, p2):
    # Merge conditions, deduplicate by feature, then sample 1-3
    combined = list(p1) + list(p2)
    seen = set()
    unique = []
    for cond in combined:
        if cond['feature'] not in seen:
            seen.add(cond['feature'])
            unique.append(cond)

    random.shuffle(unique)
    n = random.randint(1, min(3, len(unique)))
    return [copy.deepcopy(c) for c in unique[:n]]

def mutate(individual, feature_space, mutation_rate):
    if random.random() > mutation_rate:
        return individual

    ind = copy.deepcopy(individual)
    actions = ['change_feature', 'perturb_threshold', 'flip_op']
    if len(ind) < 3:
        actions.append('add_condition')
    if len(ind) > 1:
        actions.append('remove_condition')

    action = random.choice(actions)

    if action == 'change_feature':
        cond = random.choice(ind)
        cond['feature'] = random.randint(0, len(feature_space) - 1)
    elif action == 'perturb_threshold':
        cond = random.choice(ind)
        cond['threshold'] = round(cond['threshold'] + random.gauss(0, 0.3), 4)
    elif action == 'flip_op':
        cond = random.choice(ind)
        cond['op'] = '<' if cond['op'] == '>' else '>'
    elif action == 'add_condition':
        ind.append({
            'feature': random.randint(0, len(feature_space) - 1),
            'op': random.choice(['>', '<']),
            'threshold': round(random.uniform(-2.0, 2.0), 4)
        })
    elif action == 'remove_condition':
        ind.pop(random.randint(0, len(ind) - 1))

    return ind

# Evolution
def evolve(X, y, feature_space, pop_size, generations, k_clusters,
           mutation_rate=0.3, tournament_size=3, n_elites=None,
           patience=10):
    n_elites = n_elites if n_elites is not None else max(1, pop_size // 10)

    model = train_regression(X, y)
    population = initialize_population(pop_size, feature_space)

    best_fitness_history = []
    avg_fitness_history = []
    overall_best = None
    overall_best_fitness = -1.0
    no_improve_count = 0

    for gen in range(generations):
        fitness_dict = evaluate_pop(population, model, X, y)
        fitnesses = list(fitness_dict.values())

        gen_best_idx = max(fitness_dict, key=fitness_dict.get)
        gen_best_fitness = fitness_dict[gen_best_idx]
        gen_avg_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness_history.append(gen_best_fitness)
        avg_fitness_history.append(gen_avg_fitness)

        if gen_best_fitness > overall_best_fitness:
            overall_best_fitness = gen_best_fitness
            overall_best = copy.deepcopy(population[gen_best_idx])
            no_improve_count = 0
        else:
            no_improve_count += 1

        print(f"Gen {gen+1}/{generations} | best={gen_best_fitness:.4f} avg={gen_avg_fitness:.4f}")

        # Early stopping
        if no_improve_count >= patience:
            print(f"  No improvement for {patience} generations, stopping early.")
            break

        # Elitism: carry top individuals directly to next generation
        sorted_indices = sorted(fitness_dict, key=fitness_dict.get, reverse=True)
        elites = [copy.deepcopy(population[i]) for i in sorted_indices[:n_elites]]

        # Cluster and select parents
        clusters = cluster_hypo(population, X, k_clusters)
        parent_indices = select_pop(clusters, fitness_dict, tournament_size)
        parents = [population[i] for i in parent_indices]

        # Fill next generation with crossover + mutation offspring
        new_population = elites[:]
        while len(new_population) < pop_size:
            if len(parents) >= 2:
                p1, p2 = random.sample(parents, 2)
            else:
                p1 = p2 = parents[0]
            child = crossover(p1, p2)
            child = mutate(child, feature_space, mutation_rate)
            new_population.append(child)

        population = new_population[:pop_size]

    # Final evaluation on last generation
    fitness_dict = evaluate_pop(population, model, X, y)

    return {
        'population': population,
        'fitness_dict': fitness_dict,
        'model': model,
        'best_individual': overall_best,
        'best_fitness': overall_best_fitness,
        'best_fitness_history': best_fitness_history,
        'avg_fitness_history': avg_fitness_history,
    }

# Output
def get_top_hypo(population, fitness_dict, top_k):
    # Select best fraud hypotheses
    top_k = min(top_k, len(population))
    sorted_indices = sorted(fitness_dict, key=fitness_dict.get, reverse=True)

    # Return list of top hypos
    top_hypos = [population[i] for i in sorted_indices[:top_k]]
    return top_hypos

def convert_hypo(individual, feature_names=None):
    # Convert hypo to string for readability/usability
    substr = []
    for cond in individual:
        name = feature_names[cond['feature']] if feature_names else f"feature_{cond['feature']}"
        op = '>' if cond['op'] == '>' else '<'
        substr.append(f"{name} {op} {cond['threshold']}")
    return " AND ".join(substr)

def score_new_data(hypothesis, df_new, feature_names, medians, scaler):
    # Apply learned fraud rules to new data
    X_new = preprocess_new_data(df_new, scaler, feature_names, medians)
    predictions = apply_hypothesis(hypothesis, X_new)
    # Return ranked results
    return predictions

# Accuracy Test
def eval_performance(y, y_hat):
    # Test if model works
    y = np.asarray(y).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
    # Return metrics dictionary
    return {
        'precision': precision_score(y, y_hat, zero_division=0),
        'recall': recall_score(y, y_hat, zero_division=0),
        'f1': f1_score(y, y_hat, zero_division=0)
    }


def get_topk_hit_rate(y, scores, k_percent):
    # Compute percent of actual fraud captured
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores)

    k = int(k_percent * len(y))
    # Returns hit rate
    if k == 0:
        return 0.0
    topk_indices = np.argsort(scores)[::-1][:k]
    total_fraud = y.sum()
    return y[topk_indices].sum() / total_fraud if total_fraud > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Evolve fraud detection rules")
    parser.add_argument('--data_path', type=str, default='account_data.csv')
    parser.add_argument('--pop_size', type=int, default=100)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--k_clusters', type=int, default=5)
    parser.add_argument('--mutation_rate', type=float, default=0.3)
    parser.add_argument('--tournament_size', type=int, default=3)
    parser.add_argument('--n_elites', type=int, default=None)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    X, y, scaler, feature_names, medians = load_data(args.data_path)
    feature_space = range(X.shape[1])

    results = evolve(
        X, y,
        feature_space=feature_space,
        pop_size=args.pop_size,
        generations=args.generations,
        k_clusters=args.k_clusters,
        mutation_rate=args.mutation_rate,
        tournament_size=args.tournament_size,
        n_elites=args.n_elites,
        patience=args.patience
    )

    top_hypos = get_top_hypo(results['population'], results['fitness_dict'], top_k=5)
    print("\n=== Top Fraud Rules ===")
    for i, hypo in enumerate(top_hypos):
        print(f"  Rule {i+1}: {convert_hypo(hypo, feature_names)}")

    # Evaluate best hypothesis performance
    best = results['best_individual']
    y_hat = apply_hypothesis(best, X)
    metrics = eval_performance(y, y_hat)
    print(f"\n=== Best Rule Performance ===")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    # Top-k hit rates at 10% and 20%
    risk_scores = predict_risk(results['model'], X)
    for k in [0.10, 0.20]:
        rate = get_topk_hit_rate(y, risk_scores, k)
        print(f"  Top-{int(k*100)}% hit rate: {rate:.4f}")

    print(f"\nBest fitness: {results['best_fitness']:.4f}")
    return results

if __name__ == "__main__":
    main()