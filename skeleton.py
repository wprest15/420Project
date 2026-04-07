# Data Layer
def load_data(path):
    # Inputs account-level data. Scale and filter

    #returns X,y:
        # X = feature matrix
        # y = fraud labels
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(path)

    # Drop rows missing the label
    df = df.dropna(subset=['fraud_label'])

    y = df['fraud_label'].astype(int).values

    # Drop label and any non-numeric/id columns
    X_df = df.drop(columns=['fraud_label'])
    X_df = X_df.select_dtypes(include='number')
    X_df = X_df.fillna(X_df.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)

    return X, y

# Individual Representation
def create_hypothesis(feature_space):
    """
    Select subset and assign conditions
        Example: revenue_growth > threshold

    Returns individual
    """
    pass

def apply_hypothesis(individual, X):
    # Apply fraud hypo to data

    # Returns predictions or scores
    pass

# Population Management
def initialize_population(pop_size, feature_space):
    # Return list of individuals
    pass

# Clustering
def calc_hypo_representation(individual, X):
    # Convert individual into vector for clustering

    # Returns vector representation
    pass

def cluster_hypo(population, X, k):
    # Apply k-means clustering to group indivs by similarity

    # Returns clusters dict
    pass

# ElasticNet Regression
def train_regression(X, y):
    # Train ElasticNet, tune hyperparameters, use CV

    # Returns trained model
    pass

def predict_risk(model, X):
    # Generate fraud risk scores using ElasticNet

    # Returns risk scores
    pass

# Fitness
def evaluate_hypo(individual, model, X, y):
    # Compute fitness of hypothesis and return score
    pass

def evaluate_pop(population, model, X, y):
    # evalute all indivs
    # return list of (individual, fitness)
    pass

# Selection
def tourn_sel(cluster, fitness_dict, tournament_size):
    # Tourn select within cluster

    # Return selected individuals
    pass

def select_pop(clusters, fitness_dict, tournament_size):
    # Select individuals across all clusters

    # Returns selected inds for reproduction
    pass

# Genetic Variation
def crossover(p1, p2):
    # Combine two hypothesis

    # Return child
    pass

def mutate(individual, feature_space, mutation_rate):
    # Mutate hypothesis

    # Return mutated ind
    pass

# Evolution
def evolve(X, y, feature_space, pop_size, generations, k_clusters):
    # Return final population and best individuals
    pass

# Output
def get_top_hypo(population, fitness_dict, top_k):
    # Select best fraud hypotheses

    # Return list of top hypos
    pass

def convert_hypo(individual):
    # Convert hypo to string for readability/usability
    pass

def score_new_data(hypothesis, X_new):
    # Apply learned fraud rules to new data

    # Return ranked results
    pass

# Accuracy Test
def eval_performance(y, y_hat):
    # Test if model works

    # Return metrics dictionary
    pass

def get_topk_hit_rate(y, y_hat, k_percent):
    # Compute percent of actual fraud captured

    # Returns hit rate
    pass

def main():

    pass