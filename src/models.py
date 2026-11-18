"""Model zoo for Step 3: one entry per algorithm family with its initial configuration."""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def make_models(seed=SEED):
    """name -> (family, estimator, max training rows or None). Row caps keep KNN / MLP tractable on ~850k rows."""
    return {
        "LogisticRegression": ("Linear", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)), None),
        "GaussianNB": ("Probabilistic", GaussianNB(), None),
        "KNN": ("Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)), 60_000),
        "RandomForest": ("Ensemble (bagging)", RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=10, max_samples=0.3,
                                                                     n_jobs=-1, random_state=seed), None),
        "LightGBM": ("Boosting", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=seed, verbose=-1), None),
        "MLP": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024, max_iter=25,
                                                                                early_stopping=True, n_iter_no_change=4, random_state=seed)), 300_000),
    }
