# hyperparameter space for regression models used in ope estimators (uniform)

C = {"lower": 1e-3, "upper": 1e3, "log": True, "type": float}
max_iter_lr = {"lower": 10000, "upper": 10000, "log": False, "type": int}

n_estimators = {"lower": 100, "upper": 100, "log": False, "type": int}
max_depth = {"lower": 2, "upper": 10, "log": False, "type": int}
min_samples_split = {"lower": 5, "upper": 20, "log": False, "type": int}

max_iter_lgb = {"lower": 100, "upper": 100, "log": False, "type": int}
min_samples_leaf = {"lower": 5, "upper": 20, "log": False, "type": int}
learning_rate = {"lower": 1e-4, "upper": 1e-1, "log": True, "type": float}

logistic_regression_param = {
    "C": C,
    "max_iter": max_iter_lr,
}
random_forest_param = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
}
lightgbm_param = {
    "max_iter": max_iter_lgb,
    "max_depth": max_depth,
    "min_samples_leaf": min_samples_leaf,
    "learning_rate": learning_rate,
}
