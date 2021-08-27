# hyperparameter space for regression models used in ope estimators (random search)

from sklearn.utils.fixes import loguniform
from scipy.stats import randint

C = loguniform(1e-3, 1e3)
max_iter_lr = randint(10000, 10000 + 1)

n_estimators = randint(100, 100 + 1)
max_depth = randint(2, 10 + 1)
min_samples_split = randint(5, 20 + 1)

max_iter_lgb = randint(100, 100 + 1)
min_samples_leaf = randint(5, 20 + 1)
learning_rate = loguniform(1e-4, 1e-1)

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
