from dataclasses import dataclass
from typing import Union

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import HistGradientBoostingClassifier as LightGBM


# policy class
@dataclass
class Policy:
    """
    base_classifier:
        base model (classifier) for the policy,
        should be chosen from [LogisticRegression, RandomForest, LightGBM]
    alpha:
        the ratio for random action to construct the policy from classifier
    """

    base_classifier: Union[LogisticRegression, RandomForest, LightGBM]
    alpha: float  # should be in [0, 1]

    def __post_init__(self):
        assert 0 <= self.alpha <= 1


# set random_state
random_state = 12345

# set behavior policy
behavior_policy = Policy(
    base_classifier=LogisticRegression(
        C=100, max_iter=10000, random_state=random_state
    ),
    alpha=0.9,
)

# set evaluation policies
evaluation_policy_a = Policy(
    base_classifier=LogisticRegression(
        C=100, max_iter=10000, random_state=random_state
    ),
    alpha=0.8,
)
evaluation_policy_b = Policy(
    base_classifier=LogisticRegression(
        C=100, max_iter=10000, random_state=random_state
    ),
    alpha=0.2,
)
evaluation_policy_c = Policy(
    base_classifier=LogisticRegression(
        C=100, max_iter=10000, random_state=random_state
    ),
    alpha=0.0,  # uniform random
)
evaluation_policy_d = Policy(
    base_classifier=RandomForest(
        n_estimators=100, min_samples_split=5, max_depth=10, random_state=random_state
    ),
    alpha=0.8,
)
evaluation_policy_e = Policy(
    base_classifier=RandomForest(
        n_estimators=100, min_samples_split=5, max_depth=10, random_state=random_state
    ),
    alpha=0.2,
)
evaluation_policies = [
    evaluation_policy_a,
    evaluation_policy_b,
    evaluation_policy_c,
    evaluation_policy_d,
    evaluation_policy_e,
]
