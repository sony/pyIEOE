# Copyright (c) 2021 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

from inspect import isclass
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error as calc_mse,
    mean_absolute_error as calc_mae,
)
from obp.ope import (
    RegressionModel,
    OffPolicyEvaluation,
    BaseOffPolicyEstimator,
)
from obp.types import BanditFeedback

from .base import BaseOPEEvaluator
from .utils import _choose_uniform, _choose_log_uniform

COLORS = [
    "lightcoral",
    "plum",
    "lightgreen",
    "lightskyblue",
    "lightsalmon",
    "orange",
    "forestgreen",
    "royalblue",
    "gold",
    "blueviolet",
    "fuchsia",
    "lightpink",
    "firebrick",
    "peru",
    "darkkhaki",
    "darkolivegreen",
    "navy",
    "deeppink",
    "black",
    "silver",
]

LINESTYLES = [
    "solid",
    (0, (1, 0.6)),
    (0, (1, 1.2)),
    (0, (1, 1.8)),
    (0, (1, 2.4)),
    (0, (1, 3)),
]


@dataclass
class InterpretableOPEEvaluator(BaseOPEEvaluator):
    """Class to carry out Interpretable OPE Evaluation.

    Parameters
    ----------
    random_states: np.ndarray
        list of integers representing random states
        length of random_states corresponds to the number of runs

    bandit_feedbacks: List[BanditFeedback]
        list of bandit feedbacks

    evaluation_policies: List[Tuple[float, np.ndarray]]
        list of tuples representing evaluation policies
        first entry in tuple represents the ground truth policy value
        second entry in tuple represents action distribution of evaluation policy

    ope_estimators: List[BaseOffPolicyEstimator]
        list of ope estimators from obp.ope

    ope_estimator_hyperparams: dict
        dictionary storing hyperparameters for ope estimators
        must be in the following format
            ope_estimator_hyperparams = dict(
                [OffPolicyEstimator].estimator_name = dict(
                    [parameter_name] = dict(
                        "lower":
                        "upper":
                        "log":
                        "type":
                    )
                ),
            )

    regression_models: Optional[List[Union[BaseEstimator, BaseSearchCV]]]
        list of regression models to be used in off policy evaluation
        each element must either be of type BaseEstimator or BaseSearchCV

    regression_model_hyperparams: dict
        dictionary storing hyperparameters for regression models
        must be in the following format
            regression_model_hyperparams = dict(
                [model_name] = dict(
                    [parameter_name] = dict(
                        "lower":
                        "upper":
                        "log":
                        "type":
                    )
                ),
            )

    pscore_estimators: Optional[List[Union[BaseEstimator, BaseSearchCV]]]
        list of classification models to be used in estimating propensity scores of behavior policy
        each element must either be of type BaseEstimator or BaseSearchCV

    pscore_estimator_hyperparams: dict
        dictionary storing hyperparameters for pscore estimators
        must be in the following format
            pscore_estimator_hyperparams = dict(
                [model_name] = dict(
                    [parameter_name] = dict(
                        "lower":
                        "upper":
                        "log":
                        "type":
                    )
                ),
            )
    """

    random_states: np.ndarray
    ope_estimators: List[BaseOffPolicyEstimator]
    bandit_feedbacks: List[BanditFeedback]
    evaluation_policies: List[Tuple[float, np.ndarray]]
    ope_estimator_hyperparams: Optional[dict] = None
    regression_models: Optional[List[Union[BaseEstimator, BaseSearchCV]]] = None
    regression_model_hyperparams: Optional[dict] = None
    pscore_estimators: Optional[List[Union[BaseEstimator, BaseSearchCV]]] = None
    pscore_estimator_hyperparams: Optional[dict] = None

    def __post_init__(self) -> None:
        self.estimator_names = [est.estimator_name for est in self.ope_estimators]
        self.policy_value = None
        for i in np.arange(len(self.bandit_feedbacks)):
            if self.bandit_feedbacks[i]["position"] is None:
                self.bandit_feedbacks[i]["position"] = np.zeros_like(
                    self.bandit_feedbacks[i]["action"],
                    dtype=int,
                )
        if self.reward_type == "binary":
            self.reg_model_metric_names = ["auc", "rel_ce"]
        else:
            self.reg_model_metric_names = ["rel_mse", "rel_mae"]

        if not self.ope_estimator_hyperparams:
            self.ope_estimator_hyperparams = {
                estimator_name: dict() for estimator_name in self.estimator_names
            }

        if not self.regression_model_hyperparams:
            self.regression_model_hyperparams = {
                regression_model: dict() for regression_model in self.regression_models
            }

        if self.pscore_estimators and not self.pscore_estimator_hyperparams:
            self.pscore_estimator_hyperparams = {
                pscore_estimator: dict() for pscore_estimator in self.pscore_estimators
            }

    @property
    def n_runs(self) -> int:
        """Number of iterations."""
        return self.random_states.shape[0]

    @property
    def n_rounds(self) -> np.ndarray:
        """Number of observations in each given bandit_feedback in self.bandit_feedbacks"""
        return np.asarray(
            [bandit_feedback["n_rounds"] for bandit_feedback in self.bandit_feedbacks]
        )

    @property
    def n_actions(self) -> np.ndarray:
        """Number of actions in each given bandit_feedback in self.bandit_feedbacks"""
        return np.asarray(
            [bandit_feedback["n_actions"] for bandit_feedback in self.bandit_feedbacks]
        )

    @property
    def reward_type(self) -> np.ndarray:
        """Whether the reward is binary or continuous"""
        if np.unique(self.bandit_feedbacks[0]["reward"]).shape[0] == 2:
            return "binary"
        else:
            return "continuous"

    @property
    def len_list(self) -> np.ndarray:
        """Number of positions in each given bandit_feedback in self.bandit_feedbacks"""
        return np.asarray(
            [
                int(bandit_feedback["position"].max() + 1)
                for bandit_feedback in self.bandit_feedbacks
            ]
        )

    def estimate_policy_value(
        self,
        n_folds_: Union[int, Optional[dict]] = 2,
        sample_size: Optional[int] = None,
    ) -> dict:
        """Estimates the policy values using selected ope estimators under a range of environments."""
        # initialize dictionaries to store results
        self.policy_value = {est: np.zeros(self.n_runs) for est in self.estimator_names}
        self.squared_error = {
            est: np.zeros(self.n_runs) for est in self.estimator_names
        }
        self.reg_model_metrics = {
            metric: np.zeros(self.n_runs) for metric in self.reg_model_metric_names
        }
        for i, s in enumerate(tqdm(self.random_states)):
            np.random.seed(seed=s)
            # randomly select bandit_feedback
            self.bandit_feedback = self._choose_bandit_feedback(s)

            if self.pscore_estimators is not None:
                # randomly choose pscore estimator
                pscore_estimator = np.random.choice(self.pscore_estimators)
                # randomly choose hyperparameters of pscore estimator
                if isinstance(pscore_estimator, BaseEstimator):
                    classifier = pscore_estimator
                    setattr(classifier, "random_state", s)
                elif isclass(pscore_estimator) and issubclass(
                    pscore_estimator, BaseEstimator
                ):
                    pscore_estimator_hyperparam = (
                        self._choose_pscore_estimator_hyperparam(s, pscore_estimator)
                    )
                    classifier = clone(pscore_estimator(**pscore_estimator_hyperparam))
                else:
                    raise ValueError(
                        f"pscore_estimator must be BaseEstimator or BaseSearchCV, but {type(pscore_estimator)} is given."
                    )
                # fit classifier
                classifier.fit(
                    self.bandit_feedback["context"], self.bandit_feedback["action"]
                )
                estimated_pscore = classifier.predict_proba(
                    self.bandit_feedback["context"]
                )
                # replace pscore in bootstrap bandit feedback with estimated pscore
                self.bandit_feedback["pscore"] = estimated_pscore[
                    np.arange(self.bandit_feedback["n_rounds"]),
                    self.bandit_feedback["action"],
                ]

            # randomly sample from selected bandit_feedback
            bootstrap_bandit_feedback = self._sample_bootstrap_bandit_feedback(
                s, sample_size
            )
            # randomly choose hyperparameters of ope estimators
            self._choose_ope_estimator_hyperparam(s)
            # randomly choose regression model
            regression_model = self._choose_regression_model(s)
            # randomly choose hyperparameters of regression models
            if isinstance(regression_model, BaseEstimator):
                setattr(regression_model, "random_state", s)
            elif isclass(regression_model) and issubclass(
                regression_model, BaseEstimator
            ):
                regression_model_hyperparam = self._choose_regression_model_hyperparam(
                    s, regression_model
                )
                regression_model = regression_model(**regression_model_hyperparam)
            else:
                raise ValueError(
                    f"regression_model must be BaseEstimator or BaseSearchCV, but {type(regression_model)} is given."
                )
            # randomly choose evaluation policy
            ground_truth, bootstrap_action_dist = self._choose_evaluation_policy(s)
            # randomly choose number of folds
            if isinstance(n_folds_, dict):
                n_folds = _choose_uniform(
                    s,
                    n_folds_["lower"],
                    n_folds_["upper"],
                    n_folds_["type"],
                )
            else:
                n_folds = n_folds_
            # estimate policy value using each ope estimator under setting s
            (
                policy_value_s,
                estimated_rewards_by_reg_model_s,
            ) = self._estimate_policy_value_s(
                s,
                bootstrap_bandit_feedback,
                regression_model,
                bootstrap_action_dist,
                n_folds,
            )
            # calculate squared error for each ope estimator
            squared_error_s = self._calculate_squared_error_s(
                policy_value_s,
                ground_truth,
            )
            # evaluate the performance of reg_model
            r_pred = estimated_rewards_by_reg_model_s[
                np.arange(bootstrap_bandit_feedback["n_rounds"]),
                bootstrap_bandit_feedback["action"],
                bootstrap_bandit_feedback["position"],
            ]
            reg_model_metrics = self._calculate_rec_model_performance_s(
                r_true=bootstrap_bandit_feedback["reward"],
                r_pred=r_pred,
            )
            # store results
            for est in self.estimator_names:
                self.policy_value[est][i] = policy_value_s[est]
                self.squared_error[est][i] = squared_error_s[est]
            for j, metric in enumerate(self.reg_model_metric_names):
                self.reg_model_metrics[metric][i] = reg_model_metrics[j].mean()
        return self.policy_value

    def calculate_squared_error(self) -> dict:
        """Calculates the squared errors using selected ope estimators under a range of environments."""
        if not self.policy_value:
            _ = self.estimate_policy_value()
        return self.squared_error

    def calculate_variance(self, scale: bool = False, std: bool = True) -> dict:
        """Calculates the variance of squared errors."""
        if not self.policy_value:
            _ = self.estimate_policy_value()
        if std:
            self.variance = {
                key: np.sqrt(np.var(val)) for key, val in self.squared_error.items()
            }
        else:
            self.variance = {
                key: np.var(val) for key, val in self.squared_error.items()
            }
        variance = self.variance.copy()

        if scale:
            c = min(variance.values())
            for est in self.estimator_names:
                if type(variance[est]) != str:
                    variance[est] = variance[est] / c
        return variance

    def calculate_mean(self, scale: bool = False, root: bool = False) -> dict:
        """Calculates the mean of squared errors."""
        if not self.policy_value:
            _ = self.estimate_policy_value()
        if root:  # root mean squared error
            self.mean = {
                key: np.sqrt(np.mean(val)) for key, val in self.squared_error.items()
            }
        else:  # mean squared error
            self.mean = {key: np.mean(val) for key, val in self.squared_error.items()}
        mean = self.mean.copy()

        if scale:
            c = min(mean.values())
            for est in self.estimator_names:
                if type(mean[est]) != str:
                    mean[est] = mean[est] / c
        return mean

    def save_policy_value(
        self,
        file_dir: str = "results",
        file_name: str = "ieoe_policy_value.csv",
    ) -> None:
        """Save policy_value to csv file."""
        path = Path(file_dir)
        path.mkdir(exist_ok=True, parents=True)
        ieoe_policy_value_df = pd.DataFrame(self.policy_value, self.random_states)
        ieoe_policy_value_df.to_csv(f"{file_dir}/{file_name}")

    def save_squared_error(
        self,
        file_dir: str = "results",
        file_name: str = "ieoe_squared_error.csv",
    ) -> None:
        """Save squared_error to csv file."""
        path = Path(file_dir)
        path.mkdir(exist_ok=True, parents=True)
        ieoe_squared_error_df = pd.DataFrame(self.squared_error, self.random_states)
        ieoe_squared_error_df.to_csv(f"{file_dir}/{file_name}")

    def save_variance(
        self,
        file_dir: str = "results",
        file_name: str = "ieoe_variance.csv",
    ) -> None:
        """Save squared_error to csv file."""
        path = Path(file_dir)
        path.mkdir(exist_ok=True, parents=True)
        ieoe_variance_df = pd.DataFrame(self.variance.values(), self.variance.keys())
        ieoe_variance_df.to_csv(f"{file_dir}/{file_name}")

    def visualize_cdf(
        self,
        fig_dir: str = "figures",
        fig_name: str = "cdf.png",
        font_size: int = 12,
        fig_width: float = 8,
        fig_height: float = 6,
        kde: Optional[bool] = False,
    ) -> None:
        """Create a cdf graph for each ope estimator."""
        path = Path(fig_dir)
        path.mkdir(exist_ok=True, parents=True)
        for est in self.estimator_names:
            plt.clf()
            plt.style.use("ggplot")
            plt.rcParams.update({"font.size": font_size})
            _, ax = plt.subplots(figsize=(fig_width, fig_height))
            if kde:
                sns.kdeplot(
                    x=self.squared_error[est],
                    kernel="gaussian",
                    cumulative=True,
                    ax=ax,
                    label=est,
                    linewidth=3.0,
                    bw_method=0.05,
                )
            else:
                sns.ecdfplot(
                    self.squared_error[est],
                    ax=ax,
                    label=est,
                    linewidth=3.0,
                )
            plt.legend()
            plt.title(f"{est}: Cumulative distribution of squared error")
            plt.xlabel("Squared error")
            plt.ylabel("Cumulative probability")
            plt.xlim(0, None)
            plt.ylim(0, 1.1)
            plt.savefig(f"{fig_dir}/{est}_{fig_name}", dpi=100)
            plt.show()

    def visualize_cdf_aggregate(
        self,
        fig_dir: str = "figures",
        fig_name: str = "cdf.png",
        font_size: int = 12,
        fig_width: float = 8,
        fig_height: float = 6,
        xmax: Optional[float] = None,
        kde: Optional[bool] = False,
        linestyles: Optional[bool] = False,
    ) -> None:
        """Create a graph containing the cdf of all ope estimators."""
        path = Path(fig_dir)
        path.mkdir(exist_ok=True, parents=True)
        plt.clf()
        plt.style.use("ggplot")
        plt.rcParams.update({"font.size": font_size})
        _, ax = plt.subplots(figsize=(fig_width, fig_height))
        for i, est in enumerate(self.estimator_names):
            if i < len(COLORS):
                color = COLORS[i]
            else:
                color = np.random.rand(
                    3,
                )
            if linestyles:
                linestyle = LINESTYLES[i % len(LINESTYLES)]
            else:
                linestyle = "solid"
            if kde:
                sns.kdeplot(
                    x=self.squared_error[est],
                    kernel="gaussian",
                    cumulative=True,
                    ax=ax,
                    label=est,
                    linewidth=3.0,
                    bw_method=0.05,
                    alpha=0.7,
                    c=color,
                    linestyle=linestyle,
                )
            else:
                sns.ecdfplot(
                    self.squared_error[est],
                    ax=ax,
                    label=est,
                    linewidth=3.0,
                    alpha=0.7,
                    c=color,
                    linestyle=linestyle,
                )
        plt.legend(loc="lower right")
        plt.title("Cumulative distribution of squared error")
        plt.xlabel("Squared error")
        plt.ylabel("Cumulative probability")
        plt.xlim(0, xmax)
        plt.ylim(0, 1.1)
        plt.savefig(f"{fig_dir}/{fig_name}", dpi=100)
        plt.show()

    def visualize_squared_error_density(
        self,
        fig_dir: str = "figures",
        fig_name: str = "squared_error_density_estimation.png",
        font_size: int = 12,
        fig_width: float = 8,
        fig_height: float = 6,
    ) -> None:
        """Create a graph based on kernel density estimation of squared error for each ope estimator."""
        path = Path(fig_dir)
        path.mkdir(exist_ok=True, parents=True)
        for est in self.estimator_names:
            plt.clf()
            plt.style.use("ggplot")
            plt.rcParams.update({"font.size": font_size})
            _, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.kdeplot(
                self.squared_error[est],
                ax=ax,
                label=est,
                linewidth=3.0,
            )
            plt.legend()
            plt.title(f"{est}: Graph of estimated density of squared error")
            plt.xlabel(
                "Squared error",
            )
            plt.savefig(f"{fig_dir}/{est}_{fig_name}", dpi=100)
            plt.show()

    def calculate_au_cdf_score(
        self,
        threshold: float,
        scale: bool = False,
    ) -> dict:
        """Calculate AU-CDF score."""
        au_cdf_score = {est: None for est in self.estimator_names}
        for est in self.estimator_names:
            au_cdf_score[est] = np.mean(
                np.clip(threshold - self.squared_error[est], 0, None)
            )
        if scale:
            c = max(au_cdf_score.values())
            for est in self.estimator_names:
                au_cdf_score[est] = au_cdf_score[est] / c
        return au_cdf_score

    def calculate_cvar_score(
        self,
        alpha: float,
        scale: bool = False,
    ) -> dict:
        """Calculate CVaR score."""
        cvar_score = {est: None for est in self.estimator_names}
        for est in self.estimator_names:
            threshold = np.percentile(self.squared_error[est], alpha)
            bool_ = self.squared_error[est] >= threshold
            if any(bool_):
                cvar_score[est] = np.sum(self.squared_error[est] * bool_) / np.sum(
                    bool_
                )
            else:
                cvar_score[
                    est
                ] = f"the largest squared error is less than the threshold value {threshold}"
        if scale:
            c = min(cvar_score.values())
            for est in self.estimator_names:
                if type(cvar_score[est]) != str:
                    cvar_score[est] = cvar_score[est] / c
        return cvar_score

    def set_ope_estimator_hyperparam_space(
        self,
        ope_estimator_name: str,
        param_name: str,
        lower: Union[int, float],
        upper: Union[int, float],
        log: Optional[bool] = False,
        type_: Optional[type] = int,
    ) -> None:
        """Specify sampling method of hyperparameter of ope estimator."""
        assert type_ in [
            int,
            float,
        ], f"`type_` must be int or float but {type_} is given"
        dic = {
            "lower": lower,
            "upper": upper,
            "log": log,
            "type": type_,
        }
        self.ope_estimator_hyperparams[ope_estimator_name][param_name] = dic

    def set_regression_model_hyperparam_space(
        self,
        regression_model: Union[BaseEstimator, BaseSearchCV],
        param_name: str,
        lower: Union[int, float],
        upper: Union[int, float],
        log: Optional[bool] = False,
        type_: Optional[type] = int,
    ) -> None:
        """Specify sampling method of hyperparameter of regression model."""
        assert type_ in [
            int,
            float,
        ], f"`type_` must be int or float but {type_} is given"
        dic = {
            "lower": lower,
            "upper": upper,
            "log": log,
            "type": type_,
        }
        self.regression_model_hyperparams[regression_model][param_name] = dic

    def _choose_bandit_feedback(
        self,
        s: int,
    ) -> BanditFeedback:
        """Randomly select bandit_feedback."""
        np.random.seed(seed=s)
        idx = np.random.choice(len(self.bandit_feedbacks))
        return self.bandit_feedbacks[idx]

    def _sample_bootstrap_bandit_feedback(
        self, s: int, sample_size: Optional[int]
    ) -> BanditFeedback:
        """Randomly sample bootstrap data from bandit_feedback."""
        bootstrap_bandit_feedback = self.bandit_feedback.copy()
        np.random.seed(seed=s)
        if sample_size is None:
            sample_size = self.bandit_feedback["n_rounds"]
        self.bootstrap_idx = np.random.choice(
            np.arange(sample_size), size=sample_size, replace=True
        )
        for key_ in self.bandit_feedback.keys():
            # if the size of a certain key_ is not equal to n_rounds,
            # we should not resample that certain key_
            # e.g. we want to resample action and reward, but not n_rounds
            if (
                not isinstance(self.bandit_feedback[key_], np.ndarray)
                or len(self.bandit_feedback[key_]) != self.bandit_feedback["n_rounds"]
            ):
                continue
            bootstrap_bandit_feedback[key_] = bootstrap_bandit_feedback[key_][
                self.bootstrap_idx
            ]
        bootstrap_bandit_feedback["n_rounds"] = sample_size
        return bootstrap_bandit_feedback

    def _choose_ope_estimator_hyperparam(
        self,
        s: int,
    ) -> None:
        """Randomly choose hyperparameters for ope estimators."""
        for i, est in enumerate(self.ope_estimators):
            hyperparam = self.ope_estimator_hyperparams.get(est.estimator_name, None)
            if not hyperparam:
                continue
            for p in hyperparam:
                if hyperparam[p].get("log", False):
                    val = _choose_log_uniform(
                        s,
                        hyperparam[p]["lower"],
                        hyperparam[p]["upper"],
                        hyperparam[p].get("type", int),
                    )
                else:
                    val = _choose_uniform(
                        s,
                        hyperparam[p]["lower"],
                        hyperparam[p]["upper"],
                        hyperparam[p].get("type", int),
                    )
                setattr(est, p, val)
            self.ope_estimators[i] = est

    def _choose_regression_model(
        self,
        s: int,
    ) -> Union[BaseEstimator, BaseSearchCV]:
        """Randomly choose regression model."""
        idx = np.random.choice(len(self.regression_models))
        return self.regression_models[idx]

    def _choose_regression_model_hyperparam(
        self,
        s: int,
        regression_model: Union[BaseEstimator, BaseSearchCV],
    ) -> dict:
        """Randomly choose hyperparameters for regression model."""
        hyperparam = dict(
            random_state=s,
        )
        hyperparam_set = self.regression_model_hyperparams.get(regression_model, None)
        if not hyperparam_set:
            return hyperparam
        for p in hyperparam_set:
            if hyperparam_set[p].get("log", False):
                val = _choose_log_uniform(
                    s,
                    hyperparam_set[p]["lower"],
                    hyperparam_set[p]["upper"],
                    hyperparam_set[p].get("type", int),
                )
            else:
                val = _choose_uniform(
                    s,
                    hyperparam_set[p]["lower"],
                    hyperparam_set[p]["upper"],
                    hyperparam_set[p].get("type", int),
                )
            hyperparam[p] = val
        return hyperparam

    def _choose_pscore_estimator_hyperparam(
        self,
        s: int,
        pscore_estimator: Union[BaseEstimator, BaseSearchCV],
    ) -> dict:
        """Randomly choose hyperparameters for pscore estimator."""
        hyperparam = dict(
            random_state=s,
        )
        hyperparam_set = self.pscore_estimator_hyperparams.get(pscore_estimator, None)
        if not hyperparam_set:
            return hyperparam
        for p in hyperparam_set:
            if hyperparam_set[p].get("log", False):
                val = _choose_log_uniform(
                    s,
                    hyperparam_set[p]["lower"],
                    hyperparam_set[p]["upper"],
                    hyperparam_set[p].get("type", int),
                )
            else:
                val = _choose_uniform(
                    s,
                    hyperparam_set[p]["lower"],
                    hyperparam_set[p]["upper"],
                    hyperparam_set[p].get("type", int),
                )
            hyperparam[p] = val
        return hyperparam

    def _choose_evaluation_policy(
        self,
        s: int,
    ) -> Tuple[float, np.ndarray]:
        """Randomly choose evaluation policy and resample using bootstrap."""
        np.random.seed(seed=s)
        idx = np.random.choice(len(self.evaluation_policies))
        ground_truth, action_dist = self.evaluation_policies[idx]
        action_dist = action_dist[self.bootstrap_idx]
        return ground_truth, action_dist

    def _estimate_policy_value_s(
        self,
        s: int,
        bootstrap_bandit_feedback: BanditFeedback,
        _regression_model: Union[BaseEstimator, BaseSearchCV],
        bootstrap_action_dist: np.ndarray,
        n_folds: int,
    ) -> Tuple[dict, np.ndarray]:
        """Estimates the policy values using selected ope estimators under a particular environments."""
        # prepare regression model for ope
        regression_model = RegressionModel(
            n_actions=self.bandit_feedback["n_actions"],
            len_list=int(self.bandit_feedback["position"].max() + 1),
            base_model=_regression_model,
            fitting_method="normal",
        )
        estimated_reward_by_reg_model = regression_model.fit_predict(
            context=bootstrap_bandit_feedback["context"],
            action=bootstrap_bandit_feedback["action"],
            reward=bootstrap_bandit_feedback["reward"],
            position=bootstrap_bandit_feedback["position"],
            pscore=bootstrap_bandit_feedback["pscore"],
            action_dist=bootstrap_action_dist,
            n_folds=n_folds,
            random_state=int(s),
        )

        # estimate policy value using ope
        ope = OffPolicyEvaluation(
            bandit_feedback=bootstrap_bandit_feedback,
            ope_estimators=self.ope_estimators,
        )
        estimated_policy_value = ope.estimate_policy_values(
            action_dist=bootstrap_action_dist,
            estimated_rewards_by_reg_model=estimated_reward_by_reg_model,
        )

        return estimated_policy_value, estimated_reward_by_reg_model

    def _calculate_squared_error_s(
        self,
        policy_value: dict,
        ground_truth: float,
    ) -> dict:
        """Calculate squared error."""
        squared_error = {
            est: np.square(policy_value[est] - ground_truth)
            for est in self.estimator_names
        }
        return squared_error

    def _calculate_rec_model_performance_s(
        self,
        r_true: np.ndarray,
        r_pred: np.ndarray,
    ) -> Tuple[float, float]:
        """Calculate performance of reg model."""
        r_naive_pred = np.ones_like(r_true) * r_true.mean()
        if self.reward_type == "binary":
            auc = roc_auc_score(r_true, r_pred)
            ce = log_loss(r_true, r_pred)
            ce_naive = log_loss(r_true, r_naive_pred)
            rel_ce = 1 - (ce / ce_naive)
            return auc, rel_ce

        elif self.reward_type == "continuous":
            mse = calc_mse(r_true, r_pred)
            mse_naive = calc_mse(r_true, r_naive_pred)
            rel_mse = 1 - (mse / mse_naive)
            mae = calc_mae(r_true, r_pred)
            mae_naive = calc_mae(r_true, r_naive_pred)
            rel_mae = 1 - (mae / mse_naive)
            return rel_mse, rel_mae

    def load_squared_error(
        self,
        file_dir: str,
        file_name: str,
    ) -> None:
        df = pd.read_csv(f"{file_dir}/{file_name}")
        self.squared_error = {est: None for est in self.estimator_names}
        for est in self.estimator_names:
            self.squared_error[est] = df[est].values
