# Copyright (c) 2021 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import argparse
from distutils.util import strtobool
from pathlib import Path
import pickle

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

import numpy as np
from pandas import DataFrame
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import HistGradientBoostingClassifier as LightGBM
from sklearn.model_selection import RandomizedSearchCV

from obp.dataset import OpenBanditDataset
from obp.policy import Random, BernoulliTS
from obp.ope import (
    InverseProbabilityWeightingTuning,
    SelfNormalizedInverseProbabilityWeighting,
    DirectMethod,
    DoublyRobustTuning,
    SelfNormalizedDoublyRobust,
    SwitchDoublyRobustTuning,
    DoublyRobustWithShrinkageTuning,
)

from pyieoe.evaluator import InterpretableOPEEvaluator

# hyperparameter space for the OPE estimators themselves
from conf import ope_estimator_hyperparams

# hyperparameter space for the regression model used in model dependent OPE estimators
from conf import ope_regression_uniform_hyperparams
from conf import ope_regression_rscv_hyperparams


# compared ope estimators
ope_estimators = [
    InverseProbabilityWeightingTuning(
        lambdas=ope_estimator_hyperparams.tau_lambda, estimator_name="IPWps"
    ),
    SelfNormalizedInverseProbabilityWeighting(estimator_name="SNIPW"),
    DirectMethod(estimator_name="DM"),
    DoublyRobustTuning(
        lambdas=ope_estimator_hyperparams.tau_lambda, estimator_name="DRps"
    ),
    SelfNormalizedDoublyRobust(estimator_name="SNDR"),
    SwitchDoublyRobustTuning(
        taus=ope_estimator_hyperparams.tau_lambda, estimator_name="Switch-DR"
    ),
    DoublyRobustWithShrinkageTuning(
        lambdas=ope_estimator_hyperparams.tau_lambda, estimator_name="DRos"
    ),
]
ope_estimator_hyperparams_ = {
    DirectMethod.estimator_name: ope_estimator_hyperparams.dm_param,
    DoublyRobustTuning.estimator_name: ope_estimator_hyperparams.dr_param,
    SelfNormalizedDoublyRobust.estimator_name: ope_estimator_hyperparams.sndr_param,
    SwitchDoublyRobustTuning.estimator_name: ope_estimator_hyperparams.switch_dr_param,
    DoublyRobustWithShrinkageTuning.estimator_name: ope_estimator_hyperparams.dros_param,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate off-policy estimators with multi-class classification data."
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1000,
        help="number of seeds used in the experiment.",
    )
    parser.add_argument(
        "--use_random_search",
        type=strtobool,
        default=False,
        help="whether to use random search for hyperparamter selection or not, otherwise uniform sampling is used",
    )
    parser.add_argument(
        "--use_estimated_pscore",
        type=strtobool,
        default=False,
        help="whether to use estimated pscore or not, otherwise ground-truth pscore is used",
    )
    parser.add_argument(
        "--au_cdf_threshold",
        type=float,
        default=0.001,
        help="threshold (the maximum error allowed, z_max) for AU-CDF",
    )
    parser.add_argument(
        "--cvar_alpha",
        type=int,
        default=70,
        help="the percentile used for calculating CVaR, should be in (0, 100)",
    )
    parser.add_argument(
        "--campaign",
        type=str,
        default="men",
        choices=["all", "men", "women"],
        help="campaign name, men, women, or all.",
    )
    parser.add_argument(
        "--is_full_obd",
        type=strtobool,
        default=False,
        help="wheather to use the full size obd or not",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100000,
        help="(maximum) sample size for dataset to be used in the experiment (should be more than 10000)",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations
    n_seeds = args.n_seeds
    use_random_search = args.use_random_search
    use_estimated_pscore = args.use_estimated_pscore
    au_cdf_threshold = args.au_cdf_threshold
    cvar_alpha = args.cvar_alpha
    campaign = args.campaign
    obd_path = Path("./open_bandit_dataset/") if args.is_full_obd else None
    sample_size = args.sample_size
    random_state = args.random_state
    np.random.seed(random_state)
    # assertion
    assert 0 < au_cdf_threshold
    assert 0 < cvar_alpha < 100
    assert 10000 <= sample_size

    print("initializing experimental condition..")
    # load dataset
    dataset_ur = OpenBanditDataset(
        behavior_policy="random", campaign=campaign, data_path=obd_path
    )
    dataset_ts = OpenBanditDataset(
        behavior_policy="bts", campaign=campaign, data_path=obd_path
    )
    # obtain logged bandit feedback generated by the behavior policy
    bandit_feedback_ur = dataset_ur.obtain_batch_bandit_feedback()
    bandit_feedback_ts = dataset_ts.obtain_batch_bandit_feedback()
    bandit_feedbacks = [bandit_feedback_ur, bandit_feedback_ts]
    # define sample size to use
    sample_size = min(
        [sample_size, bandit_feedback_ur["n_rounds"], bandit_feedback_ts["n_rounds"]]
    )
    # obtain the ground-truth policy value
    ground_truth_ur = OpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy="random", campaign=campaign, data_path=obd_path
    )
    ground_truth_ts = OpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy="bts", campaign=campaign, data_path=obd_path
    )
    # define policies
    policy_ur = Random(
        n_actions=dataset_ur.n_actions,
        len_list=dataset_ur.len_list,
        random_state=random_state,
    )
    policy_ts = BernoulliTS(
        n_actions=dataset_ts.n_actions,
        len_list=dataset_ts.len_list,
        random_state=random_state,
        is_zozotown_prior=True,
        campaign=campaign,
    )
    # obtain action choice probabilities
    action_dist_ur = policy_ur.compute_batch_action_dist(n_rounds=1000000)
    action_dist_ts = policy_ts.compute_batch_action_dist(n_rounds=1000000)
    # define evaluation policies
    evaluation_policies = [
        (ground_truth_ts, action_dist_ts),
        (ground_truth_ur, action_dist_ur),
    ]

    # regression models used in ope estimators
    if use_random_search:
        logistic_regression = RandomizedSearchCV(
            LogisticRegression(),
            ope_regression_rscv_hyperparams.logistic_regression_param,
            random_state=random_state,
            n_iter=5,
        )
        random_forest = RandomizedSearchCV(
            RandomForest(),
            ope_regression_rscv_hyperparams.random_forest_param,
            random_state=random_state,
            n_iter=5,
        )
        lightgbm = RandomizedSearchCV(
            LightGBM(),
            ope_regression_rscv_hyperparams.lightgbm_param,
            random_state=random_state,
            n_iter=5,
        )
        regression_models = [
            logistic_regression,
            random_forest,
            lightgbm,
        ]

    else:  # uniform sampling
        regression_models = [
            LogisticRegression,
            RandomForest,
            LightGBM,
        ]
        regression_model_hyperparams = {
            LogisticRegression: ope_regression_uniform_hyperparams.logistic_regression_param,
            RandomForest: ope_regression_uniform_hyperparams.random_forest_param,
            LightGBM: ope_regression_uniform_hyperparams.lightgbm_param,
        }

    # initializing class
    if use_estimated_pscore:
        if use_random_search:
            evaluator = InterpretableOPEEvaluator(
                random_states=np.arange(n_seeds),
                bandit_feedbacks=bandit_feedbacks,
                evaluation_policies=evaluation_policies,
                ope_estimators=ope_estimators,
                ope_estimator_hyperparams=ope_estimator_hyperparams_,
                regression_models=regression_models,
                pscore_estimators=regression_models,
            )

        else:  # uniform sampling
            evaluator = InterpretableOPEEvaluator(
                random_states=np.arange(n_seeds),
                bandit_feedbacks=bandit_feedbacks,
                evaluation_policies=evaluation_policies,
                ope_estimators=ope_estimators,
                ope_estimator_hyperparams=ope_estimator_hyperparams_,
                regression_models=regression_models,
                regression_model_hyperparams=regression_model_hyperparams,
                pscore_estimators=regression_models,
                pscore_estimator_hyperparams=regression_model_hyperparams,
            )

    else:  # ground-truth pscore
        if use_random_search:
            evaluator = InterpretableOPEEvaluator(
                random_states=np.arange(n_seeds),
                bandit_feedbacks=bandit_feedbacks,
                evaluation_policies=evaluation_policies,
                ope_estimators=ope_estimators,
                ope_estimator_hyperparams=ope_estimator_hyperparams_,
                regression_models=regression_models,
            )

        else:  # uniform sampling
            evaluator = InterpretableOPEEvaluator(
                random_states=np.arange(n_seeds),
                bandit_feedbacks=bandit_feedbacks,
                evaluation_policies=evaluation_policies,
                ope_estimators=ope_estimators,
                ope_estimator_hyperparams=ope_estimator_hyperparams_,
                regression_models=regression_models,
                regression_model_hyperparams=regression_model_hyperparams,
            )

    # estimate policy values
    print("started experiment")
    policy_value = evaluator.estimate_policy_value(sample_size=sample_size)
    # calculate statistics
    print("calculating statistics of estimators' performance..")
    au_cdf = evaluator.calculate_au_cdf_score(threshold=au_cdf_threshold)
    au_cdf_scaled = evaluator.calculate_au_cdf_score(
        threshold=au_cdf_threshold, scale=True
    )
    cvar = evaluator.calculate_cvar_score(alpha=cvar_alpha)
    cvar_scaled = evaluator.calculate_cvar_score(alpha=cvar_alpha, scale=True)
    std = evaluator.calculate_variance(std=True)
    std_scaled = evaluator.calculate_variance(scale=True, std=True)
    mean = evaluator.calculate_mean()
    mean_scaled = evaluator.calculate_mean(scale=True)

    # rscv/uniform, estimated/ground-truth pscore option
    if use_random_search:
        if use_estimated_pscore:
            option = "rscv_pscore_estimate"
        else:
            option = "rscv_pscore_true"
    else:
        if use_estimated_pscore:
            option = "uniform_pscore_estimate"
        else:
            option = "uniform_pscore_true"
    # save results of the evaluation of off-policy estimators in './logs/(option)' directory.
    log_path = Path("./logs/" + option)
    log_path.mkdir(exist_ok=True, parents=True)
    print("the results will be saved in", log_path)
    # save evaluator in order to change au_cdf_threshold and cvar_alpha afterwhile
    f = open(log_path / "evaluator.pickle", "wb")
    pickle.dump(evaluator, f)
    f.close()
    # save au_cdf
    au_cdf_df = DataFrame()
    au_cdf_df["estimator"] = list(au_cdf.keys())
    au_cdf_df["AU-CDF"] = list(au_cdf.values())
    au_cdf_df["AU-CDF(scaled)"] = list(au_cdf_scaled.values())
    au_cdf_df.to_csv(
        log_path / f"au_cdf_of_ope_estimators_threshold_{au_cdf_threshold}.csv"
    )
    # save cvar
    cvar_df = DataFrame()
    cvar_df["estimator"] = list(cvar.keys())
    cvar_df["CVaR"] = list(cvar.values())
    cvar_df["CVaR(scaled)"] = list(cvar_scaled.values())
    cvar_df.to_csv(log_path / f"cvar_of_ope_estimators_alpha_{cvar_alpha}.csv")
    # save variance
    std_df = DataFrame()
    std_df["estimator"] = list(std.keys())
    std_df["std"] = list(std.values())
    std_df["std(scaled)"] = list(std_scaled.values())
    std_df.to_csv(log_path / "std_of_ope_estimators.csv")
    # save mean
    mean_df = DataFrame()
    mean_df["estimator"] = list(mean.keys())
    mean_df["mean"] = list(mean.values())
    mean_df["mean(scaled)"] = list(mean_scaled.values())
    mean_df.to_csv(log_path / "mean_of_ope_estimators.csv")
    # printout result
    print(au_cdf_df)
    print(cvar_df)
    print(std_df)
    # save cdf plot
    evaluator.visualize_cdf_aggregate(
        fig_dir=log_path, fig_name="cdf_full.png", font_size=16
    )
