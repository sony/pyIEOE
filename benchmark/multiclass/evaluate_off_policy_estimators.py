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
from sklearn.calibration import CalibratedClassifierCV

from obp.dataset import MultiClassToBanditReduction
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

# parameters for behavior policy and candidate evaluation policies
from conf import base_eval_policy_params

# preprocess for dataset
from util.preprocess import preprocess_and_get_dataset


# load and preprocess datasets
filepath = "data/"
optdigits, pendigits, satimage = preprocess_and_get_dataset(filepath)
# dict for datasets
dataset_dict = {
    "optdigits": optdigits,
    "pendigits": pendigits,
    "satimage": satimage,
}

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
        default=500,
        help="number of seeds used in the experiment",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["optdigits", "pendigits", "satimage"],
        required=True,
        help="the name of the multi-class classification dataset",
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
        "--use_calibration",
        type=strtobool,
        default=False,
        help="whether to use calibration for pscore estimation or not, only available when use_random_search=True",
    )
    parser.add_argument(
        "--eval_size",
        type=float,
        default=0.7,
        help="the proportion of the dataset to include in the evaluation split",
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
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations
    n_seeds = args.n_seeds
    dataset_name = args.dataset_name
    use_random_search = args.use_random_search
    use_estimated_pscore = args.use_estimated_pscore
    use_calibration = args.use_calibration
    eval_size = args.eval_size
    au_cdf_threshold = args.au_cdf_threshold
    cvar_alpha = args.cvar_alpha
    random_state = args.random_state
    np.random.seed(random_state)
    # assertion
    assert 0 < eval_size < 1
    assert 0 < au_cdf_threshold
    assert 0 < cvar_alpha < 100
    assert not (not (use_estimated_pscore and use_random_search) and use_calibration)

    print("initializing experimental condition..")

    # load raw data
    X, y = dataset_dict[dataset_name]
    # convert the raw classification data into a logged bandit dataset
    dataset = MultiClassToBanditReduction(
        X=X,
        y=y,
        base_classifier_b=base_eval_policy_params.behavior_policy.base_classifier,
        alpha_b=base_eval_policy_params.behavior_policy.alpha,
        dataset_name=dataset_name,
    )
    # split the original data into the training and evaluation sets
    dataset.split_train_eval(eval_size=eval_size, random_state=random_state)
    # obtain logged bandit feedback generated by the behavior policy
    bandit_feedback = dataset.obtain_batch_bandit_feedback(random_state=random_state)

    # obtain action choice probabilities and the ground-truth policy value for each evaluation policies
    evaluation_policies = []
    for eval_policy in base_eval_policy_params.evaluation_policies:
        action_dist_e = dataset.obtain_action_dist_by_eval_policy(
            base_classifier_e=eval_policy.base_classifier, alpha_e=eval_policy.alpha
        )
        ground_truth_e = dataset.calc_ground_truth_policy_value(
            action_dist=action_dist_e
        )
        evaluation_policies.append((ground_truth_e, action_dist_e))

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
        regressor_hyperparams = None

    else:  # uniform sampling
        regression_models = [
            LogisticRegression,
            RandomForest,
            LightGBM,
        ]
        regressor_hyperparams = {
            LogisticRegression: ope_regression_uniform_hyperparams.logistic_regression_param,
            RandomForest: ope_regression_uniform_hyperparams.random_forest_param,
            LightGBM: ope_regression_uniform_hyperparams.lightgbm_param,
        }

    # pscore estimator
    if use_estimated_pscore:
        if use_random_search and use_calibration:
            pscore_estimation_models = [
                CalibratedClassifierCV(
                    base_estimator=regression_models[i],
                    cv=2,
                )
                for i in range(len(regression_models))
            ]
        else:
            pscore_estimation_models = regression_models

        pscore_estimator_hyperparams = regressor_hyperparams

    # initializing class
    if use_estimated_pscore:
        if use_random_search:
            evaluator = InterpretableOPEEvaluator(
                random_states=np.arange(n_seeds),
                bandit_feedbacks=[bandit_feedback],
                evaluation_policies=evaluation_policies,
                ope_estimators=ope_estimators,
                ope_estimator_hyperparams=ope_estimator_hyperparams_,
                regression_models=regression_models,
                pscore_estimators=pscore_estimation_models,
            )

        else:  # uniform sampling
            evaluator = InterpretableOPEEvaluator(
                random_states=np.arange(n_seeds),
                bandit_feedbacks=[bandit_feedback],
                evaluation_policies=evaluation_policies,
                ope_estimators=ope_estimators,
                ope_estimator_hyperparams=ope_estimator_hyperparams_,
                regression_models=regression_models,
                regression_model_hyperparams=regressor_hyperparams,
                pscore_estimators=pscore_estimation_models,
                pscore_estimator_hyperparams=pscore_estimator_hyperparams,
            )

    else:  # ground-truth pscore
        if use_random_search:
            evaluator = InterpretableOPEEvaluator(
                random_states=np.arange(n_seeds),
                bandit_feedbacks=[bandit_feedback],
                evaluation_policies=evaluation_policies,
                ope_estimators=ope_estimators,
                ope_estimator_hyperparams=ope_estimator_hyperparams_,
                regression_models=regression_models,
            )

        else:  # uniform sampling
            evaluator = InterpretableOPEEvaluator(
                random_states=np.arange(n_seeds),
                bandit_feedbacks=[bandit_feedback],
                evaluation_policies=evaluation_policies,
                ope_estimators=ope_estimators,
                ope_estimator_hyperparams=ope_estimator_hyperparams_,
                regression_models=regression_models,
                regression_model_hyperparams=regressor_hyperparams,
            )

    # estimate policy values
    print("started experiment")
    policy_value = evaluator.estimate_policy_value()
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
            if use_calibration:
                option = "/rscv_pscore_estimate_calibration"
            else:
                option = "/rscv_pscore_estimate"
        else:
            option = "/rscv_pscore_true"
    else:
        if use_estimated_pscore:
            if use_calibration:
                option = "/uniform_pscore_estimate_calibration"
            else:
                option = "/uniform_pscore_estimate"
        else:
            option = "/uniform_pscore_true"

    # save results of the evaluation of off-policy estimators
    # in './logs/(dataset_name)/(option)' directory.
    log_path = Path("./logs/" + dataset_name + option)
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
