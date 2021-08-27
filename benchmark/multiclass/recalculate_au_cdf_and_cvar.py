# Copyright (c) 2021 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import argparse
from distutils.util import strtobool
from pathlib import Path
import pickle

import warnings

warnings.simplefilter("ignore")

from pandas import DataFrame


parser = argparse.ArgumentParser(
    description="recalculate off-policy estimators' performance with au_cdf_threshold and cvar_alpha"
)
parser.add_argument(
    "--dataset_name",
    type=str,
    choices=["optdigits", "pendigits", "satimage"],
    required=True,
    help="the name of the multi-class classification dataset.",
)
parser.add_argument(
    "--cdf_plot_xmax",
    type=float,
    required=True,
    help="the maximum error to be shown in CDF plot (x_max)",
)
parser.add_argument(
    "--au_cdf_threshold",
    type=float,
    required=True,
    help="threshold (the maximum error allowed, z_max) for AU-CDF",
)
parser.add_argument(
    "--cvar_alpha",
    type=int,
    required=True,
    help="the percentile used for calculating CVaR, should be in (0, 100)",
)
parser.add_argument(
    "--use_random_search",
    type=strtobool,
    default=False,
    help="whether to use random search for hyperparamter selection or not, otherwise unform sampling is used",
)
parser.add_argument(
    "--use_estimated_pscore",
    type=strtobool,
    default=False,
    help="wether to use estimated pscore or not, otherwise ground-truth pscore is used",
)
parser.add_argument(
    "--use_calibration",
    type=strtobool,
    default=False,
    help="whether to use calibration for pscore estimation or not",
)
args = parser.parse_args()
print(args)

# configurations
dataset_name = args.dataset_name
cdf_plot_xmax = args.cdf_plot_xmax
au_cdf_threshold = args.au_cdf_threshold
cvar_alpha = args.cvar_alpha
use_random_search = args.use_random_search
use_estimated_pscore = args.use_estimated_pscore
use_calibration = args.use_calibration
# assertion
assert 0 < cdf_plot_xmax
assert 0 < au_cdf_threshold
assert 0 < cvar_alpha < 100

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
# path
log_path = Path("./logs/" + dataset_name + option)
# load evaluator
f = open(log_path / "evaluator.pickle", "rb")
evaluator = pickle.load(f)
f.close()
# recalculation of au_cdf and cvar
print("calculating statistics of estimators' performance..")
au_cdf = evaluator.calculate_au_cdf_score(threshold=au_cdf_threshold)
au_cdf_scaled = evaluator.calculate_au_cdf_score(threshold=au_cdf_threshold, scale=True)
cvar = evaluator.calculate_cvar_score(alpha=cvar_alpha)
cvar_scaled = evaluator.calculate_cvar_score(alpha=cvar_alpha, scale=True)
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
# printout result
print(au_cdf_df)
print(cvar_df)
# save cdf plot
evaluator.visualize_cdf_aggregate(
    fig_dir=log_path,
    fig_name=f"cdf_xmax_{cdf_plot_xmax}.png",
    font_size=16,
    xmax=cdf_plot_xmax,
)
