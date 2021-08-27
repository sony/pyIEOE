# Benchmark with Multi-class Classification Data
## Description
Here, we use multi-class classification datasets to evaluate OPE estimators with IEOE procedure. Specifically, we evaluate the estimation performances of well-known off-policy estimators using the ground-truth policy value of an evaluation policy calculable with multi-class classification data. We use three classification datasets, OptDigits, PenDigits, SatImage from [UCI repository](https://archive.ics.uci.edu/ml/index.php).

## Evaluating Off-Policy Estimators
In the following, we evaluate the estimation performances of
- Direct Method (DM)
- Inverse Probability Weighting (IPWps)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DRps)
- Self-Normalized Doubly Robust (SNDR)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)

For DM, DRps, SNDR, Switch-DR and DRos, we randomly choose different values of hyperparameters. See [obp's documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details about these estimators and `./conf/ope_estimator_hyperparams.py` and `./conf/ope_regression_uniform_hyperparams.py` for the hyperparameter space we set for the experiment.

### Files
- `./evaluate_off_policy_estimators.py` implements the evaluation of OPE estimators using multi-class classification data with IEOE procedure.
- `./recalculate_au_cdf_and_cvar.py` implements the recalculation of performance indicator for OPE estimators, namely AU-CDF and CVaR with different hyperparameters.
- `./conf/ope_estimator_hyperparams.py` defines hyperparameters space for some ope estimators and machine learning methods used to define regression model.
- `./conf/base_eval_policy_params.py` defines behavior policy and a set of evaluation policies used in the experiment.
- `./util/preprocess.py` implements preprocessing for the classification datasets.
- `./data/*` should contain datasets, `optdigits.tra`, `optdigits.tes`, `pendigits.tra`, `pendigits.tes`, `satimage.trn`, `satimage.tst`. Get datasets from [UCI repository](https://archive.ics.uci.edu/ml/index.php).
- `./logs/(dataset_name)/(option)/*` stores the result of experiment. Option is either `uniform_pscore_true`, `uniform_pscore_estimate`, `rscv_pscore_true`, `rscv_pscore_estimate`, or `rscv_pscore_calibration`. `cdf_full.png`, `au_cdf_of_ope_estimators_threshold_{au_cdf_threshold}.csv`, `cvar_of_ope_estimators_alpha_{cvar_alpha}.csv`, `variance_of_ope_estimators.csv`, `mean_of_ope_estimators.csv`, and `evaluator.pickle` will be saved.

### Scripts
#### evaluate_off_policy_estimators.py
```
# run evaluation of OPE estimators with multi-class classification data
python evaluate_off_policy_estimators.py \
    --n_seeds $n_seeds\
    --dataset_name $dataset_name \
    --use_random_search $use_random_search \
    --use_estimated_pscore $use_estimated_pscore \
    --use_calibration $ $use_calibration \
    --eval_size $eval_size \
    --au_cdf_threshold $au_cdf_threshold \
    --cvar_alpha $cvar_alpha \
    --random_state $random_state
```
- `$n_seeds` specifies the number of seeds used in IEOE procedure to estimate the performance of OPE estimators.
- `$dataset_name` specifies the name of the multi-class classification dataset and should be one of "optdigits", "pendigits", or "satimage".
- `$use_random_search` is boolean which specifies whether to use RandomizedSearchCV for hyperparameter sampling for the reward estimator and the behavior policy (pscore) estimator. Otherwise, the uniform sampling is used.
- `$use_estimated_pscore` is boolean which specifies whether to use estimated behavior policy (pscore). Otherwise, the true pscore is used.
- `$use_calibration` is boolean which specifies whether to use calibration for behavior policy (pscore) estimation. This option is available only when use_estimated_pscore=True and use_random_search=True.
- `$eval_size` specifies the proportion of the dataset to include in the evaluation split.
- `$au_cdf_threshold` specifies the threshold (the maximum error allowed, z_max) used for calculating AU-CDF.
- `$cvar_alpha` specifies the percentile used for calculating CVaR.
- `$random_state` specifies random_states used for preparation of dataset. Note that we also need to specify random_state in `./conf/base_eval_policy_params.py`.
- We also specify hyperparameter space for each OPE estimators in `./conf/ope_estimator_hyperparams.py` and for regression model used in some OPE estimators either in `./conf/ope_regression_uniform_hyperparams.py` or `./conf/ope_regression_rscv_hyperparams.py`, and behavior and evaluation policies in `./conf/base_eval_policy_params.py`.

For example, the following command compares the estimation performances of the OPE estimators with IEOE procedure using the Optdigits dataset. The detailed result (plot of cdf of squared error, dataframe of AU-CDF, CVaR, variance, and mean for each estimators will be saved in `./logs/(dataset_name)/(option)/`. The evaluator instance will also be saved in order to recalculate AU-CDF and CVaR with different hyperparameters afterwhile.)
```
python evaluate_off_policy_estimators.py \
    --n_seeds 500 \
    --dataset_name optdigits \
    --use_random_search False \
    --use_estimated_pscore False \
    --use_calibration False \
    --eval_size 0.7 \
    --au_cdf_threshold 0.001 \
    --cvar_alpha 70 \
    --random_state 12345

# AU-CDF (higher is better), CVaR (lower is better), and variance (lower is better) of OPE estimators.
# It seems estimators without any hyperparameter (IPWps, SNIPW) has more reliable performance than advanced estimators.
# =============================================
# random_state=12345
# ---------------------------------------------
#    estimator        AU-CDF    AU-CDF(scaled)
# 0      IPWps      0.000925          1.000000
# 1      SNIPW      0.000839          0.907351
# 2         DM      0.000000          0.000000
# 3       DRps      0.000230          0.248582
# 4       SNDR      0.000346          0.374142
# 5  Switch-DR      0.000266          0.287434
# 6       DRos      0.000266          0.287434
#    estimator          CVaR      CVaR(scaled)
# 0      IPWps      0.000209          1.000000
# 1      SNIPW      0.000524          2.506530
# 2         DM      0.463378       2215.365312
# 3       DRps      0.026691        127.609319
# 4       SNDR      0.020271         96.913620
# 5  Switch-DR      0.026399        126.210624
# 6       DRos      0.026399        126.210624
#    estimator      variance  variance(scaled)
# 0      IPWps  1.465258e-08      1.000000e+00
# 1      SNIPW  1.457859e-07      9.949501e+00
# 2         DM  3.898552e-02      2.660659e+06
# 3       DRps  2.871548e-04      1.959755e+04
# 4       SNDR  2.323406e-04      1.585663e+04
# 5  Switch-DR  2.835910e-04      1.935434e+04
# 6       DRos  2.835910e-04      1.935434e+04
# =============================================
```

#### recalculate_au_cdf_and_cvar.py
```
# recalculation of AU-CDF and CVaR with different parameters and re-plot of CDF with specified xmax.
python recalculate_au_cdf_and_cvar.py \
    --dataset_name optdigits
    --use_random_search False \
    --use_estimated_reward False \
    --use_calibration False \
    --cdf_plot_xmax $cdf_plot_xmax \
    --au_cdf_threshold $au_cdf_threshold \
    --cvar_alpha $cvar_alpha \
```
- `$dataset_name` specifies the name of the multi-class classification dataset and should be one of "optdigits", "pendigits", or "satimage".
- `$use_random_search` is boolean which specifies whether to use RandomizedSearchCV for hyperparameter sampling for the reward estimator and the behavior policy (pscore) estimator. Otherwise, the uniform sampling is used.
- `$use_estimated_pscore` is boolean which specifies whether to use estimated behavior policy (pscore). Otherwise, the true pscore is used.
- `$use_calibration` is boolean which specifies whether to use calibration for behavior policy (pscore) estimation. This option is available only when use_estimated_pscore=True and use_random_search=True.
- `$cdf_plot_xmax` specifies the maximum error to be shown in CDF plot.
- `$au_cdf_threshold` specifies the threshold (the maximum error allowed, z_max) used for calculating AU-CDF.
- `$cvar_alpha` specifies the percentile used for calculating CVaR.

The result will be saved in `./logs/(dataset_name)/(option)/` with name `cdf_xmax_{cdf_plot_xmax}.png`, `au_cdf_of_ope_estimators_threshold_{au_cdf_threshold}.csv` and `cvar_of_ope_estimators_alpha_{cvar_alpha}.csv`.
