# Benchmark with Open Bandit Dataset
## Description
Here, we use [Open Bandit Dataset (OBD)](https://research.zozo.com/data.html) to evaluate OPE estimators with IEOE procedure. Specifically, we evaluate the estimation performances of well-known off-policy estimators using the ground-truth policy value of an evaluation policy calculable with OBD.

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
- `./evaluate_off_policy_estimators.py` implements the evaluation of OPE estimators using OBD with IEOE procedure.
- `./recalculate_au_cdf_and_cvar.py` implements the recalculation of performance indicator for OPE estimators, namely AU-CDF and CVaR with different hyperparameters.
- `./conf/ope_estimator_hyperparams.py` defines hyperparameters space for some ope estimators and machine learning methods used to define regression model.
- `./open_bandit_dataset/*` should contain full size OBD. Get dataset from [here](https://research.zozo.com/data.html) (optional). Required when `--is_full_obd True`.
- `./obd/*` should contain light version of OBD, which is available [here](https://github.com/st-tech/zr-obp/tree/master/obd). Either full or light version of OBD is required to the experiment.
- `./logs/(option)/*` stores the result of experiment. Option is either `uniform_pscore_true`, `uniform_pscore_estimate`, `rscv_pscore_true`, or `rscv_pscore_estimate`. `cdf_full.png`, `au_cdf_of_ope_estimators_threshold_{au_cdf_threshold}.csv`, `cvar_of_ope_estimators_alpha_{cvar_alpha}.csv`, `variance_of_ope_estimators.csv`, `mean_of_ope_estimators.csv`, and `evaluator.pickle` will be saved.

### Scripts
#### evaluate_off_policy_estimators.py
```
# run evaluation of OPE estimators with OBD
python evaluate_off_policy_estimators.py \
    --n_seeds $n_seeds\
    --use_random_search $use_random_search \
    --use_estimated_pscore $use_estimated_pscore \
    --au_cdf_threshold $au_cdf_threshold \
    --cvar_alpha $cvar_alpha \
    --campaign $campaign \
    --is_full_obd $is_full_obd \
    --sample_size $sample_size \
    --random_state $random_state
```
- `$n_seeds` specifies the number of seeds used in IEOE procedure to estimate the performance of OPE estimators.
- `$use_random_search` is boolean which specifies whether to use RandomizedSearchCV for hyperparameter sampling for the reward estimator and the behavior policy (pscore) estimator. Otherwise, the uniform sampling is used.
- `$use_estimated_pscore` is boolean which specifies whether to use estimated behavior policy (pscore). Otherwise, the true pscore is used.
- `$au_cdf_threshold` specifies the threshold (the maximum error allowed, z_max) used for calculating AU-CDF.
- `$cvar_alpha` specifies the percentile used for calculating CVaR.
- `$campaign` specifies the campaign of the dataset and should be one of "all", "men" and "women".
- `$is_full_obd` specifies whether to use full size of OBD or not and should be boolean. 
- `$sample_size` specifies (maximum) sample size of the dataset to be used in the experiment and should be more than 10000.
- `$random_state` specifies random_states used for preparation of dataset.
- We also specify hyperparameter space for each OPE estimators in `./conf/ope_estimator_hyperparams.py` and for regression model used in some OPE estimators either in `./conf/ope_regression_uniform_hyperparams.py` or `./conf/ope_regression_rscv_hyperparams.py`, and behavior and evaluation policies in `./conf/base_eval_policy_params.py`.

For example, the following command compares the estimation performances of the OPE estimators with IEOE procedure. The detailed result (plot of cdf of squared error, dataframe of AU-CDF, CVaR, variance, and mean for each estimators will be saved in `./logs/(option)/`. The evaluator instance will also be saved in order to recalculate AU-CDF and CVaR with different hyperparameters afterwhile.)
```
python evaluate_off_policy_estimators.py \
    --n_seeds 500 \
    --au_cdf_threshold 0.000001 \
    --use_random_search False \
    --use_estimated_pscore False \
    --cvar_alpha 70 \
    --campaign all \
    --is_full_obd True \
    --sample_size 300000 \
    --random_state 12345

# AU-CDF (higher is better), CVaR (lower is better), and variance (lower is better) of OPE estimators.
# It seems estimators without any hyperparameter (IPWps, SNIPW) has more reliable performance than advanced estimators.
# =============================================
# random_state=12345
# ---------------------------------------------
#    estimator        AU-CDF    AU-CDF(scaled)
# 0      IPWps  8.422079e-07          1.000000
# 1      SNIPW  7.722167e-07          0.916896
# 2         DM  1.567398e-07          0.186106
# 3       DRps  7.474507e-07          0.887490
# 4       SNDR  6.962037e-07          0.826641
# 5  Switch-DR  6.951286e-07          0.825365
# 6       DRos  6.951286e-07          0.825365
#    estimator          CVaR      CVaR(scaled)
# 0      IPWps  4.243385e-07          1.000000
# 1      SNIPW  6.677742e-07          1.573683
# 2         DM  3.160304e-06          7.447601
# 3       DRps  1.131544e-06          2.666607
# 4       SNDR  1.483420e-06          3.495841
# 5  Switch-DR  1.485016e-06          3.499602
# 6       DRos  1.485016e-06          3.499602
#    estimator      variance  variance(scaled)
# 0      IPWps  4.784457e-14          1.000000
# 1      SNIPW  1.391087e-13          2.907512
# 2         DM  2.306980e-12         48.218234
# 3       DRps  1.203898e-12         25.162686
# 4       SNDR  1.728508e-12         36.127569
# 5  Switch-DR  1.722155e-12         35.994778
# 6       DRos  1.722155e-12         35.994778
# =============================================
```

#### recalculate_au_cdf_and_cvar.py
```
# recalculation of AU-CDF and CVaR with different parameters and re-plot of CDF with specified xmax.
python recalculate_au_cdf_and_cvar.py \
    --use_random_search False \
    --use_estimated_pscore False \
    --cdf_plot_xmax $cdf_plot_xmax \
    --au_cdf_threshold $au_cdf_threshold \
    --cvar_alpha $cvar_alpha \
```
- `$use_random_search` is boolean which specifies whether to use RandomizedSearchCV for hyperparameter sampling for the reward estimator and the behavior policy (pscore) estimator. Otherwise, the uniform sampling is used.
- `$use_estimated_pscore` is boolean which specifies whether to use estimated behavior policy (pscore). Otherwise, the true pscore is used.
- `$cdf_plot_xmax` specifies the maximum error to be shown in CDF plot.
- `$au_cdf_threshold` specifies the threshold (the maximum error allowed, z_max) used for calculating AU-CDF.
- `$cvar_alpha` specifies the percentile used for calculating CVaR.

The result will be saved in `./logs/(option)/` with name `cdf_xmax_{cdf_plot_xmax}.png`, `au_cdf_of_ope_estimators_threshold_{au_cdf_threshold}.csv` and `cvar_of_ope_estimators_alpha_{cvar_alpha}.csv`.
