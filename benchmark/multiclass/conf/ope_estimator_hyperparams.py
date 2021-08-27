# hyperparam space for OPE estimators
import numpy as np

tau_lambda = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, np.inf]
K = {"lower": 1, "upper": 5, "log": False, "type": int}

dm_param = {"K": K}
dr_param = {"K": K}
sndr_param = {"K": K}
switch_dr_param = {"K": K}
dros_param = {"K": K}
