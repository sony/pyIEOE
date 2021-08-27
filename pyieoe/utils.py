# Copyright (c) 2021 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

from typing import Union

import numpy as np
from scipy.stats import loguniform


def _choose_uniform(
    s: int,
    lower: Union[int, float],
    upper: Union[int, float],
    type_: type,
) -> Union[int, float]:
    np.random.seed(seed=s)
    assert lower <= upper, "`upper` must be larger than or equal to `lower`"
    assert type_ in [int, float], f"`type_` must be int or float but {type_} is given"
    if lower == upper:
        return lower
    if type_ == int:
        return np.random.randint(lower, upper, dtype=type_)
    else:  # type_ == float:
        return np.random.uniform(lower, upper)


def _choose_log_uniform(
    s: int,
    lower: Union[int, float],
    upper: Union[int, float],
    type_: type,
) -> Union[int, float]:
    assert (
        lower > 0
    ), f"`lower` must be greater than 0 when drawing from log uniform distribution but {lower} is given"
    assert lower <= upper, "`upper` must be larger than or equal to `lower`"
    assert type_ in [int, float], f"`type_` must be int or float but {type_} is given"
    if lower == upper:
        return lower
    if type_ == int:
        return int(loguniform.rvs(lower, upper, random_state=s))
    else:  # type_ == float:
        return loguniform.rvs(lower, upper, random_state=s)
