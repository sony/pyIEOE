# Copyright (c) 2021 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

from abc import ABCMeta, abstractmethod


class BaseOPEEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def estimate_policy_value(self) -> None:
        """Estimate policy values."""
        raise NotImplementedError

    @abstractmethod
    def calculate_squared_error(self) -> None:
        """Calculate squared errors."""
        raise NotImplementedError

    @abstractmethod
    def visualize_cdf(self) -> None:
        """Create graph of cumulative distribution function of an estimator."""
        raise NotImplementedError

    @abstractmethod
    def visualize_cdf_aggregate(self) -> None:
        """Create graph of cumulative distribution function of all estimators."""
        raise NotImplementedError

    @abstractmethod
    def save_policy_value(self) -> None:
        """Save estimate policy values to csv file."""
        raise NotImplementedError

    @abstractmethod
    def save_squared_error(self) -> None:
        """Save squared errors to csv file."""
        raise NotImplementedError

    @abstractmethod
    def calculate_au_cdf_score(self) -> None:
        """Calculate AU-CDF score."""
        raise NotImplementedError

    @abstractmethod
    def calculate_cvar_score(self) -> None:
        """Calculate CVaR score."""
        raise NotImplementedError
