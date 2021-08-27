# Copyright (c) 2021 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

from pyieoe.version import __version__
from setuptools import setup, find_packages
from os import path
import sys

here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, "pyieoe"))

print(f"version: {__version__}")

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyieoe",
    version=__version__,
    description="pyIEOE: a Python package to facilitate interpretable OPE evaluation",
    url="https://github.com/sony/pyIEOE",
    keywords=["off-policy evaluation"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "matplotlib>=3.3.2",
        "numpy>=1.20.1",
        "obp>=0.4.1",
        "pandas>=1.2.3",
        "seaborn>=0.11.1",
        "scikit-learn>=0.24.1",
        "scipy>=1.6.0",
        "tqdm>=4.56.0",
    ],
    license="MIT License",
    packages=find_packages(exclude=["benchmark", "examples"]),
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
