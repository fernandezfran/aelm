#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of aelm (https://github.com/fernandezfran/aelm/).
# Copyright (c) 2022, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/aelm/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""This file is for distribute and install aelm."""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

from setuptools import find_packages, setup

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

REQUIREMENTS = [
    "exma",
    "numpy",
    "pandas",
]

VERSION = "0.0.1"

with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()


# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="aelm",
    version=VERSION,
    description="Perform arrhenius plots for diffusion coefficients",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Francisco Fernandez",
    author_email="fernandezfrancisco2195@gmail.com",
    url="https://github.com/fernandezfran/aelm",
    packages=find_packages(),
    license="The MIT License",
    install_requires=REQUIREMENTS,
    keywords=["aelm", "energy-minimization", "molecular-dynamics"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)
