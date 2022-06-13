#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of aelm (https://github.com/fernandezfran/aelm/).
# Copyright (c) 2022, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/aelm/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import contextlib
import os
import pathlib
from unittest import mock

from aelm import aelm

import numpy as np

import pandas as pd

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)

with open(TEST_DATA / "log.lammps", "r") as f:
    MOCK_LOG = f.read().split("\n")
ATTRS = {"stdout.split.return_value": MOCK_LOG}


# =============================================================================
# TESTS
# =============================================================================
# stackoverflow: /questions/66332005/how-to-mock-subprocess-run-in-pytest


@mock.patch("aelm.subprocess.run")
def test_aelm_raise(mock_run):
    """Test the raise of a RuntimeError."""
    with pytest.raises(RuntimeError):
        mock_stdout = mock.MagicMock()
        mock_stdout.configure_mock(
            **{"stdout.split.return_value": "A log\n with errors".split("\n")}
        )

        mock_run.return_value = mock_stdout

        aelm(
            TEST_DATA / "test.xyz",
            TEST_DATA / "dump.all.lammpstrj",
            {"box": np.full(3, 10.60908684634919), "type": {"Si": 1, "Li": 2}},
            lmp_min=TEST_DATA / "dump.minimization.lammpstrj",
            lmp_data=TEST_DATA / "in.frame",
            rm_tmp=False,
        )


@mock.patch("aelm.subprocess.run")
def test_aelm_rm_tmp(mock_run):
    """Test the aelm rm tmp files."""
    mock_stdout = mock.MagicMock()
    mock_stdout.configure_mock(**ATTRS)

    mock_run.return_value = mock_stdout

    os.system(f"cp {TEST_DATA / 'log.*'} .")

    aelm(
        TEST_DATA / "test.xyz",
        TEST_DATA / "dump.all.lammpstrj",
        {"box": np.full(3, 10.60908684634919), "type": {"Si": 1, "Li": 2}},
        lmp_min=TEST_DATA / "rm_data" / "dump.minimization.lammpstrj",
        lmp_data=TEST_DATA / "rm_data" / "in.frame",
    )

    rmdir = TEST_DATA / "rm_data"
    os.system(f"cp {TEST_DATA / 'in.frame'} {rmdir}")
    os.system(f"cp {TEST_DATA / 'dump.minimization.lammpstrj'} {rmdir}")

    ls = os.popen("ls").read().split("\n")

    assert ("in.frame" in ls) is False


@mock.patch("aelm.subprocess.run")
def test_aelm_df(mock_run):
    """Test the aelm returned pd.DataFrame."""
    df_ref = pd.DataFrame(
        {
            "initial": np.asarray([-4426.57531107183], dtype=np.float32),
            "next_to_last": np.asarray([-4810.79048271336], dtype=np.float32),
            "final": np.asarray([-4810.79047946943], dtype=np.float32),
        }
    )

    mock_stdout = mock.MagicMock()
    mock_stdout.configure_mock(**ATTRS)

    mock_run.return_value = mock_stdout

    result = aelm(
        TEST_DATA / "test.xyz",
        TEST_DATA / "dump.all.lammpstrj",
        {"box": np.full(3, 10.60908684634919), "type": {"Si": 1, "Li": 2}},
        lmp_min=TEST_DATA / "dump.minimization.lammpstrj",
        lmp_data=TEST_DATA / "in.frame",
        rm_tmp=False,
    )
    os.remove(TEST_DATA / "dump.all.lammpstrj")

    pd.testing.assert_frame_equal(result, df_ref)


@mock.patch("aelm.subprocess.run")
def test_aelm_df_flags(mock_run):
    """Test the aelm returned pd.DataFrame with lmp extra flags."""
    df_ref = pd.DataFrame(
        {
            "initial": np.asarray([-4426.57531107183], dtype=np.float32),
            "next_to_last": np.asarray([-4810.79048271336], dtype=np.float32),
            "final": np.asarray([-4810.79047946943], dtype=np.float32),
        }
    )

    mock_stdout = mock.MagicMock()
    mock_stdout.configure_mock(**ATTRS)

    mock_run.return_value = mock_stdout

    result = aelm(
        TEST_DATA / "test.xyz",
        TEST_DATA / "dump.all.lammpstrj",
        {"box": np.full(3, 10.60908684634919), "type": {"Si": 1, "Li": 2}},
        lmp_min=TEST_DATA / "dump.minimization.lammpstrj",
        lmp_data=TEST_DATA / "in.frame",
        lmp_flags={"sf": "omp"},
        rm_tmp=False,
    )
    os.remove(TEST_DATA / "dump.all.lammpstrj")

    pd.testing.assert_frame_equal(result, df_ref)


@mock.patch("aelm.subprocess.run")
def test_aelm_dump(mock_run):
    """Test the aelm dump.lammpstrj generated."""
    mock_stdout = mock.MagicMock()
    mock_stdout.configure_mock(**ATTRS)

    mock_run.return_value = mock_stdout

    aelm(
        TEST_DATA / "test.xyz",
        TEST_DATA / "dump.all.lammpstrj",
        {"box": np.full(3, 10.60908684634919), "type": {"Si": 1, "Li": 2}},
        lmp_min=TEST_DATA / "dump.minimization.lammpstrj",
        lmp_data=TEST_DATA / "in.frame",
        rm_tmp=False,
    )

    filenames = ["dump.lammpstrj", "dump.all.lammpstrj"]
    with contextlib.ExitStack() as stack:
        files = [
            stack.enter_context(open(TEST_DATA / f, "r")) for f in filenames
        ]
        frames = [f.read() for f in files]
    os.remove(TEST_DATA / "dump.all.lammpstrj")

    assert frames[1] == frames[0]
