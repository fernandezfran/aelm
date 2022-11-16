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

from aelm import AELM

import exma

import numpy as np

import pandas as pd

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)
TEST_DATA_LAMMPS = TEST_DATA / "LAMMPS"
TEST_DATA_GEMS = TEST_DATA / "GEMS"

with open(TEST_DATA_LAMMPS / "log.lammps", "r") as f:
    MOCK_LOG_LAMMPS = f.read().split("\n")
ATTRS_LAMMPS = {"stdout.split.return_value": MOCK_LOG_LAMMPS}


# =============================================================================
# TESTS
# =============================================================================
# stackoverflow: /questions/66332005/how-to-mock-subprocess-run-in-pytest


# # GENERAL TESTS
def test_aelm_ve_raise():
    """Test the raise of a ValueError."""
    with pytest.raises(ValueError):
        AELM("DUMMY_PROGRAM", ["dummy.traj"], {"dummy": "info"})


@mock.patch("aelm.subprocess.run")
def test_aelm_re_raise(mock_run):
    """Test the raise of a RuntimeError."""
    with pytest.raises(RuntimeError):
        mock_stdout = mock.MagicMock()
        mock_stdout.configure_mock(
            **{"stdout.split.return_value": "A log\n with errors".split("\n")}
        )

        mock_run.return_value = mock_stdout

        biased_traj = exma.read_xyz(TEST_DATA_LAMMPS / "test.xyz")

        obj = AELM(
            "LAMMPS",
            biased_traj,
            {"box": np.full(3, 10.60908684634919), "type": {"Si": 1, "Li": 2}},
        )
        obj.run(
            "./lmp -in in.minimization",
            TEST_DATA_LAMMPS / "log.lammps",
            TEST_DATA_LAMMPS / "in.frame",
            TEST_DATA_LAMMPS / "dump.minimization.lammpstrj",
        )


# # TESTS WITH LAMMPS
@mock.patch("aelm.subprocess.run")
def test_aelm_df_lammps(mock_run):
    """Test the aelm returned pd.DataFrame."""
    df_ref = pd.DataFrame(
        {
            "initial": np.asarray([-4426.57531107183], dtype=np.float32),
            "final": np.asarray([-4810.79047946943], dtype=np.float32),
        }
    )

    mock_stdout = mock.MagicMock()
    mock_stdout.configure_mock(**ATTRS_LAMMPS)

    mock_run.return_value = mock_stdout

    biased_traj = exma.read_xyz(TEST_DATA_LAMMPS / "test.xyz")

    obj = AELM(
        "LAMMPS",
        biased_traj,
        {"box": np.full(3, 10.60908684634919), "type": {"Si": 1, "Li": 2}},
    )
    obj.run(
        "./lmp -in in.minimization",
        TEST_DATA_LAMMPS / "log.lammps",
        TEST_DATA_LAMMPS / "in.frame",
        TEST_DATA_LAMMPS / "dump.minimization.lammpstrj",
    )
    result = obj.energies

    pd.testing.assert_frame_equal(result, df_ref)


@mock.patch("aelm.subprocess.run")
def test_aelm_dump_lammps(mock_run):
    """Test the aelm dump.lammpstrj generated."""
    mock_stdout = mock.MagicMock()
    mock_stdout.configure_mock(**ATTRS_LAMMPS)

    mock_run.return_value = mock_stdout

    biased_traj = exma.read_xyz(TEST_DATA_LAMMPS / "test.xyz")

    obj = AELM(
        "LAMMPS",
        biased_traj,
        {"box": np.full(3, 10.60908684634919), "type": {"Si": 1, "Li": 2}},
    )
    obj.run(
        "./lmp -in in.minimization",
        TEST_DATA_LAMMPS / "log.lammps",
        TEST_DATA_LAMMPS / "in.frame",
        TEST_DATA_LAMMPS / "dump.minimization.lammpstrj",
    )
    exma.write_lammpstrj(
        obj.minimized_frames, TEST_DATA_LAMMPS / "dump.all.lammpstrj"
    )

    filenames = ["dump.lammpstrj", "dump.all.lammpstrj"]
    with contextlib.ExitStack() as stack:
        files = [
            stack.enter_context(open(TEST_DATA_LAMMPS / f, "r"))
            for f in filenames
        ]
        frames = [f.read() for f in files]
    os.remove(TEST_DATA_LAMMPS / "dump.all.lammpstrj")

    assert frames[1] == frames[0]


# # TESTS WITH GEMS
@mock.patch("aelm.subprocess.run")
def test_aelm_df_gems(mock_run):
    """Test the aelm returned pd.DataFrame."""
    df_ref = pd.DataFrame(
        {
            "initial": np.asarray([-796.328455], dtype=np.float32),
            "final": np.asarray([-799.493874], dtype=np.float32),
        }
    )

    biased_traj = exma.read_xyz(TEST_DATA_GEMS / "test.xyz")

    obj = AELM("GEMS", biased_traj, {"box": np.full(3, 10.566048)})
    obj.run(
        "./gems lbfgs.gms",
        TEST_DATA_GEMS / "lbfgs.log",
        TEST_DATA_GEMS / "to_min.xyz",
        TEST_DATA_GEMS / "traj.lbfgs.xyz",
    )
    result = obj.energies

    pd.testing.assert_frame_equal(result, df_ref)


@mock.patch("aelm.subprocess.run")
def test_aelm_traj_gems(mock_run):
    """Test the aelm all_gen.xyz generated."""
    biased_traj = exma.read_xyz(TEST_DATA_GEMS / "test.xyz")

    obj = AELM("GEMS", biased_traj, {"box": np.full(3, 10.566048)})
    obj.run(
        "./gems lbfgs.gms",
        TEST_DATA_GEMS / "lbfgs.log",
        TEST_DATA_GEMS / "to_min.xyz",
        TEST_DATA_GEMS / "traj.lbfgs.xyz",
    )
    exma.write_xyz(obj.minimized_frames, TEST_DATA_GEMS / "all_gen.xyz")

    filenames = ["all.xyz", "all_gen.xyz"]
    with contextlib.ExitStack() as stack:
        files = [
            stack.enter_context(open(TEST_DATA_GEMS / f, "r"))
            for f in filenames
        ]
        frames = [f.read() for f in files]
    os.remove(TEST_DATA_GEMS / "all_gen.xyz")

    assert frames[1] == frames[0]
