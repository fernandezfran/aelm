#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of aelm (https://github.com/fernandezfran/aelm/).
# Copyright (c) 2022, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/aelm/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Accelerated exploration of local minima.

This is a fast and effective method to find relevant minima in which an energy
bias function is used to overcome the energetic barriers followed by a local
minimization.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import subprocess

import exma

import numpy as np

import pandas as pd

# ============================================================================
# FUNCTIONS
# ============================================================================


def aelm(
    biased_traj,
    minimizations_output,
    cell_info,
    lmp_exec="./lmp",
    lmp_in="in.minimization",
    lmp_data="in.frame",
    lmp_min="dump.minimization.lammpstrj",
    lmp_flags=None,
    verbose=True,
):
    """Accelerated exploration of local minima minimization.

    This function takes care of the local minimization of AELM after obtaining
    the biased trajectory [1]_.

    Parameters
    ----------
    biased_traj : str
        filename of the xyz file with the biased trajectory

    minimizations_output : str
        filename where the last frame of each minimization will be written

    cell_info : dict
        with the `box`, the lenght of the box in each direction, another
        dictionary identified with the `type` key that has within it a
        correspondence between the elements present in xyz file with integer
        identification numbers, e.g. {"Si": 1, "Li": 2}

    lmp_exec : str, default="./lmp"
        the path and name of the LAMMPS executable

    lmp_in : str, default="in.minimization"
        the input file for LAMMPS

    lmp_data : str, default="in.frame"
        the file with the frame to be minimized by LAMMPS

    lmp_min : str, default="dump.minimization.lammpstrj"
        the dump file where the local minimization is written

    lmp_flags : dict, default=None
        command-line options for LAMMPS, where the key is the flag and the
        value the option

    verbose : bool, default=True
        print on the screen each of the values obtained for the performed
        minimizations

    Returns
    -------
    pd.DataFrame
        A `pd.DataFrame` with three columsn: the initial, the next to last and
        final energies of each frame. It also generates a lammpstrj file with
        all the minimum energy frames in the same order as the initial biased
        xyz file.

    Raises
    ------
    RuntimeError
        A problem has occurred, it may be because the LAMMPS simulation has
        not finished correctly or the log file does not correspond to a
        minimization.

    References
    ----------
    .. [1] Fernandez, F., Paz, S.A., Otero, M., Barraco, D. and Leiva, E.P.,
       2021. Characterization of amorphous Li x Si structures from ReaxFF via
       accelerated exploration of local minima. `Physical Chemistry Chemical
       Physics`, 23(31), pp.16776-16784.
    """
    if verbose:
        k = 0
        print("# minimization number, initial energy, final energy")

    initial, final = [], []

    bias_traj = exma.read_xyz(biased_traj)
    min_traj = []

    for bias_frame in bias_traj:
        bias_frame.box = cell_info["box"]
        bias_frame.idx = np.arange(1, bias_frame.natoms + 1)
        bias_frame.types = [cell_info["type"][t] for t in bias_frame.types]
        bias_frame.q = np.zeros(bias_frame.natoms, dtype=np.float32)

        exma.write_in_lammps(bias_frame, lmp_data)

        # run lammps minimization with the corresponding flags
        run_cmd = [lmp_exec, "-in", lmp_in]
        if lmp_flags is not None:
            for key, value in lmp_flags.items():
                if key != "screen":
                    run_cmd.append(f"-{key}")
                    run_cmd.append(value)
        lmp_run = subprocess.run(run_cmd, capture_output=True, text=True)
        log = lmp_run.stdout.split("\n")

        # get the energies and save for the pd.DataFrame
        try:
            for line, next_line in zip(log, log[1:] + [log[0]]):
                if line.strip().startswith("Energy initial"):
                    e1, e2, e3 = next_line.strip().split()
            initial.append(e1)
            final.append(e3)
        except UnboundLocalError:
            raise RuntimeError(
                "A problem occurred when obtaining the energies"
            )

        # accumulate the minimum energy frames
        min_frame = exma.read_lammpstrj(lmp_min)[-1]
        min_traj.append(
            min_frame._sort_frame() if not min_frame._sorted() else min_frame
        )

        if verbose:
            print(k, e1, e3)
            k += 1

    exma.write_lammpstrj(min_traj, minimizations_output)

    initial = np.asarray(initial, dtype=np.float32)
    final = np.asarray(final, dtype=np.float32)

    return pd.DataFrame({"initial": initial, "final": final})
