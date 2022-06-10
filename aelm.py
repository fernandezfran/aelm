#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of aelm (https://github.com/fernandezfran/aelm/).
# Copyright (c) 2022, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/aelm/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Accelerated exploration of local minima."""

# ============================================================================
# IMPORTS
# ============================================================================
# import subprocess

import exma

import numpy as np

import pandas as pd


def aelm(biased_traj, minimizations_output, cell_info):
    """Accelerated exploration of local minima minimization.

    Parameters
    ----------
    biased_traj : str
        filename of the biased trajectory

    minimizations_output : str
        filename where the last frame of each minimization will be written

    cell_info : dict
        with the `box`, the lenght of the box in each direction, another
        dictionary identified with the `type` key that has within it a
        correspondence between the elements present in xyz file with integer
        identification numbers, e.g. {"Si": 1, "Li": 2}

    Returns
    -------
    pd.DataFrame
        A `pd.DataFrame with three columsn: the initial, the next to last and
        final energies of each frame.
    """
    initial, next_to_last, final = [], [], []
    with exma.io.reader.XYZ(biased_traj, ftype="xyz") as bias:
        try:
            while True:
                frame = bias.read_frame()
                frame.box = cell_info["box"]
                frame.idx = np.arange(1, frame.natoms + 1)
                frame.types = [cell_info["type"][t] for t in frame.types]

                exma.io.writer.in_lammps("in.frame", frame)

                # run lammps minimization and get the energy data and last
                # frame of the structure:
                #
                # os.system("./lmp_serial -in in.minimization")
                # os.system(
                #     "awk '/Energy\ initial/,/Force\ two-norm/' log.lammps |"
                #     "sed '1d; $d' >> emin.dat")
                # os.system(
                #     "tail -%s dump.minimization.lammpstrj >> "
                #     "dump.all.lammpstrj" % (N+9))
                # os.system("rm in.frame dump.minimization.lammpstrj log.*")
        except EOFError:
            ...

        finally:
            return pd.DataFrame(
                {
                    "initial": np.asarray(initial, dtype=np.float32),
                    "next_to_last": np.asarray(next_to_last, dtype=np.float32),
                    "final": np.asarray(final, dtype=np.float32),
                }
            )
