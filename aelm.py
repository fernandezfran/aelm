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
import os

import exma

import numpy as np

import pandas as pd


def aelm(
    biased_traj,
    minimizations_output,
    cell_info,
    lmp_exec="./lmp",
    lmp_in="in.minimization",
    lmp_min="dump.minimization.lammpstrj",
    lmp_flags=None,
):
    """Accelerated exploration of local minima minimization.

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

    lmp_min : str, default="dump.minimization.lammpstrj"
        the dump file where the local minimization is written

    lmp_flags : dict, default=None
        command-line options for LAMMPS, where the key is the flag and the value
        the option.

    Returns
    -------
    pd.DataFrame
        A `pd.DataFrame with three columsn: the initial, the next to last and
        final energies of each frame. It also generates a lammpstrj file with
        all the minimum energy frames in the same order as the initial biased
        xyz file.
    """
    initial, next_to_last, final = [], [], []
    with exma.io.reader.XYZ(biased_traj, ftype="xyz") as bias:
        try:
            while True:
                # read and xyz frame
                frame = bias.read_frame()
                frame.box = cell_info["box"]
                frame.idx = np.arange(1, frame.natoms + 1)
                frame.types = [cell_info["type"][t] for t in frame.types]
                frame.q = np.zeros(frame.natoms, dtype=np.float32)

                # write the frame to a configurations input file for lammps
                exma.io.writer.in_lammps("in.frame", frame)

                # run lammps minimization with the corresponding flags
                run_cmd = f"{lmp_exec} -in {lmp_in}"
                if lmp_flags is not None:
                    for key, value in lmp_flags.items():
                        if key != "screen" and value != "none":
                            run_cmd += f" -{key} {value}"
                os.system(run_cmd)

                # get the energies and save for the pd.DataFrame
                os.system(
                    f"awk '/Energy\ initial/,/Force\ two-norm/' log.lammps | sed '1d; $d' > emin.dat"
                )
                e1, e2, e3 = np.loadtxt(
                    "emin.dat", unpack=True, dtype=np.float32
                )
                initial.append(e1)
                next_to_last.append(e2)
                final.append(e3)

                # accumulate the minimum energy frames
                os.system(
                    f"tail -{frame.natoms + 9} {lmp_min} >> dump.all.lammpstrj"
                )

                # remove tmp files
                os.system("rm in.frame {lmp_min} log.* emin.dat")

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