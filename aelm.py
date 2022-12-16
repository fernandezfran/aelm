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
# CLASSES
# ============================================================================
class AELM:
    """Accelerated Exploration of Local Minima minimizations.

    This class manages the process of local minimizations after obtaining a
    biased trajectory [1]_.

    Parameters
    ----------
    program : str
        'LAMMPS' or 'GEMS' are the possible values to run the minimizations.

    biased_traj : list of `exma.core.AtomicSystem`
        the biased frames.

    cell_info : dict
        with the `box`, the lenght of the box in each direction, and,
        eventually, with another dictionary identified with the `type` key that
        has within it a correspondence between the elements present in xyz file
        with integer identification numbers, e.g. {"Si": 1, "Li": 2}, necesary
        for minimizations with LAMMPS.

    Raises
    ------
    ValueError
        If program is not 'LAMMPS' or 'GEMS'.

    References
    ----------
    .. [1] Fernandez, F., Paz, S.A., Otero, M., Barraco, D. and Leiva, E.P.,
       2021. Characterization of amorphous Li x Si structures from ReaxFF via
       accelerated exploration of local minima. `Physical Chemistry Chemical
       Physics`, 23(31), pp.16776-16784.
    """

    def __init__(self, program, biased_traj, cell_info):
        if program not in ("LAMMPS", "GEMS"):
            raise ValueError("program must be 'LAMMPS' or 'GEMS'")

        self.program = program

        self.biased_traj = biased_traj
        self.cell_info = cell_info

    def _with_lmp(self):
        """Minimization loop with LAMMPS."""
        k = 0
        for bias_frame in self.biased_traj:
            bias_frame.box = self.cell_info["box"]
            bias_frame.idx = np.arange(1, bias_frame.natoms + 1)
            bias_frame.types = [
                self.cell_info["type"][t] for t in bias_frame.types
            ]
            bias_frame.q = np.zeros(bias_frame.natoms, dtype=np.float32)

            exma.write_in_lammps(bias_frame, self.to_min_frame)

            # run the minimization
            lmp_run = subprocess.run(
                self.run_cmd, capture_output=True, text=True
            )

            # get the energies and save for the pandas.DataFrame
            try:
                log = lmp_run.stdout.split("\n")
                for line, next_line in zip(log, log[1:] + [log[0]]):
                    if line.strip().startswith("Energy initial"):
                        ei, ebf, ef = next_line.strip().split()

                self.initial_.append(ei)
                self.final_.append(ef)

            except UnboundLocalError:
                raise RuntimeError(
                    "A problem occurred when obtaining the energies"
                )

            # accumulate the minimum energy frames
            mef = exma.read_lammpstrj(self.minimization_frames)[-1]
            self.min_traj_.append(mef._sort() if not mef._sorted() else mef)

            if self.verbose:
                print(f"{k},{ei},{ef}")
                k += 1

    def _with_gms(self):
        """Minimization loop with GEMS."""
        k = 0
        for bias_frame in self.biased_traj:
            exma.write_xyz([bias_frame], self.to_min_frame)

            # run the minimization
            _ = subprocess.run(self.run_cmd)

            # get the energies and save for the pandas.DataFrame
            with open(self.log_run, "r") as f:
                log = f.read()
            log = log.split("\n")
            lbfgs_log = [
                line for line in log if line.strip().startswith("# LBFGS")
            ]
            ei = lbfgs_log[2].split()[4]
            ef = lbfgs_log[-1].split()[4]

            self.initial_.append(ei)
            self.final_.append(ef)

            # accumulate the minimum energy frames
            mef = exma.read_xyz(self.minimization_frames)[-1]
            self.min_traj_.append(mef)

            if self.verbose:
                print(f"{k},{ei},{ef}")
                k += 1

    def run(
        self, run_cmd, log_run, to_min_frame, minimization_frames, verbose=True
    ):
        """Run the minimizations.

        Parameters
        ----------
        run_cmd : str
            the command to run the minimization

        log_run : str
            the name of the file with the output of the program minimization.

        to_min_frame : str
            the file with the frame to be minimized.

        minimization_frames : str
            the file where the local minimization were written by the MD
            program.

        verbose : bool, default=True
            print on the screen each of the values obtained for the performed
            minimizations

        Raises
        ------
        RuntimeError
            A problem has occurred, it may be because the minimization has not
            finished correctly or the log file does not correspond to a
            minimization.
        """
        self.run_cmd = run_cmd.split()
        self.log_run = log_run
        self.to_min_frame = to_min_frame
        self.minimization_frames = minimization_frames
        self.verbose = verbose
        if verbose:
            print("minimization_number,initial_energy,final_energy")

        self.initial_, self.final_, self.min_traj_ = [], [], []

        if self.program == "LAMMPS":
            self._with_lmp()
        elif self.program == "GEMS":
            self._with_gms()

        initial = np.array(self.initial_, dtype=np.float32)
        final = np.array(self.final_, dtype=np.float32)

        self.energies_ = pd.DataFrame({"initial": initial, "final": final})

        return self

    @property
    def energies(self):
        """Energies of each frame as a `pandas.DataFrame`."""
        return self.energies_

    @property
    def minimized_frames(self):
        """List of `exma.core.AtomicSystem` for each frame."""
        return self.min_traj_
