import os

from aelm import aelm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ["OMP_NUM_THREADS"] = "2"

df = aelm(
    "biased_traj.xyz",
    "dump.lammpstrj",
    {"box": np.full(3, 10.609089), "type": {"Si": 1, "Li": 2}},
    lmp_flags={"sf": "omp"},
)

df.to_csv("example.csv", index=False)

df.plot.hist(column=["initial", "final"], bins=75)

plt.xlabel("Energy")
plt.savefig("example.png", dpi=200)
plt.show()
