# aelm

[![Github Actions CI](https://github.com/fernandezfran/aelm/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandezfran/aelm/actions/workflows/ci.yml)

**aelm** provides a function to find relevant energy minima and associated 
structures by performing local minimizations from a biased trajectory with LAMMPS.


## Usage

```python
from aelm import aelm

df = aelm(
    "biased_traj.xyz",
    "dump.minimized.lammpstrj",
    {"box": np.array([xbox, ybox, zbox])}, "type": {"Si": 1, "Li": 2},
)
```

In this example you need an executable of [LAMMPS](https://www.lammps.org/), 
_lmp_, its input file with the minimization script, _in.minimization_, and a 
biased trajectory, _biased_traj.xyz_ (that can be simulated, for example, with 
[GEMS](https://github.com/alexispaz/GEMS)) in the same directory as the script.


## Requirements

You need Python 3.9+ and an executable of LAMMPS to run aelm.


## Installation

### Stable release

To install the most recent stable release of aelm with [pip](https://pip.pypa.io/en/stable/), 
run the following command in your termninal:

```bash
pip install aelm
```

### From sources

To installing it from sources you can clone this [GitHub repo](https://github.com/fernandezfran/aelm) 

```bash
git clone https://github.com/fernandezfran/aelm.git
```

and inside your local directory install it in the following way 

```bash
pip install -e .
```


## License

[MIT License](https://github.com/fernandezfran/aelm/blob/master/LICENSE)


----------------------------------------------------------------------------------

BibTeX citation of the 
[paper](https://pubs.rsc.org/en/content/articlelanding/2021/cp/d1cp02216d/unauth)
in which the method is explained:
```
@article{fernandez2021characterization,
  title={Characterization of amorphous Li x Si structures from ReaxFF via accelerated exploration of local minima},
  author={Fernandez, Francisco and Paz, Sergio Alexis and Otero, Manuel and Barraco, Daniel and Leiva, Ezequiel PM},
  journal={Physical Chemistry Chemical Physics},
  volume={23},
  number={31},
  pages={16776--16784},
  year={2021},
  publisher={Royal Society of Chemistry}
}
```

<fernandezfrancisco2195@gmail.com>
