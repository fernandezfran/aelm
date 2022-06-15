.. aelm documentation master file, created by
   sphinx-quickstart on Wed Jun 15 14:41:53 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to aelm's documentation!
================================

.. image:: https://github.com/fernandezfran/aelm/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/fernandezfran/aelm/actions/workflows/ci.yml
   :alt: GitHub Actions CI

.. image:: https://readthedocs.org/projects/aelm/badge/?version=latest
   :target: https://aelm.readthedocs.io/
   :alt: ReadTheDocs

.. image:: https://img.shields.io/pypi/v/aelm
   :target: https://pypi.org/project/aelm/
   :alt: PyPI Version


**aelm** provides a function to find relevant energy minima and associated 
structures by performing local minimizations from a biased trajectory with LAMMPS.


Requirements
------------

You need Python 3.9+ and an executable of LAMMPS to run aelm.


Repository
----------

https://github.com/fernandezfran/aelm/


License
-------

aelm is under `MIT License <https://github.com/fernandezfran/aelm/blob/master/LICENSE>`__

---------------------------------------------------------------------------------

BibTeX citation of the 
`paper <https://pubs.rsc.org/en/content/articlelanding/2021/cp/d1cp02216d/unauth>`__
in which the method is explained: 

.. code-block:: text

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


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   tutorial.ipynb
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
