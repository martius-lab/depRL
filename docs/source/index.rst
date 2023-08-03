.. deprl documentation master file, created by
   sphinx-quickstart on Mon Jul 31 13:35:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to deprl's documentation!
=================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   usage
   myo_baselines
   config_files


.. note::

   This project is under active development.


Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I created this package to make my research contributions easily available to others. It allows to train general reinforcement learning agents, while using components and algorithms developed from my research. The main focus is on RL control algorithms for muscle-driven systems, but some components can also be used in a more general scenario. The deprl codebase is based on `TonicRL <https://github.com/fabiopardo/tonic>`_ and uses its core RL algorithms, while I extended it to include my own ideas.


Environment and algorithm support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the moment, there is basic support for general gym and dmcontrol environments, while myosuite as well as the environments from my publications are tested for. This includes ostrichrl and warmup.
As deprl is built on TonicRL, it supports various RL algorithms. However, as I primarily use MPO with PyTorch as a learner for my experiments, that is the most well tested.


How to cite
-----------

.. code-block:: bibtex

  @inproceedings{
    schumacher2023deprl,
    title={{DEP}-{RL}: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems},
    author={Pierre Schumacher and Daniel Haeufle and Dieter B{\"u}chler and Syn Schmitt and Georg Martius},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=C-xa_D3oTj6}
    }
