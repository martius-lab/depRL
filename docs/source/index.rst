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
   hyfydy_baselines
   myo_baselines
   config_files
   loading_checkpoints


.. note::

   This project is under active development.


Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I created this package to make my research contributions easily available to others. It allows to train general reinforcement learning agents, while using components and algorithms developed from my research. The main focus is on RL control algorithms for muscle-driven systems, but some components can also be used in a more general scenario. The deprl codebase is based on `TonicRL <https://github.com/fabiopardo/tonic>`_ and uses its core RL algorithms, while I extended it to include my own ideas.


Environment and algorithm support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the moment, there is basic support for general gym and dmcontrol environments. The myosuite and sconegym (hyfydy) environments, used in my publications, are tested for. The packages for ostrichrl and warmup are also supported.
As deprl is built on TonicRL, it supports various RL algorithms. However, as I primarily use MPO with PyTorch as a learner for my experiments, that is the most well tested. I aim to verify my code on Windows, MacOS and Linux, but can't guarantee that all configurations work.

Hyperparameters and baselines from my publications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I provide a range of different training parameters and pre-trained baselines from my works. Check out  :ref:`config_files` to see configuration files for MyoSuite and sconegym that train biomechanical models for walking and running. These training runs generally finish in under 24h on a single workstation with a GPU and a large number of cores (~16). If you do not have enough RAM, try to run them with a reduced `parallel` setting. Systems with fewer muscles train much faster, usually less than 5 hours on a single workstation.

I also provide pre-trained policies for myosuite and for different scongym environments. Check  :ref:`myobaselines` and  :ref:`hyfydybaselines` respectively.


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

.. code-block:: bibtex

  @misc{schumacher2023natural,
    title={Natural and Robust Walking using Reinforcement Learning without Demonstrations in High-Dimensional Musculoskeletal Models},
    author={Pierre Schumacher and Thomas Geijtenbeek and Vittorio Caggiano and Vikash Kumar and Syn Schmitt and Georg Martius and Daniel F. B. Haeufle},
    year={2023},
    eprint={2309.02976},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
  }
