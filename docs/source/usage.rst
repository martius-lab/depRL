Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _usage:

deprl uses 4 basic commands, partially derived from TonicRL: main, play, log, plot. We refer to the TonicRL `README <https://github.com/fabiopardo/tonic>`_ for further explanations on how to use them.

See below for brief explanations.


Main modules
.................................

These are the primary modules that you need to use deprl as an RL training library.

main - train policies
`````````````````````````````````

The train command receives a settings file and trains an RL agent. The training file should contain all information about the desired algorithm, environment, number of training iterations, ...

Take a look at the config_files header in the docs for more information on how to create a settings file and to see which ones we provide.

This function also creates an output folder with the trained policy checkpoints, as well as a `log.csv` file that contains all information that was logged during the training process and a `config.yaml` that contains training settings.
Here is an example folder structure for a baseline we trained, we will refer to this structure in later sections.

 .. note::
  The saving mechanism for SCONE/Hyfydy is slightly different from the default, improving integration with the remaining SCONE features. When a sconegym experiment is detected, the experiment is automatically saved to the results folder defined in the SCONE interface.

.. code-block:: bash

 myoLegWalk_20230514/
     │
     ├── myoLeg/
     │   │
     │   └───230514.142312/
     │       │
     │       ├── config.yaml
     │       ├── log.csv
     │       ├── checkpoints/
     │       │   ├── step_1000000.pt
     │       │   ├── step_2000000.pt
     │       │   └── ...
     │       └── ...
     └── ...

Usage:

.. code-block:: bash

   python -m deprl.main experiments/settings.yaml

.. _play:

play - render policies
`````````````````````````````````
This function allows you to render and execute trained policies. You can modify it to collect metrics, or just use it to visualize your policy.

Note that the last folder in the experiment subdirectory has to be given to the play function. In our example, the  command to render this policy would be:

.. code-block:: bash

   python -m deprl.play --path myoLegWalk_20230514/myoLeg/230514.142312/

On top of the commands included by TonicRL, deprl also provides some additional arguments:

--no_render        Prevent the play function from rendering.
--noisy            Use the Gaussian MPO policy instead of the deterministic one.
--num_episodes N   Play N episodes.


plot - plot training data
`````````````````````````````````
This function can plot recorded training data from the `log.csv` file in the experiment directory. See TonicRL `README <https://github.com/fabiopardo/tonic>`_ for more details.

Usage:

.. code-block:: bash

 python -m deprl.plot --path myoLegWalk_20230514/myoLeg/230514.142312/


log - log training data to wandb
`````````````````````````````````

This convenience function takes the created `log.csv` file and parses it. The read information is uploaded to wandb, where users can create their own dashboards to analyze it. The log-function periodically checks if the log-file has been updated and transmits the new information. The use requires users to install `wandb` and to create a user account.

Usage:

.. code-block:: bash

   python -m deprl.log --path myoLegWalk_20230514/myoLeg/230514.142312/log.csv

It allows two additional CLI arguments:

--project    The name of the wandb project to which the data should be uploaded.
--user       The name of the wandb user that it should be uploaded to.
              Once wandb is configured, you can leave this blank.
