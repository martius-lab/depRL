Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _usage:

deprl uses 4 basic commands, partially derived from TonicRL: main, play, log, plot. We refer to the TonicRL `README <https://github.com/fabiopardo/tonic>`_ for further exlanations on how to use them.

See below for brief explanations.


Main modules
.................................

train
`````````````````````````````````

The train command receives a settings file and trains an RL agent. The training file should contain all information about the desired algorithm, environment, number of training iterations, ...

This function also creates an output folder with the trained policy checkpoints, as well as a `log.csv` file that contains all information that was logged during the training process. The log-file can either be parsed with the plot- or the log-function of deprl.


Usage:

.. code-block:: bash

   python -m deprl.train experiments/settings.json

play
`````````````````````````````````


.. code-block:: bash

   python -m deprl.play --path folder/folder


plot
`````````````````````````````````


.. code-block:: bash

   python -m deprl.plot --path folder/folder



log
`````````````````````````````````

This convenience function takes the created `log.csv` file and parses it. The read information is uploaded to wandb, where users can create their own dashboards to analyze it. The log-function periodically checks if the log-file has been updated and transmits the new information. The use requires users to install `wandb` and to create a user account.

Usage:

.. code-block:: bash

   python -m deprl.log --path folder/log.csv


Useful functions
.................................

deprl also provide some useful code-level functions that you can use inside your python script.



load
`````````````````````````````````
This functon allows you to load a policy checkpoint inside any python script and just play with it. It is assumed that the passed env has `action_space` and `observation_space` attributes.


.. code-block:: python

  import gym
  import myosuite
  import deprl

  folder = 'path/to/your/policy/'

  env = gym.make('myoLegWalk-v0'):
  policy = deprl.load(folder, env)

  for ep in range(5):
      obs = env.reset()
      for i in range(1000):
          action = policy(obs)
          next_obs, reward, done, info = env.step(action)
          env.sim.renderer.render_to_window()
          obs = next_obs
