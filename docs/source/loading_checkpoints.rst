.. _loading:

Loading checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several ways to load previously trained policies.

Loading a policy in external code
.................................

If you are using your own code and want to load a deprl-policy into it, you can use the following example. It is assumed that the passed environment has `action_space` and `observation_space` attributes that are used to check if the trained policy matches the gym environment in input and output.

Note that we assume that the last subdirectory is given as a folder.

.. code-block:: python

  import gym
  import myosuite
  import deprl

  path = 'myoLegWalk_20230514/myoLeg/'

  env = gym.make('myoLegWalk-v0'):
  # load here
  policy = deprl.load(path, env)

  for ep in range(5):
      obs = env.reset()
      for i in range(1000):
          action = policy(obs)
          next_obs, reward, done, info = env.step(action)
          env.sim.renderer.render_to_window()
          obs = next_obs

This is different from the pre-trained baselines, because you are able to pass a custom path into the `load` function.

Continuing training with deprl
.................................

When you start a training run, a folder inside `working_dir/name` with the current date and time as foldername will be created. The `working_dir/name` path is extracted from the given settings file. After terminating a run, you can then simply call
.. code-block:: bash

        python -m deprl.main experiments/settings.yaml

.. _continue_training:

If in the config file you have the setting `"tonic.resume": True` then the most recent run in the specified `working_dir/name` will be loaded, where recency is determined by the time and date in the foldername. If `resume` is `False`, then a new folder with the current date and time will be created. Make sure you enabled the `full_save` argument in the config file, see :ref:`config_files`.
This will continue training from the chosen checkpoint. The `full_save` argument ensures that not only the policy, but also the replay buffer as well as the optimizer and normalizer states are saved, which are required to continue training. The settings for the continued training run are in this case loaded from the file provided to your main function, not from the settings in your output folder.

.. note::
   While the replay buffer is saved and loaded in chunks to reduce the memory load, there is still a spike in RAM usage during saving.

As an example, we can start training with:

.. code-block:: bash

   python -m deprl.main experiments/settings.yaml


Now you might cancel training at some point. If you would like to continue, simply call it again:

.. code-block:: bash

   python -m deprl.main experiments/settings.yaml

and training will continue.

.. note::

   * The output folder should not change between calls.

   * You must have enabled the option `full_save` during initial training, otherwise only the actor policy will be loaded.

   * `resume` should be enabled during loading.

   * If the training time becomes larger than the maximum training steps that you specified, you can edit the settings file in the folder and increase the `Trainer(steps=T)` value.

   * Don't cancel training while saving is in progress as this might corrupt large files.


Loading and visualizing a single checkpoint
...........................................

We refer to the documentation on the :ref:`play` function. The settings for the play and rendering are loaded from the output folder that you specify.


Loading a pre-trained baseline
.................................


We refer to the documentation on the :ref:`myobaselines` or the :ref:`hyfydybaselines` functions.
