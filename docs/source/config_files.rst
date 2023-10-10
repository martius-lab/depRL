.. _config_files:

Configuration files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

deprl uses json files to configure training runs. The `experiments <https://github.com/martius-lab/depRL/tree/main/experiments>`_ folder in the main github directory contains training settings for several experiments.
For the MyoSuite and the MyoChallenge2023, you can find settings files `here <https://github.com/martius-lab/depRL/tree/main/experiments/myosuite_training_files>`_.


.. list-table:: Provided Settings files
   :widths: 30 25 60
   :header-rows: 1

   * - environment id
     - file name
     - description
   * - myoLegWalk-v0
     - myoLegWalk.json
     - Train a straight walking myoLeg agent.
   * - myoChallengeChaseTagP1-v0
     - myoChaseTag.json
     - Used to create the ChaseTag baseline, but rewards are not provided.
   * - myoChallengeRelocateP1-v0
     - myoRelocate.json
     - Used to create the Relocate baseline, but rewards are not provided.

To train a myoLeg agent, you can use the following command:

.. code-block:: bash

  python -m deprl.main experiments/myosuite_training_files/myoLegWalk.json

Detailed example
---------------------------------

In this section, we'll start with an example settings file, explain the main purpose of each line and then go into details.

.. code-block:: bash
  :linenos:

  {
   "tonic": {
      "header": "import deprl, gym, myosuite",
      "agent": "deprl.custom_agents.dep_factory(3, deprl.agents.MPO())(replay=deprl.replays.buffers.Buffer(return_steps=3, batch_size=256, steps_between_batches=1000, batch_iterations=30, steps_before_batches=2e5))",
      "environment": "deprl.environments.Gym('myoLegWalk-v0', scaled_actions=False, reset_type='random')",
      "test_environment": null,
      "trainer": "deprl.custom_trainer.Trainer(steps=int(1e8), epoch_steps=int(2e5), save_steps=int(1e6))",
      "before_training": "",
      "after_training": "",
      "parallel": 20,
      "sequential": 10,
      "seed": 0,
      "name": "myoLeg",
      "environment_name": "deprl_baseline",
      "checkpoint": "last",
      "full_save": 1,
    },

    "working_dir": "./baselines_DEPRL",
    "env_args":{},
    "DEP":{
      "test_episode_every": 3,
      "kappa": 1169.7,
      "tau": 40,
      "buffer_size": 200,
      "bias_rate": 0.002,
      "s4avg": 2,
      "time_dist": 5,
      "normalization":  "independent",
      "sensor_delay": 1,
      "regularization": 32,
      "with_learning": true,
      "q_norm_selector": "l2",
      "intervention_length": 5,
      "intervention_proba": 0.0004
    }
  }

* header: The header is executed in `main.py` before training starts and should contain all needed dependencies for training.
* agent: This is the agent used for training.
* environment: This is the environment used for training.
* test_environment: This is the environment used for evaluation. If `null`, the training environment is used.
* trainer: This is the main trainer function. It contains the maximum training time, the number of steps per epoch and the number of steps between saving checkpoints.
* parallel: This is the number of parallel environments used for training. We recommend setting this to the number of cores your machine has.
* sequential: This is the number of sequential environments used for training. The total number of environments is parallel x sequential. If parallel=P and sequential=S, this will create P parallel groups of S environments which are executed in sequence. Mind that this number strongly affects the required RAM for training.
* environment_name: This is a name used for saving the results. It doesn't affect the training environment.
* working_dir: This is the directory where the results are saved.
* env_args: This is a dictionary of arguments passed to the environment. The environments will save this as `env.keyword = value`. It's only useful for specific environments that use these settings.
* DEP: These are specific arguments passed to DEP, see the DEP-RL publication for details.

Subcommand explanations
.........................

This will be added later.
