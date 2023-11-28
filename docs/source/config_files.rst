.. _config_files:

Configuration files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

deprl uses json files to configure training runs. The `experiments <https://github.com/martius-lab/depRL/tree/main/experiments>`_ folder in the main github directory contains training settings for several experiments.
For the MyoSuite and the MyoChallenge2023, you can find settings files `here <https://github.com/martius-lab/depRL/tree/main/experiments/myosuite_training_files>`_.

Links for sconegym will be added once the main repo has been updated.


.. list-table:: Configuration files for myosuite.
   :widths: 30 25 60
   :header-rows: 1

   * - environment id
     - file name
     - description
   * - myoLegWalk-v0
     - myoLegWalk.yaml
     - Train a straight walking myoLeg agent.
   * - myoChallengeChaseTagP1-v0
     - myoChaseTag.yaml
     - Used to create the ChaseTag baseline, but rewards are not provided.
   * - myoChallengeRelocateP1-v0
     - myoRelocate.yaml
     - Used to create the Relocate baseline, but rewards are not provided.

.. list-table:: Configuration files for sconegym.
   :widths: 30 25 60
   :header-rows: 1

   * - environment id
     - file name
     - description
   * - sconewalk_h0918-v1
     - sconewalk_h0918.yaml
     - Used to train energy-efficient walking with the H0918 model.
   * - sconewalk_h1622-v1
     - sconewalk_h1622.yaml
     - Used to train energy-efficient walking with the H1622 model.
   * - sconewalk_h2190-v1
     - sconewalk_h2190.yaml
     - Used to train energy-efficient walking with the H2190 model.
   * - sconerun_h0918-v1
     - sconerun_h0918.yaml
     - Used to train running with the H0918 model.
   * - sconerun_h1622-v1
     - sconerun_h1622.yaml
     - Used to train running with the H1622 model.
   * - sconerun_h2190-v1
     - sconerun_h2190.yaml
     - Used to train running with the H2190 model.

To train a myoLeg agent, you can use the following command:

.. code-block:: bash

  python -m deprl.main experiments/myosuite_training_files/myoLegWalk.yaml

Similarly for sconegym, you can use:

.. code-block:: bash

  python -m deprl.main experiments/hyfydy/sconerun_h2190.yaml

Detailed example
---------------------------------

In this section, we'll start with an example settings file, explain the main purpose of each line and then go into details.

.. code-block:: bash
  :linenos:

  tonic:
    after_training: ''
    header: "import deprl, gym, sconegym"
    agent: "deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(replay=deprl.replays.buffers.Buffer(return_steps=1,
      batch_size=256, steps_between_batches=1000, batch_iterations=30, steps_before_batches=1e6))"
    before_training: ''
    checkpoint: "last"
    environment: "deprl.environments.Gym('sconerun_h2190-v1', scaled_actions=False)"
    full_save: 1
    name: "sconerun_h2190_v1"
    resume: false
    seed: 0
    parallel: 20
    sequential: 10
    test_environment: null
    trainer: "deprl.custom_trainer.Trainer(steps=int(5e8), epoch_steps=int(2e5), save_steps=int(1e6))"

  working_dir: "IGNORED_FOR_HYFYDY"

  env_args:
    clip_actions: false
    grf_coeff: -0.07281
    joint_limit_coeff: -0.1307
    nmuscle_coeff: -1.57929
    smooth_coeff: -0.097
    self_contact_coeff: -10.0
    vel_coeff: 10.0
    step_size: 0.025
    run: true
    init_activations_mean: 0.01
    init_activations_std: 0 # if 0: deterministic

  mpo_args:
    hidden_size: 1024
    lr_actor: 3.53e-05
    lr_critic: 6.081e-05
    lr_dual: 0.00213

  DEP:
    bias_rate: 0.002
    buffer_size: 200
    intervention_length: 8
    intervention_proba: 0.00371
    kappa: 1000
    normalization: "independent"
    q_norm_selector: "l2"
    regularization: 32
    s4avg: 2
    sensor_delay: 1


* `header`: The header is executed in `main.py` before training starts and should contain all needed dependencies for training.
* `agent`: This is the agent used for training.
* `environment`: This is the environment used for training.
* `test_environment`: This is the environment used for evaluation. If `null`, the training environment is used.
* `trainer`: This is the main trainer function. It contains the maximum training time `steps`, the number of steps per epoch `epoch_steps` and the number of steps between saving checkpoints `save_steps`. `save_steps` should be larger than `epoch_steps`.
* `parallel`: This is the number of parallel environments used for training. We recommend setting this to the number of cores your machine has.
* `sequential`: This is the number of sequential environments used for training. The total number of environments is `parallel` x `sequential`. If `parallel=P` and `sequential=S`, this will create `P` parallel groups of `S` environments which are executed in sequence. Mind that this number strongly affects the required RAM for training.
* `working_dir`: This is the directory where the results are saved.

.. note::
  The saving mechanism for SCONE/Hyfydy is slightly different from the default, improving integration with the remaining SCONE features. When a sconegym experiment is detected, the experiment is automatically saved to the results folder defined in the SCONE interface. The `working_dir` argument is ignored in that case.

* `env_args`: This is a dictionary of arguments passed to the environment. The environments will save this as `env.keyword = value`. It's only useful for specific environments that use these settings. This is distinct from the passing of additional keyword arguments to `deprl.environments.Gym(...)` which will be given to the `gym.make` function at the first creation of the env.
* `mpo_args`: These settings get passed to the MPO algorithm and can be used to adapt the learning rates of actor and critic `lr_actor`, `lr_critic`, the learning rate of the dual optimizer `lr_dual` and the hidden size of the actor and critic networks.

.. note::
 The passing of arguments to the learner is only suppoerted for MPO. Take a look at the `TunedMPO` class to see how you can implement it for other algorithms.

* `DEP`: These are specific arguments passed to DEP, see the DEP-RL publication for details.
* `name`: The name of the experiment. A folder with this name will be created at `working_dir/name/` and the experiment will be saved inside.
* `full_save`: Whether all training components (replay buffer, optimizer state, policies, critics, ...) should be saved or only the actor policy. We take care to save the replay buffer in chunks to not increase RAM consumption too much. Nevertheless, full saves will temporarily use more RAM and hard disk storage. Only the latest replay buffer is stored.
* `resume`: When a new training run is started with an already existing `working_dir` and `name`, we either load from the previous experiment, when `resume: true`, or start a new run in the same folder, when `resume: false`. This assumes that the previous run was saved with `full_save: true`, otherwise we cannot continue the experiment.

Subcommand explanations
.........................

* For DEP, the function `deprl.custom_agents.dep_factory` takes a `TonicRL` agent and connects it to DEP. Several ways are implemented and can be chosen by passing the right integer here.

.. list-table:: DEP types.
   :widths: 30 25 60
   :header-rows: 1

   * - DEP type
     - value
     - description
   * - No DEP
     - 0
     - Train a normal agent without any DEP.
   * - InitExploreDEP
     - 1
     - DEP is only initially used to pre-fill the replay buffer of an off-policy algorithm.
   * - DetSwitchDEP
     - 2
     - DEP and the RL agent are deterministically switched. Also aplies InitExploreDEP.
   * - StochSwitchDEP
     - 3
     - DEP and the RL agent are stochastically switched. Also aplies InitExploreDEP. This was used in the paper.

* When creating the environment, we call `deprl.environments.Gym(name, scaled_actions=False, ...)`. This instantiates a previously registered gym environment and wraps it for use with the deprl framework. The `scaled_actions` flag tells the wrapper to normalize the action space. We disable this feature as DEP exploration works better without it. All additional keywords passed to the function will be given to `gym.make(name, keyword=value, ...)`. This mechanism can be used to change the environment, if supported.
