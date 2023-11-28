.. _myobaselines:

MyoSuite baselines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We include several pretrained baselines for MyoSuite and MyoChallenge2023 environments. This includes straight walking for `myoLegWalk-v0`, standing for `myoChallengeChaseTagP1-v0` and cube lifting for `myoChallengeRelocateP1-v0`.

To try the baselines, you need to first install `myosuite==2.0.1`.
You can play with the pre-trained baselines by using the code in this section. To train agents yourself, go to the :ref:`config_files` section.

.. list-table:: Pre-trained baselines for myosuite.
   :widths: 30 60
   :header-rows: 1

   * - environment id
     - description
   * - myoLegWalk-v0
     - Train a straight walking myoLeg agent.
   * - myoChallengeChaseTagP1-v0
     - Used to create the ChaseTag baseline, but rewards are not provided.
   * - myoChallengeRelocateP1-v0
     - Used to create the Relocate baseline, but rewards are not provided.

Usage example
-------------

.. code-block:: python

 import gym
 import myosuite
 import deprl

 # we can also change the reset_type of the environment here
 env = gym.make('myoLegWalk-v0', reset_type='random')
 policy = deprl.load_baseline(env)

 for ep in range(5):
     obs = env.reset()
     for i in range(1000):
         action = policy(obs)
         next_obs, reward, done, info = env.step(action)
         env.sim.renderer.render_to_window()
         obs = next_obs
         if done:
             break


For the other baselines, just use: `env = gym.make('myoChallengeRelocateP1-v0')` or `env = gym.make('myoChallengeChaseTagP1-v0')`


You can also use noisy policy steps with:

.. code-block:: python

 import gym
 import myosuite
 import deprl

 # we can also change the reset_type of the environment here
 env = gym.make('myoLegWalk-v0', reset_type='random')
 policy = deprl.load_baseline(env)

 for ep in range(5):
     obs = env.reset()
     for i in range(1000):
         # we use a noisy policy here
         action = policy.noisy_test_step(obs)
         next_obs, reward, done, info = env.step(action)
         env.sim.renderer.render_to_window()
         obs = next_obs
         if done:
             break


This can affect your performance positively or negatively, depending on the task!
