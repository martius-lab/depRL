.. _hyfydybaselines:


Hyfydy baselines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We include several pretrained baselines for Hyfydy. They are similar to the ones trained for our `preprint <https://arxiv.org/abs/2309.02976>`_.
The baselines includes straight walking for `sconewalk_h0918-v1`, running for `sconerun_h0918-v1` and similar for the other models.
There is also an example for OpenSim `sconewalk_h0918_osim-v1`

To try the baselines, you need to first install `sconegym` and scone. See `here <https://github.com/tgeijten/sconegym>`_ for installation help.

You can play with the pre-trained baselines by using the code in this section. To train agents yourself, go to the :ref:`config_files` section.


.. list-table:: Pre-trained baselines.
   :widths: 30 60
   :header-rows: 1

   * - environment id
     - description
   * - sconewalk_h0918_osim-v1
     - Energy-efficient walking with the H0918 model in OpenSim (slow performance).
   * - sconewalk_h0918-v1
     - Energy-efficient walking with the H0918 model.
   * - sconewalk_h1622-v1
     - Energy-efficient walking with the H1622 model.
   * - sconewalk_h2190-v1
     - Energy-efficient walking with the H2190 model.
   * - sconerun_h0918-v1
     - Running with the H0918 model.
   * - sconerun_h1622-v1
     - Running with the H1622 model.
   * - sconerun_h2190-v1
     - Running with the H2190 model.

Usage example
-------------

.. code-block:: python

 import gym
 import sconegym
 import deprl

 env = gym.make('sconewalk_h0918-v1')
 policy = deprl.load_baseline(env)

 for ep in range(5):
     obs = env.reset()
     for i in range(1000):
         action = policy(obs)
         next_obs, reward, done, info = env.step(action)
         obs = next_obs
         if done:
             break


For the other baselines, just use: `env = gym.make('sconewalk_h2190-v1')` or `env = gym.make('sconerun_h2190-v1')`


You can also use noisy policy steps with:

.. code-block:: python

 import gym
 import sconegym
 import deprl

 env = gym.make('sconewalk_h0918-v1')
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
