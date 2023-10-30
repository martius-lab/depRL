.. _myobaselines:


Hyfydy baselines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We include several pretrained baselines for Hyfydy. They are similar to the ones trained for our `preprint <https://arxiv.org/abs/2309.02976>`_.
The baselines includes straight walking for `sconewalk-h0918-v0`, running for ...

To try the baselines, you need to first install `sconegym` and scone. TODO add information here or link.

You can play with the pre-trained baselines by using the code in this section. To train agents yourself, go to the Configuration Files section.


.. code-block:: python

 import gym
 import sconegym
 import deprl

 env = gym.make('sconewalk_h0918-v0')
 policy = deprl.load_baseline(env)

 for ep in range(5):
     obs = env.reset()
     for i in range(1000):
         action = policy(obs)
         next_obs, reward, done, info = env.step(action)
         obs = next_obs
         if done:
             break


For the other baselines, just use: `env = gym.make('sconewalk_h2190-v0')` or ...


You can also use noisy policy steps with:

.. code-block:: python

 import gym
 import myosuite
 import deprl

 env = gym.make('sconewalk_h0918')
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


This can affect your performance positively and negatively, depending on the task!
