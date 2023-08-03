
MyoSuite Baselines 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _myobaselines:



We include several pretrained baselines for MyoSuite and MyoChallenge2023 environments. This includes straight walking for `myoLegWalk-v0`, standing for `myoChallengeChaseTagP1-v0` and cube lifting for `myoChallengeRelocateP1-v0`.

To try the baselines, you need to first install `myosuite==1.7.0`. Afterwards you can try:


.. code-block:: python

  import gym
  import myosuite
  import deprl

  env = gym.make('myoLegWalk-v0'):
  policy = deprl.load_baseline(env)

  for ep in range(5):
      obs = env.reset()
      for i in range(1000):
          action = policy(obs)
          next_obs, reward, done, info = env.step(action)
          env.sim.renderer.render_to_window()
          obs = next_obs
     



