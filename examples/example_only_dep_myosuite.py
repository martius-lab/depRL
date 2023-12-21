import time

import gym
import myosuite  # noqa

from deprl import env_wrappers
from deprl.dep_controller import DEP

env = gym.make("myoLegWalk-v0")
env = env_wrappers.GymWrapper(env)

dep = DEP()
dep.initialize(env.observation_space, env.action_space)

env.reset()
for i in range(1000):
    action = dep.step(env.muscle_lengths())[0, :]
    next_state, reward, done, _ = env.step(action)
    time.sleep(0.005)
    env.mj_render()
