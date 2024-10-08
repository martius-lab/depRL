# This example requires the installation of myosuite
# pip install myosuite

import time

import myosuite  # noqa
from myosuite.utils import gym

import deprl
from deprl import env_wrappers

env = gym.make("myoLegWalk-v0", reset_type="random")
env = env_wrappers.GymWrapper(env)
policy = deprl.load_baseline(env)

env.seed(0)
for ep in range(10):
    ep_steps = 0
    ep_tot_reward = 0
    state = env.reset()

    while True:
        # samples random action
        action = policy(state)
        # applies action and advances environment by one step
        state, reward, done, info = env.step(action)

        ep_steps += 1
        ep_tot_reward += reward
        env.mj_render()
        time.sleep(0.01)

        # check if done
        if done or (ep_steps >= 1000):
            print(
                f"Episode {ep} ending; steps={ep_steps}; reward={ep_tot_reward:0.3f};"
            )
            env.reset()
            break
