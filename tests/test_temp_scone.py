import os

import gym
import sconegym
import numpy as np
import torch

import deprl

SEED = 1


def helper_env_loop(env):
    returns = []
    qpos = []
    for ep in range(1):
        ret = 0
        env.seed(SEED)
        obs = env.reset()
        for i in range(500):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            ret += reward
            if done:
                break
        returns.append(ret)
    env.close()
    return returns


def test_envs():
    for name in ['sconegaith0918-v0', 
                 'sconegaith1622-v0',
                 'sconegaith2190-v0',
                 'sconegaith0918S2-v0',
                 'sconegaith1622S2-v0',
                 'sconegaith2190S2-v0',
                 'sconegaith0918_delay-v0']:
        env = gym.make(name)
        env.seed(SEED)
        torch.manual_seed(SEED)
        returns = helper_env_loop(env)
        # assert np.floor(returns[0]) == 3511



if __name__ == "__main__":
    test_envs()
    # test_chasetag_actionrng()
    # test_chasetag_obs_rng()
    # test_relocate()
    # test_rng_noise()
    # test_myolegwalk()
    # test_relocate()
