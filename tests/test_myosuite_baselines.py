import gym
import myosuite  # noqa
import numpy as np
import torch

import deprl

torch.set_default_device("cpu")

SEED = 1


def helper_env_loop(env):
    policy = deprl.load_baseline(env)
    returns = []
    qpos = []
    env.seed(SEED)
    for ep in range(10):
        ret = 0
        obs = env.reset()
        for i in range(2000):
            action = policy.noisy_test_step(obs)
            obs, reward, done, _ = env.step(action)
            # env.mj_render()
            ret += reward
            qpos.append(env.sim.data.qpos[1])
            if done:
                break
        returns.append(ret)
    env.close()
    return returns, qpos


def test_myolegwalk():
    name = "myoLegWalk-v0"
    env = gym.make(name, reset_type="random")
    env.seed(SEED)
    torch.manual_seed(SEED)
    returns, qpos = helper_env_loop(env)
    # assert np.round(np.mean(qpos), 2) == -1.47


def test_chasetag():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="random")
    env.seed(SEED)
    returns, qpos = helper_env_loop(env)
    print(np.mean(qpos))
    # assert np.mean(qpos) < -1.5


def test_relocate():
    name = "myoChallengeRelocateP1-v0"
    env = gym.make(name)
    env.seed(SEED)
    torch.manual_seed(SEED)
    returns, _ = helper_env_loop(env)
    # assert np.abs(np.floor(returns[0])) == 7538


def test_rng():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="init")
    env.seed(SEED)
    policies = []
    for i in range(3):
        env.reset()
        policies.append(env.opponent.opponent_policy)


def test_rng_noise():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="random")
    env.seed(SEED)
    for i in range(3):
        env.reset()
        for i in range(5):
            env.opponent.noise_process.sample()
    # assert not (np.mean(noise) + 1.3004040323) > 1e-6


def test_chasetag_obs_rng():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="init")
    env.seed(SEED)
    env.reset()
    for i in range(20):
        old_obs, *_ = env.step(env.np_random.normal(0, 1, size=(80,)))

    diff = 0
    for i in range(100):
        env.seed(SEED)
        obs = env.reset()
        for i in range(20):
            obs, *_ = env.step(env.np_random.normal(0, 1, size=(80,)))
        diff += np.abs(old_obs - obs)


def test_chasetag_actionrng():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="init")
    env.seed(SEED)
    env.reset()
    for i in range(20):
        old_obs, *_ = env.step(env.np_random.normal(0, 1, size=(80,)))

    policy = deprl.load_baseline(env)
    init_action = policy(old_obs)
    diff = 0
    for i in range(100):
        action = policy(old_obs)
        diff += np.abs(action - init_action)
    print(diff)


if __name__ == "__main__":
    test_relocate()
    test_myolegwalk()
    test_chasetag()
    # test_chasetag_actionrng()
    # test_chasetag_obs_rng()
    # test_rng_noise()
    # test_relocate()
