import gym
import myosuite  # noqa
import numpy as np

import deprl


def helper_env_loop(name):
    env = gym.make(name, reset_type="random")
    env.seed(1)
    policy = deprl.load_baseline(env)
    returns = []
    for ep in range(1):
        ret = 0
        obs = env.reset()
        for i in range(200):
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            # env.sim.renderer.render_to_window()
            ret += reward
            if done:
                break
        returns.append(ret)
    return returns


def test_myolegwalk():
    name = "myoLegWalk-v0"
    returns = helper_env_loop(name)
    assert np.floor(returns[0]) == 3511


def test_chasetag():
    name = "myoChallengeChaseTagP1-v0"
    returns = helper_env_loop(name)
    print(returns)


def test_relocate():
    name = "myoChallengeRelocateP1-v0"
    returns = helper_env_loop(name)
    print(returns)


if __name__ == "__main__":
    test_myolegwalk()
