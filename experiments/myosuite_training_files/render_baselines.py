import gym
import myosuite  # noqa
import torch

import deprl

SEED = 1


def helper_env_loop(env):
    policy = deprl.load_baseline(env)
    policy.noisy = False
    returns = []
    qpos = []
    for ep in range(5):
        ret = 0
        env.seed(SEED)
        obs = env.reset()
        for i in range(500):
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            env.sim.renderer.render_to_window()
            ret += reward
            qpos.append(env.sim.data.qpos[1])
            if done:
                break
        returns.append(ret)
    env.close()
    return returns, qpos


def render_myolegwalk():
    name = "myoLegWalk-v0"
    env = gym.make(name, reset_type="random")
    env.seed(SEED)
    torch.manual_seed(SEED)
    returns, _ = helper_env_loop(env)


def render_chasetag():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="init")
    env.seed(SEED)
    returns, qpos = helper_env_loop(env)


def render_relocate():
    name = "myoChallengeRelocateP1-v0"
    env = gym.make(name)
    env.seed(SEED)
    torch.manual_seed(SEED)
    returns, _ = helper_env_loop(env)


if __name__ == "__main__":
    render_chasetag()
    render_myolegwalk()
    render_relocate()
