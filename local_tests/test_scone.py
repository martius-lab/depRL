import gym
import sconegym  # noqa

SEED = 1


def helper_env_loop(env):
    returns = []
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
    for name in [
        "sconewalk_h0918-v0",
        "sconewalk_h1622-v0",
        "sconewalk_h2190-v0",
        "sconerun_h0918-v0",
        "sconerun_h1622-v0",
        "sconerun_h2190-v0",
        "sconewalk_h0918-v1",
        "sconewalk_h1622-v1",
        "sconewalk_h2190-v1",
        "sconerun_h0918-v1",
        "sconerun_h1622-v1",
        "sconerun_h2190-v1",
    ]:
        env = gym.make(name)
        env.seed(SEED)
        print(f"Testing {name=}")
        helper_env_loop(env)
        # assert np.floor(returns[0]) == 3511


if __name__ == "__main__":
    test_envs()
    # test_chasetag_actionrng()
    # test_chasetag_obs_rng()
    # test_relocate()
    # test_rng_noise()
    # test_myolegwalk()
    # test_relocate()
