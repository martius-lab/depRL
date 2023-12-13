import shutil
import sys

import gym
import myosuite  # noqa
import torch

import deprl
from deprl import main, play
from deprl.vendor.tonic import logger

SEED = 1


def test_play():
    name = "myoLegWalk-v0"
    env = gym.make(name, reset_type="random")
    env.seed(SEED)
    torch.manual_seed(SEED)
    _ = deprl.load_baseline(env)
    kwargs = dict(
        path="./baselines_DEPRL/myoLegWalk_20230514/myoLeg",
        checkpoint="last",
        seed=SEED,
        agent=None,
        environment=None,
        header=None,
        noisy=False,
        num_episodes=3,
        no_render=True,
        checkpoint_file=None,
    )
    play.play(**kwargs)


def test_train():
    config_path = "./tests/test_files/test_settings.json"
    sys.argv.append(config_path)
    main.main()


def test_load_resume():
    config_path = "./tests/test_files/test_settings_load_resume.json"
    sys.argv.append(config_path)
    main.main()
    shutil.rmtree("./tests/test_DEPRL", ignore_errors=True)


def test_load_no_resume():
    config_path = "./tests/test_files/test_settings_load_no_resume.json"
    sys.argv.append(config_path)
    main.main()
    shutil.rmtree("./tests/test_DEPRL", ignore_errors=True)


def test_exception():
    class TestException(Exception):
        pass

    class ExceptionTester(gym.Wrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.exception_steps = 0
            self.exception_limit = 5e2

        def step(self, action):
            self.exception_steps += 1
            if not self.exception_steps % self.exception_limit:
                raise TestException("Artificial Exception initiated")
            return self.env.step(action)

    env = ExceptionTester(gym.make("myoLegWalk-v0"))
    env.reset()
    for i in range(1000):
        try:
            env.step(env.action_space.sample())
        except TestException as e:
            logger.log(f"TestError is: {e}")


if __name__ == "__main__":
    # test_exception()
    test_train()
    test_load_resume()
    test_load_no_resume()
