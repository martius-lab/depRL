import shutil
import sys

import gym
import myosuite  # noqa
import torch

import deprl
from deprl import main, play

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
    )
    play.play(**kwargs)


def test_train():
    config_path = "./tests/test_files/test_settings.json"
    sys.argv.append(config_path)
    main.main()
    shutil.rmtree("./tests/test_DEPRL", ignore_errors=True)


if __name__ == "__main__":
    test_train()
    test_play()
