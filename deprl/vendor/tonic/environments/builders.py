"""Environment builders for popular domains."""

import os
from types import SimpleNamespace

import numpy as np
from gymnasium import wrappers

try:
    from myosuite.utils import gym

    print("Importing Myosuite gym, no scone possible")
except ModuleNotFoundError:
    import gym

    print("Importing gym for scone")


from deprl.vendor.tonic import environments
from deprl.vendor.tonic.utils import logger


def gym_environment(*args, **kwargs):
    """Returns a wrapped Gym environment."""
    if "header" in kwargs:
        kwargs.pop("header")

    def _builder(*args, **kwargs):
        return gym.make(*args, **kwargs)

    return build_environment(_builder, *args, **kwargs, header=None)


def build_environment(
    builder,
    name,
    terminal_timeouts=False,
    time_feature=False,
    max_episode_steps="default",
    scaled_actions=True,
    header=None,
    *args,
    **kwargs,
):
    """Builds and wrap an environment.
    Time limits can be properly handled with terminal_timeouts=False or
    time_feature=True, see https://arxiv.org/pdf/1712.00378.pdf for more
    details.
    """
    if header is not None:
        exec(header)

    # Build the environment.
    environment = builder(name, *args, **kwargs)

    # Get the default time limit.
    if max_episode_steps == "default":
        if hasattr(environment, "_max_episode_steps"):
            max_episode_steps = environment._max_episode_steps
        elif hasattr(environment, "horizon"):
            max_episode_steps = environment.horizon
        elif hasattr(environment, "max_episode_steps"):
            max_episode_steps = environment.max_episode_steps

        else:
            logger.log("No max episode steps found, setting them to 1000")
            max_episode_steps = 1000

    # Remove the TimeLimit wrapper if needed.
    if not terminal_timeouts:
        if type(environment) == wrappers.TimeLimit:
            environment = environment.env

    # Add time as a feature if needed.
    if time_feature:
        environment = environments.wrappers.TimeFeature(
            environment, max_episode_steps
        )

    # Scale actions from [-1, 1]^n to the true action space if needed.
    if scaled_actions:
        environment = environments.wrappers.ActionRescaler(environment)

    environment.name = name
    environment.max_episode_steps = max_episode_steps

    return environment


def _flatten_observation(observation):
    """Turns OrderedDict observations into vectors."""
    observation = [
        np.array([o]) if np.isscalar(o) else o.ravel()
        for o in observation.values()
    ]
    return np.concatenate(observation, axis=0)


# Aliases.
Gym = gym_environment
