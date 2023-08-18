import argparse
import json
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import gdown
import yaml

from deprl.vendor.tonic import logger


def prepare_files(orig_params):
    params = get_params(orig_params)
    os.makedirs(params.working_dir, exist_ok=True)
    return params


def get_params(orig_params):
    params = orig_params.copy()
    for key, val in params.items():
        if type(params[key]) == dict:
            params[key] = SimpleNamespace(**val)
    params = SimpleNamespace(**params)
    return params


def prepare_params():
    f = open(sys.argv[-1], "r")
    orig_params = json.load(f)
    params = prepare_files(orig_params)
    return orig_params, params


def load(path, environment, checkpoint="last", noisy=False):
    config, checkpoint_path = load_config_and_paths(path, checkpoint)
    header = config.header
    agent = config.agent
    # Run the header
    exec(header)
    # Build the agent.
    agent = eval(agent)
    # Adapt mpo specific settings
    if "mpo_args" in config:
        agent.set_params(**config.mpo_args)
    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
    )
    # Load the weights of the agent form a checkpoint.
    agent.load(checkpoint_path)
    agent.noisy = noisy
    return agent


def load_baseline(environment):
    identifier = (
        environment.env_name
        if hasattr(environment, "env_name")
        else str(environment)
    )
    if "myoLegWalk" in identifier:
        logger.log("Load LegWalk Baseline")
        return load_baseline_myolegwalk(environment)
    if "myoChallengeChaseTagP1" in identifier:
        logger.log("Load ChaseTagP1 Baseline")
        return load_baseline_myochasetagp1(environment, noisy=True)
    if "myoChallengeRelocateP1" in identifier:
        logger.log("Load RelocateP1 Baseline")
        return load_baseline_myorelocatep1(environment)


def load_baseline_myolegwalk(environment, noisy=False):
    modelurl = (
        "https://drive.google.com/uc?id=1UkiUozk-PM8JbQCZNoy2Jr2t1StLcAzx"
    )
    configurl = (
        "https://drive.google.com/uc?id=1knsol05ZL14aqyuaT-TlOvahr31gKQwE"
    )
    foldername = "./baselines_DEPRL/myoLegWalk_20230514/myoLeg"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_150000000.pt")
    configpath = os.path.join(foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment, noisy=noisy)


def load_baseline_myochasetagp1(environment, noisy=False):
    modelurl = (
        "https://drive.google.com/uc?id=12mEWnwGe7aWzfaHIJT8_qZPINGlPSLfQ"
    )
    configurl = (
        "https://drive.google.com/uc?id=11TRLmNtLMeBQ5H_JZ_tORxl9Idxhq-ec"
    )
    foldername = "./baselines_DEPRL/chasetagp1/"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_100000000.pt")
    configpath = os.path.join(foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment, noisy=noisy)


def load_baseline_myorelocatep1(environment, noisy=False):
    modelurl = (
        "https://drive.google.com/uc?id=1aBBamewALMxBglkR7nw8gLplKLam3AAO"
    )
    configurl = (
        "https://drive.google.com/uc?id=1UphxBaBLhPplZzhmhZNtkoAcW3M19bmo"
    )
    foldername = "./baselines_DEPRL/relocatep1/"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_11000000.pt")
    configpath = os.path.join(foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment, noisy=noisy)


def load_config_and_paths(path, checkpoint="last"):
    logger.log(f"Loading experiment from {path}")
    checkpoint_path = os.path.join(path, "checkpoints")
    if not os.path.isdir(checkpoint_path):
        logger.error(f"{checkpoint_path} is not a directory")
        checkpoint_path = None

    # List all the checkpoints.
    checkpoint_ids = []
    for file in os.listdir(checkpoint_path):
        if file[:5] == "step_":
            checkpoint_id = file.split(".")[0]
            checkpoint_ids.append(int(checkpoint_id[5:]))

    if checkpoint_ids:
        # Use the last checkpoint.
        if checkpoint == "last":
            checkpoint_id = max(checkpoint_ids)
            checkpoint_path = os.path.join(
                checkpoint_path, f"step_{checkpoint_id}"
            )

        # Use the specified checkpoint.
        else:
            checkpoint_id = int(checkpoint)
            if checkpoint_id in checkpoint_ids:
                checkpoint_path = os.path.join(
                    checkpoint_path, f"step_{checkpoint_id}"
                )
            else:
                logger.error(
                    f"Checkpoint {checkpoint_id} "
                    f"not found in {checkpoint_path}"
                )
                checkpoint_path = None

    else:
        logger.error(f"No checkpoint found in {checkpoint_path}")
        checkpoint_path = None

    # Load the experiment configuration.
    arguments_path = os.path.join(path, "config.yaml")
    with open(arguments_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = argparse.Namespace(**config)
    return config, checkpoint_path


def mujoco_render(env, *args, **kwargs):
    if "mujoco_py" in str(type(env.sim)):
        env.render(*args, **kwargs)
    else:
        env.sim.renderer.render_to_window(*args, **kwargs)


@contextmanager
def stdout_suppression():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
