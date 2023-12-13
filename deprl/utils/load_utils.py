import os

import gdown
import torch
import yaml

from deprl.vendor.tonic import logger


def load(path, environment, checkpoint="last"):
    config, checkpoint_path, _ = load_checkpoint(path, checkpoint)
    header = config["tonic"]["header"]
    agent = config["tonic"]["agent"]
    # Run the header
    exec(header)
    # Build the agent.
    agent = eval(agent)
    # Adapt mpo specific settings
    if "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
    )
    # Load the weights of the agent form a checkpoint.
    agent.load(checkpoint_path, only_checkpoint=True)
    return agent


def load_time_dict(checkpoint_path):
    try:
        return torch.load(os.path.join(checkpoint_path, "time.pt"))
    except FileNotFoundError:
        logger.log(
            "Found only the policy checkpoint, the previous run was likely only run with  <'full_save': False>"
        )
        logger.log("Only loading policy checkpoint.")
        return None


def load_checkpoint(checkpoint_path, checkpoint="last"):
    """
    Checkpoint loading for main() function.
    """
    if not os.path.isdir(checkpoint_path):
        return None, None, None

    path = (
        checkpoint_path
        if "checkpoints" not in checkpoint_path
        else checkpoint_path.split("checkpoints")[0]
    )
    if not os.path.exists(os.path.join(path, "config.yaml")):
        raise FileNotFoundError(
            f"The given path does not contain a <config.yaml> file: {path}"
        )
    if checkpoint_path.split(os.sep)[-1] != "checkpoints":
        checkpoint_path += "checkpoints"
    logger.log(f"Loading experiment from {checkpoint_path}")
    time_dict = load_time_dict(checkpoint_path)

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
    arguments_path = os.path.join(
        checkpoint_path.split("checkpoints")[0], "config.yaml"
    )
    with open(arguments_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config, checkpoint_path, time_dict


# All kinds of pretrained baselines
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
        return load_baseline_myochasetagp1(environment)
    if "myoChallengeRelocateP1" in identifier:
        logger.log("Load RelocateP1 Baseline")
        return load_baseline_myorelocatep1(environment)
    if "sconewalk_h0918" in identifier:
        logger.log("Load SconeWalk H0918 Baseline")
        return load_baseline_sconewalk_h0918(environment)
    if "sconewalk_h1622" in identifier:
        logger.log("Load SconeWalk H1622 Baseline")
        return load_baseline_sconewalk_h1622(environment)
    if "sconewalk_h2190" in identifier:
        logger.log("Load SconeWalk H2190 Baseline")
        return load_baseline_sconewalk_h2190(environment)
    if "sconerun_h0918" in identifier:
        logger.log("Load SconeRun H0918 Baseline")
        return load_baseline_sconerun_h0918(environment)
    if "sconerun_h1622" in identifier:
        logger.log("Load SconeRun H1622 Baseline")
        return load_baseline_sconerun_h1622(environment)
    if "sconerun_h2190" in identifier:
        logger.log("Load SconeRun H2190 Baseline")
        return load_baseline_sconerun_h2190(environment)
    raise NotImplementedError(
        "The chosen environment has no pre-trained baseline."
    )


# MyoSuite Baselines
def load_baseline_myolegwalk(environment):
    modelurl = (
        "https://drive.google.com/uc?id=1UkiUozk-PM8JbQCZNoy2Jr2t1StLcAzx"
    )
    configurl = (
        "https://drive.google.com/uc?id=1knsol05ZL14aqyuaT-TlOvahr31gKQwE"
    )
    foldername = "./baselines_DEPRL/myoLegWalk_20230514/myoLeg/"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_150000000.pt")
    configpath = os.path.join(foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment)


def load_baseline_myochasetagp1(environment):
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
    return load(foldername, environment)


def load_baseline_myorelocatep1(environment):
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
    return load(foldername, environment)


# Hyfydy Baselines
def load_baseline_sconewalk_h0918(environment):
    modelurl = (
        "https://drive.google.com/uc?id=1TM86OHWa06m0at-4-tBdNLRIcj7iCkEb"
    )
    configurl = (
        "https://drive.google.com/uc?id=1hlgK3JdjKe0F3-JOsoZHhdpOoxwtfpoc"
    )
    foldername = "./baselines_DEPRL/sconewalk_h0918/"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_50000000.pt")
    configpath = os.path.join(foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment)


def load_baseline_sconewalk_h1622(environment):
    modelurl = (
        "https://drive.google.com/uc?id=1ddx-S9uxk34Q1kWDjco0aKpBURkGpFiS"
    )
    configurl = (
        "https://drive.google.com/uc?id=1ygFFYAOpifC_EV017v_H9EZnDu-hOfRO"
    )
    foldername = "./baselines_DEPRL/sconewalk_h1622/"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_37000000.pt")
    configpath = os.path.join(foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment)


def load_baseline_sconewalk_h2190(environment):
    modelurl = (
        "https://drive.google.com/uc?id=1Np-XuD0VNquJaM8OREi8op5Xh9U05Br1"
    )
    configurl = (
        "https://drive.google.com/uc?id=1khC-55_nfz5uMsRUJN8QYjac3UEUlhEh"
    )
    foldername = "./baselines_DEPRL/sconewalk_h2190/"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_50000000.pt")
    configpath = os.path.join(foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment)


def load_baseline_sconerun_h0918(environment):
    modelurl = (
        "https://drive.google.com/uc?id=1YLnFpY-2T4JTe3ds8FO_C68thXuUXwVj"
    )
    configurl = (
        "https://drive.google.com/uc?id=1n9EfnTzoAzeRx2jO5GXlOSJNt0glTxVh"
    )
    foldername = "./baselines_DEPRL/sconerun_h0918/"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_101000000.pt")
    configpath = os.path.join(foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment)


def load_baseline_sconerun_h1622(environment):
    modelurl = (
        "https://drive.google.com/uc?id=1O8SUkEYiFX2HfUW2NEvpVnNJveymIjha"
    )
    configurl = (
        "https://drive.google.com/uc?id=1YU2FC1h_FoAFMMuViLCp0ldgUqPh6W_a"
    )
    foldername = "./baselines_DEPRL/sconerun_h1622/"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_100000000.pt")
    configpath = os.path.join(foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment)


def load_baseline_sconerun_h2190(environment):
    modelurl = (
        "https://drive.google.com/uc?id=1L_ohXkPCWW1n8TgsLZDEOJjXjz6KYkKa"
    )
    configurl = (
        "https://drive.google.com/uc?id=18mECgv2I7UnA8m4RuhRjdKZr8L7y3p5T"
    )
    foldername = "./baselines_DEPRL/sconerun_h2190/"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_10000000.pt")
    configpath = os.path.join(foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment)
