"""Script used to train agents."""
import os
import traceback

import torch

from deprl import custom_distributed
from deprl.utils import load_config_and_paths, prepare_params
from deprl.vendor.tonic import logger


def train(
    config,
):
    """
    Trains an agent on an environment.
    """
    tonic_conf = config["tonic"]

    path = os.path.join(config["working_dir"], tonic_conf["name"])

    # Process the checkpoint path same way as in tonic_conf.play
    checkpoint_path = os.path.join(path, "checkpoints")

    time_dict = {"steps": 0, "epochs": 0, "episodes": 0}
    (
        loaded_config,
        checkpoint_path,
        loaded_time_dict,
    ) = load_config_and_paths(checkpoint_path, checkpoint="last")
    time_dict = time_dict if loaded_time_dict is None else loaded_time_dict
    config = config if loaded_config is None else loaded_config

    # Run the header first, e.g. to load an ML framework.
    if "header" in tonic_conf:
        exec(tonic_conf["header"])
    # In case no env_args are passed via the config
    if "env_args" not in config or config["env_args"] is None:
        config["env_args"] = {}
    # Build the training environment.
    _environment = tonic_conf["environment"]
    environment = custom_distributed.distribute(
        environment=_environment,
        tonic_conf=tonic_conf,
        env_args=config["env_args"],
    )
    # TODO check if this neesd to be changed
    environment.initialize(seed=tonic_conf["seed"])
    # Build the testing environment.
    _test_environment = (
        tonic_conf["test_environment"]
        if "test_environment" in tonic_conf
        and tonic_conf["test_environment"] is not None
        else _environment
    )
    test_env_args = (
        config["test_env_args"]
        if "test_env_args" in config
        else config["env_args"]
    )
    test_environment = custom_distributed.distribute(
        environment=_test_environment,
        tonic_conf=tonic_conf,
        env_args=test_env_args,
        parallel=1,
        sequential=1,
    )
    test_environment.initialize(seed=tonic_conf["seed"] + 1000000)

    # Build the agent.
    if "agent" not in tonic_conf or tonic_conf["agent"] is None:
        raise ValueError("No agent specified.")
    agent = eval(tonic_conf["agent"])

    # Set custom mpo parameters
    if "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=tonic_conf["seed"],
    )
    # Set DEP parameters
    if hasattr(agent, "expl") and "DEP" in config:
        agent.expl.set_params(config["DEP"])

    # Initialize the logger to save data to the path environment/name/seed.
    logger.initialize(path, script_path=__file__, config=config)

    if checkpoint_path:
        # Load the logger from a checkpoint.
        logger.load(checkpoint_path, time_dict)
        # Load the weights of the agent form a checkpoint.
        agent.load(checkpoint_path)

    # Build the trainer.
    trainer = tonic_conf["trainer"] or "tonic_conf.Trainer()"
    trainer = eval(trainer)
    trainer.initialize(
        agent=agent,
        environment=environment,
        test_environment=test_environment,
        full_save=tonic_conf["full_save"],
    )

    # Run some code before training.
    if tonic_conf["before_training"]:
        exec(tonic_conf["before_training"])

    # Train.
    try:
        trainer.run(config, **time_dict)
    except Exception as e:
        logger.log(f"trainer failed. Exception: {e}")
        traceback.print_tb(e.__traceback__)

    # Run some code after training.
    if tonic_conf["after_training"]:
        exec(["after_training"])


def main():
    # use CUDA or apple metal
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")
    else:
        logger.log("No CUDA or MPS detected, running on CPU")

    config = prepare_params()
    train(config)


if __name__ == "__main__":
    main()
