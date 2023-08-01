"""Script used to train agents."""
import argparse
import os
import traceback

import torch
import yaml

from deprl import custom_distributed
from deprl.utils import prepare_params
from deprl.vendor import tonic


def maybe_load_checkpoint(
    header,
    agent,
    environment,
    trainer,
    time_dict,
    checkpoint_path,
    checkpoint,
    eff_path,
):
    if os.path.isdir(checkpoint_path):
        tonic.logger.log(f"Loading experiment from {eff_path}")
        # Use no checkpoint, the agent is freshly created.
        if checkpoint == "none":
            tonic.logger.log("Not loading any weights")

        else:
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
                        tonic.logger.error(
                            f"Checkpoint {checkpoint_id} not found in {checkpoint_path}"
                        )
                        checkpoint_path = None
            else:
                tonic.logger.error(f"No checkpoint found in {checkpoint_path}")
                checkpoint_path = None

        # Load the experiment configuration.
        arguments_path = os.path.join(eff_path, "config.yaml")
        with open(arguments_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = argparse.Namespace(**config)

        header = header or config.header
        agent = agent or config.agent
        environment = environment or config.test_environment
        environment = environment or config.environment
        trainer = trainer or config.trainer
        return header, agent, environment, trainer, time_dict, checkpoint_path

    else:
        checkpoint_path = None
        return header, agent, environment, trainer, time_dict, checkpoint_path


def train(
    orig_params,
    header,
    agent,
    environment,
    test_environment,
    trainer,
    before_training,
    after_training,
    parallel,
    sequential,
    seed,
    name,
    environment_name,
    checkpoint,
    path,
    preid=0,
    env_args=None,
):
    """Trains an agent on an environment."""
    # Capture the arguments to save them, e.g. to play with the trained agent.
    # TODO fix this mess and do it properly
    args = dict(locals())
    del args["orig_params"]
    if args["env_args"]:
        args["env_args"] = dict(args["env_args"])
        if "target" in args["env_args"]:
            args["env_args"]["target"] = list(args["env_args"]["target"])
        if "rew_args" in args["env_args"]:
            args["env_args"]["rew_args"] = dict(args["env_args"]["rew_args"])
    if "mpo_args" in orig_params:
        args["mpo_args"] = dict(orig_params["mpo_args"])

    eff_path = os.path.join(path, environment_name, name)
    # Process the checkpoint path same way as in tonic.play
    checkpoint_path = os.path.join(eff_path, "checkpoints")
    time_dict = {"steps": 0, "epochs": 0, "episodes": 0}
    (
        header,
        agent,
        environment,
        trainer,
        time_dict,
        checkpoint_path,
    ) = maybe_load_checkpoint(
        header,
        agent,
        environment,
        trainer,
        time_dict,
        checkpoint_path,
        checkpoint,
        eff_path,
    )
    # Run the header first, e.g. to load an ML framework.
    if header:
        exec(header)
    if "env_args" not in locals():
        env_args = {}
    else:
        env_args = dict(env_args)
    env_args["starting_steps"] = time_dict["steps"]
    # Build the training environment.
    _environment = environment
    environment = custom_distributed.distribute(
        dict(
            env=_environment,
            preid=preid,
            parallel=parallel,
            sequential=sequential,
        ),
        parallel,
        sequential,
        env_args=env_args,
    )
    environment.initialize(seed=seed)
    # Build the testing environment.
    _test_environment = test_environment if test_environment else _environment
    test_env_args = dict(env_args)
    if "Baoding" in _test_environment:
        test_env_args["step_fn"] = "step_baoding_test"
    else:
        test_env_args["step_fn"] = "step_die"
    test_environment = custom_distributed.distribute(
        dict(env=_test_environment, preid=preid + 1000000),
        env_args=test_env_args,
    )
    test_environment.initialize(seed=seed + 1000000)

    # Build the agent.
    if not agent:
        raise ValueError("No agent specified.")
    agent = eval(agent)
    if "mpo_args" in orig_params:
        agent.set_params(**orig_params["mpo_args"])
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )
    if hasattr(agent, "expl") and "DEP" in orig_params:
        agent.expl.set_params(orig_params["DEP"])
    # Load the weights of the agent form a checkpoint.
    if checkpoint_path:
        agent.load(checkpoint_path)

    # Initialize the logger to save data to the path environment/name/seed.
    if not environment_name:
        if hasattr(test_environment, "name"):
            environment_name = test_environment.name
        else:
            environment_name = test_environment.__class__.__name__
    if not name:
        if hasattr(agent, "name"):
            name = agent.name
        else:
            name = agent.__class__.__name__
        if parallel != 1 or sequential != 1:
            name += f"-{parallel}x{sequential}"
    eff_path = os.path.join(path, environment_name, name)

    tonic.logger.initialize(eff_path, script_path=__file__, config=args)

    # Build the trainer.
    trainer = trainer or "tonic.Trainer()"
    trainer = eval(trainer)
    trainer.initialize(
        agent=agent, environment=environment, test_environment=test_environment
    )

    # Run some code before training.
    if before_training:
        exec(before_training)

    # Train.
    try:
        scores = trainer.run(orig_params, **time_dict)
    except Exception as e:
        tonic.logger.log(f"trainer failed. Exception: {e}")
        traceback.print_tb(e.__traceback__)

    # Run some code after training.
    if after_training:
        exec(after_training)


if __name__ == "__main__":
    try:
        torch.zeros((0, 1), device="cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    except Exception as e:
        print(f"No cuda detected, running on cpu: {e}")
    orig_params, params = prepare_params()
    train_params = dict(orig_params["tonic"])
    train_params["path"] = orig_params["working_dir"]
    train_params["preid"] = orig_params["id"]
    if "env_args" in orig_params or "env_args" in train_params:
        train_params["env_args"] = (
            orig_params["env_args"]
            if "env_args" in orig_params
            else train_params["env_args"]
        )
    train(orig_params, **train_params)
