"""Script used to play with trained agents."""

import argparse
import os

import numpy as np

from deprl import env_wrappers, mujoco_render
from deprl.utils import load_checkpoint
from deprl.vendor.tonic import logger


def set_scone_save_path(checkpoint_path, env, name):
    prefix = env.results_dir
    if checkpoint_path.startswith(prefix):
        path = os.path.join(
            *checkpoint_path[len(prefix) :]
            .lstrip(os.path.sep)
            .split(os.sep)[:-2]
        )
    else:
        path = name
    checkpoint_step = checkpoint_path.split("step_")[1]
    path = os.path.join(path, f"run_checkpoint_{checkpoint_step}")
    env.set_output_dir(path)
    logger.log(
        f"Saving sto files to {os.path.join(env.results_dir, env.output_dir)}"
    )


def check_args(args):
    if args["path"] is None and args["checkpoint_file"] is None:
        raise Exception(
            "You need to specify either a <--path> or a <--checkpoint_file>"
        )

    if args["path"] is not None and args["checkpoint_file"] is not None:
        raise Exception(
            "Do not simultaneously specify <--checkpoint_file> and \
                        <--path>."
        )

    if args["checkpoint"] != "last" and args["checkpoint_file"] is not None:
        raise Exception(
            "Do not simultaneously specify a checkpoint step with <--checkpoint> and a checkpoint file with \
                        <--checkpoint_file>."
        )
    if args["checkpoint_file"] is not None:
        if ("checkpoints" not in args["checkpoint_file"]) or (
            ".pt" not in args["checkpoint_file"]
        ):
            raise Exception(
                f'Invalid <--checkpoint_file> given: {args["checkpoint_file"]}'
            )

    if args["path"] is not None:
        assert os.path.isfile(os.path.join(args["path"], "config.yaml"))


def get_paths(path, checkpoint, checkpoint_file):
    """
    Checkpoints can be given as number e.g. <--checkpoint 1000000> or as file paths
    e.g. <--checkpoint_file path/checkpoints/step_1000000.pt'>
    This function handles this functionality.
    """
    if checkpoint_file is not None:
        path = checkpoint_file.split("checkpoints")[0]
        checkpoint = checkpoint_file.split("step_")[1].split(".")[0]
    checkpoint_path = os.path.join(path, "checkpoints")
    return path, checkpoint, checkpoint_path


def play_gym(agent, environment, noisy, num_episodes, no_render):
    """Launches an agent in a Gym-based environment."""
    observations = environment.reset()
    muscle_states = environment.muscle_states

    score = 0
    length = 0
    min_reward = float("inf")
    max_reward = -float("inf")
    global_min_reward = float("inf")
    global_max_reward = -float("inf")
    steps = 0
    episodes = 0

    while True:
        if not noisy:
            actions = agent.test_step(
                observations, muscle_states=muscle_states, steps=1e6
            )
        else:
            actions = agent.noisy_test_step(
                observations, muscle_states=muscle_states, steps=1e6
            )
        if len(actions.shape) > 1:
            actions = actions[0, :]
        observations, reward, done, info = environment.step(actions)
        muscle_states = environment.muscle_states
        if not no_render:
            mujoco_render(environment)

        steps += 1
        score += reward
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
        global_min_reward = min(global_min_reward, reward)
        global_max_reward = max(global_max_reward, reward)
        length += 1

        if done or length >= environment.max_episode_steps:
            episodes += 1

            print()
            print(f"Episodes: {episodes:,}")
            print(f"Score: {score:,.3f}")
            print(f"Length: {length:,}")
            print(f"Terminal: {done:}")
            print(f"Min reward: {min_reward:,.3f}")
            print(f"Max reward: {max_reward:,.3f}")
            print(f"Global min reward: {min_reward:,.3f}")
            print(f"Global max reward: {max_reward:,.3f}")
            observations = environment.reset()
            muscle_states = environment.muscle_states

            score = 0
            length = 0
            min_reward = float("inf")
            max_reward = -float("inf")
            if episodes >= num_episodes:
                break


def play_scone(
    agent, environment, noisy, num_episodes, no_render, checkpoint_path, name
):
    """Launches an agent in a Gym-based environment."""
    set_scone_save_path(checkpoint_path, environment, name)
    if not no_render:
        environment.store_next_episode()
    observations = environment.reset()
    muscle_states = environment.muscle_states

    score = 0
    length = 0
    min_reward = float("inf")
    max_reward = -float("inf")
    global_min_reward = float("inf")
    global_max_reward = -float("inf")
    steps = 0
    episodes = 0
    while True:
        if not noisy:
            actions = agent.test_step(
                observations, muscle_states=muscle_states, steps=1e6
            )
        else:
            actions = agent.noisy_test_step(
                observations, muscle_states=muscle_states, steps=1e6
            )
        if len(actions.shape) > 1:
            actions = actions[0, :]
        observations, reward, done, info = environment.step(actions)
        muscle_states = environment.muscle_states

        steps += 1
        score += reward
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
        global_min_reward = min(global_min_reward, reward)
        global_max_reward = max(global_max_reward, reward)
        length += 1
        if done or length >= environment.max_episode_steps:
            episodes += 1

            print()
            print(f"Episodes: {episodes:,}")
            print(f"Score: {score:,.3f}")
            print(f"Length: {length:,}")
            print(f"Terminal: {done:}")
            print(f"Min reward: {min_reward:,.3f}")
            print(f"Max reward: {max_reward:,.3f}")
            print(f"Global min reward: {min_reward:,.3f}")
            print(f"Global max reward: {max_reward:,.3f}")
            if not no_render:
                environment.write_now()
                environment.store_next_episode()
            observations = environment.reset()
            muscle_states = environment.muscle_states

            score = 0
            length = 0
            min_reward = float("inf")
            max_reward = -float("inf")
            if episodes >= num_episodes:
                break


def play_control_suite(agent, environment):
    """Launches an agent in a DeepMind Control Suite-based environment."""

    from dm_control import viewer

    class Wrapper:
        """Wrapper used to plug a Tonic environment in a dm_control viewer."""

        def __init__(self, environment):
            self.environment = environment
            self.unwrapped = environment.unwrapped
            self.action_spec = self.unwrapped.environment.action_spec
            self.physics = self.unwrapped.environment.physics
            self.infos = None
            self.steps = 0
            self.episodes = 0
            self.min_reward = float("inf")
            self.max_reward = -float("inf")
            self.global_min_reward = float("inf")
            self.global_max_reward = -float("inf")
            self.max_vel = 0

        def reset(self):
            """Mimics a dm_control reset for the viewer."""

            self.observations = self.environment.reset()[None]

            self.score = 0
            self.length = 0
            self.min_reward = float("inf")
            self.max_reward = -float("inf")

            return self.unwrapped.last_time_step

        def step(self, actions):
            """Mimics a dm_control step for the viewer."""

            assert not np.isnan(actions.sum())
            ob, rew, term, _ = self.environment.step(actions[0])

            self.score += rew
            self.length += 1
            self.min_reward = min(self.min_reward, rew)
            self.max_reward = max(self.max_reward, rew)
            self.global_min_reward = min(self.global_min_reward, rew)
            self.global_max_reward = max(self.global_max_reward, rew)
            timeout = self.length == self.environment.max_episode_steps
            done = term or timeout

            if done:
                self.episodes += 1
                print()
                print(f"Episodes: {self.episodes:,}")
                print(f"Score: {self.score:,.3f}")
                print(f"Length: {self.length:,}")
                print(f"Terminal: {term:}")
                print(f"Min reward: {self.min_reward:,.3f}")
                print(f"Max reward: {self.max_reward:,.3f}")
                print(f"Global min reward: {self.min_reward:,.3f}")
                print(f"Global max reward: {self.max_reward:,.3f}")

            self.observations = ob[None]
            self.infos = dict(
                observations=ob[None],
                rewards=np.array([rew]),
                resets=np.array([done]),
                terminations=np.array([term]),
            )
            return self.unwrapped.last_time_step

        @property
        def muscle_states(self):
            return self.environment.muscle_states

    # Wrap the environment for the viewer.
    environment = Wrapper(environment)

    def policy(timestep):
        """Mimics a dm_control policy for the viewer."""
        if environment.infos is not None:
            agent.test_update(**environment.infos, steps=environment.steps)
            environment.steps += 1
        muscle_states = environment.muscle_states
        return agent.test_step(
            environment.observations,
            muscle_states=muscle_states,
            steps=environment.steps,
        )

    # Launch the viewer with the wrapped environment and policy.
    viewer.launch(environment, policy)


def play(
    path,
    checkpoint,
    seed,
    header,
    agent,
    environment,
    noisy,
    no_render,
    num_episodes,
    checkpoint_file,
):
    """Reloads an agent and an environment from a previous experiment."""

    logger.log(f"Loading experiment from {path}")
    # Load config file and checkpoint path from folder
    path, checkpoint, checkpoint_path = get_paths(
        path, checkpoint, checkpoint_file
    )
    config, checkpoint_path, _ = load_checkpoint(checkpoint_path, checkpoint)

    # Get important info from config
    header = header or config["tonic"]["header"]
    agent = agent or config["tonic"]["agent"]
    environment = environment or config["tonic"]["test_environment"]
    environment = environment or config["tonic"]["environment"]

    # Run the header first, e.g. to load an ML framework.
    if header:
        exec(header)

    # Build the agent.
    if not agent:
        raise ValueError("No agent specified.")
    agent = eval(agent)

    # Build the environment.
    environment = eval(environment)
    environment.seed(seed)
    environment = env_wrappers.apply_wrapper(environment)
    if config and "env_args" in config:
        environment.merge_args(config["env_args"])
        environment.apply_args()

    # Adapt mpo specific settings
    if config and "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )

    # Load the weights of the agent from a checkpoint.
    if checkpoint_path:
        agent.load(checkpoint_path, only_checkpoint=True)
    if "control" in str(environment).lower():
        if no_render or num_episodes != 5:
            logger.log(
                "no_render and num_episodes only implemented for gym tasks"
            )
        play_control_suite(agent, environment)
    elif "scone" in str(type(environment)).lower():
        play_scone(
            agent,
            environment,
            noisy,
            num_episodes,
            no_render,
            checkpoint_path,
            config["tonic"]["name"],
        )
    else:
        play_gym(agent, environment, noisy, num_episodes, no_render)


if __name__ == "__main__":
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--header", default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--checkpoint_file", default=None)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--environment", "--env")
    args = vars(parser.parse_args())
    check_args(args)
    play(**args)
