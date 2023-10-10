import os
import time

import numpy as np
import torch

from deprl.custom_test_environment import test_dm_control, test_mujoco
from deprl.vendor.tonic import logger

if "ROBOHIVE_VERBOSITY" not in os.environ:
    os.environ["ROBOHIVE_VERBOSITY"] = "ALWAYS"


class Trainer:
    """Trainer used to train and evaluate an agent on an environment."""

    def __init__(
        self,
        steps=1e7,
        epoch_steps=2e4,
        save_steps=5e5,
        test_episodes=20,
        show_progress=True,
        replace_checkpoint=False,
    ):
        self.max_steps = int(steps)
        self.epoch_steps = int(epoch_steps)
        self.save_steps = int(save_steps)
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint

    def initialize(
        self, agent, environment, test_environment=None, full_save=False
    ):
        self.full_save = full_save
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def run(self, params, steps=0, epochs=0, episodes=0):
        """Runs the main training loop."""

        start_time = last_epoch_time = time.time()

        # Start the environments.
        observations, muscle_states = self.environment.start()

        num_workers = len(observations)
        scores = np.zeros(num_workers)
        lengths = np.zeros(num_workers, int)
        self.steps, epoch_steps = steps, 0
        steps_since_save = 0

        while True:
            # Select actions.
            if hasattr(self.agent, "expl"):
                greedy_episode = (
                    not episodes % self.agent.expl.test_episode_every
                )
            else:
                greedy_episode = None
            assert not np.isnan(observations.sum())
            actions = self.agent.step(
                observations, self.steps, muscle_states, greedy_episode
            )
            assert not np.isnan(actions.sum())
            # raise Exception(f'{type(self.environment.environments[0])}')
            logger.store("train/action", actions, stats=True)

            # Take a step in the environments.
            observations, muscle_states, info = self.environment.step(actions)
            if "env_infos" in info:
                info.pop("env_infos")
            self.agent.update(**info, steps=self.steps)

            scores += info["rewards"]
            lengths += 1
            self.steps += num_workers
            epoch_steps += num_workers
            steps_since_save += num_workers

            # Show the progress bar.
            if self.show_progress:
                logger.show_progress(
                    self.steps, self.epoch_steps, self.max_steps
                )

            # Check the finished episodes.
            for i in range(num_workers):
                if info["resets"][i]:
                    logger.store("train/episode_score", scores[i], stats=True)
                    logger.store(
                        "train/episode_length", lengths[i], stats=True
                    )
                    scores[i] = 0
                    lengths[i] = 0
                    episodes += 1

            # End of the epoch.
            if epoch_steps >= self.epoch_steps:
                # Evaluate the agent on the test environment.
                if self.test_environment:
                    if "Control" not in str(
                        type(self.test_environment.environments[0].unwrapped)
                    ):
                        _ = test_mujoco(
                            self.test_environment, self.agent, steps, params
                        )
                    else:
                        _ = test_dm_control(
                            self.test_environment, self.agent, steps, params
                        )

                # Log the data.
                epochs += 1
                current_time = time.time()
                epoch_time = current_time - last_epoch_time
                sps = epoch_steps / epoch_time
                logger.store("train/episodes", episodes)
                logger.store("train/epochs", epochs)
                logger.store("train/seconds", current_time - start_time)
                logger.store("train/epoch_seconds", epoch_time)
                logger.store("train/epoch_steps", epoch_steps)
                logger.store("train/steps", self.steps)
                logger.store("train/worker_steps", self.steps // num_workers)
                logger.store("train/steps_per_second", sps)
                last_epoch_time = time.time()
                epoch_steps = 0

                logger.dump()

            # End of training.
            stop_training = self.steps >= self.max_steps

            # Save a checkpoint.
            if stop_training or steps_since_save >= self.save_steps:
                path = os.path.join(logger.get_path(), "checkpoints")
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith("step_"):
                            os.remove(os.path.join(path, file))
                checkpoint_name = f"step_{self.steps}"
                save_path = os.path.join(path, checkpoint_name)
                # save agent checkpoint
                self.agent.save(save_path, full_save=self.full_save)
                # save logger checkpoint
                logger.save(save_path)
                # save time iteration dict
                self.save_time(save_path, epochs, episodes)
                steps_since_save = self.steps % self.save_steps
                current_time = time.time()

            if stop_training:
                self.close_mp_envs()
                return scores

    def close_mp_envs(self):
        for index in range(len(self.environment.processes)):
            self.environment.processes[index].terminate()
            self.environment.action_pipes[index].close()
        self.environment.output_queue.close()

    def save_time(self, path, epochs, episodes):
        time_path = self.get_path(path, "time")
        time_dict = {
            "epochs": epochs,
            "episodes": episodes,
            "steps": self.steps,
        }
        torch.save(time_dict, time_path)

    def get_path(self, path, post_fix):
        return path.split("step")[0] + post_fix + ".pt"
