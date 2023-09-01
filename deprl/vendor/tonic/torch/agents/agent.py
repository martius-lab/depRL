import os
import random

import numpy as np
import torch

from deprl.vendor.tonic import agents
from deprl.vendor.tonic.utils import logger


class Agent(agents.Agent):
    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

    def save(self, path, full_save=True):
        path = path + ".pt"
        logger.log(f"\nSaving weights to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        if full_save:
            logger.log("Saving full model")
            self.save_optimizer(path)
            self.save_buffer(path)
            self.save_observation_normalizer(path)
            self.save_return_normalizer(path)

    def get_device(self):
        # support for cpu, cuda and mps
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def load(self, path, only_checkpoint=False):
        path = path + ".pt"
        logger.log(f"\nLoading weights from {path}")
        self.get_device()
        try:
            self._load_weights(path)
        except Exception as e:
            logger.log("Error, not loading model")
            logger.log(f"{e=}")
        if not only_checkpoint:
            try:
                self.load_optimizer(path)
                self.load_buffer(path)
                self.load_observation_normalizer(path)
                self.load_return_normalizer(path)
            except Exception:
                logger.log(
                    "Could not find full model, only loading policy checkpoint."
                )

    def save_return_normalizer(self, path):
        if self.model.return_normalizer is not None:
            reno = self.model.return_normalizer
            norm_path = self.get_path(path, "ret_norm")
            ret_norm_dict = {
                "min_reward": reno.min_reward,
                "max_reward": reno.max_reward,
                "_low": reno._low,
                "_high": reno._high,
                "coefficient": reno.coefficient,
            }
            torch.save(ret_norm_dict, norm_path)

    def save_observation_normalizer(self, path):
        if (
            hasattr(self.model, "observation_normalizer")
            and self.model.observation_normalizer is not None
        ):
            ono = self.model.observation_normalizer
            norm_path = self.get_path(path, "obs_norm")
            obs_norm_dict = {
                "clip": ono.clip,
                "count": ono.count,
                "mean": ono.mean,
                "mean_sq": ono.mean_sq,
                "std": ono.std,
                "_mean": ono._mean,
                "_std": ono._std,
                "new_sum": ono.new_sum,
                "new_sum_sq": ono.new_sum_sq,
                "new_count": ono.new_count,
            }
            torch.save(obs_norm_dict, norm_path)

    def load_observation_normalizer(self, path):
        if self.model.observation_normalizer is not None:
            try:
                norm_path = self.get_path(path, "obs_norm")
                load_dict = torch.load(norm_path, map_location=self.device)
                for k, v in load_dict.items():
                    setattr(self.model.observation_normalizer, k, v)
            except Exception:
                logger.log("Not loading observation normalizer")

    def load_return_normalizer(self, path):
        if self.model.return_normalizer is not None:
            try:
                norm_path = self.get_path(path, "ret_norm")
                load_dict = torch.load(norm_path, map_location=self.device)
                for k, v in load_dict.items():
                    setattr(self.model.return_normalizer, k, v)
            except Exception:
                logger.log("Not loading return normalizer")

    def save_optimizer(self, path):
        if hasattr(self, "actor_updater"):
            if hasattr(self.actor_updater, "optimizer"):
                opt_path = self.get_path(path, "actor")
                torch.save(self.actor_updater.optimizer.state_dict(), opt_path)
            else:
                # so far, only MPO has different optimizers
                opt_path = self.get_path(path, "actor")
                torch.save(
                    self.actor_updater.actor_optimizer.state_dict(), opt_path
                )
                opt_path = self.get_path(path, "dual")
                torch.save(
                    self.actor_updater.dual_optimizer.state_dict(), opt_path
                )
        if hasattr(self, "critic_updater"):
            opt_path = self.get_path(path, "critic")
            torch.save(self.critic_updater.optimizer.state_dict(), opt_path)

    def load_optimizer(self, path):
        if hasattr(self, "actor_updater"):
            if hasattr(self.actor_updater, "optimizer"):
                opt_path = self.get_path(path, "actor")
                self.actor_updater.optimizer.load_state_dict(
                    torch.load(opt_path, map_location=self.device)
                )
            else:
                opt_path = self.get_path(path, "actor")
                self.actor_updater.actor_optimizer.load_state_dict(
                    torch.load(opt_path, map_location=self.device)
                )
                opt_path = self.get_path(path, "dual")
                self.actor_updater.dual_optimizer.load_state_dict(
                    torch.load(opt_path, map_location=self.device)
                )

        if hasattr(self, "critic_updater"):
            opt_path = self.get_path(path, "critic")
            self.critic_updater.optimizer.load_state_dict(
                torch.load(opt_path, map_location=self.device)
            )

    def save_buffer(self, path):
        self.replay.save(path)

    def load_buffer(self, path):
        self.replay.load(path, self.device)

    def get_path(self, path, post_fix):
        return path.split("step")[0] + post_fix + ".pt"

    def _load_weights(self, path):
        """
        Load weights either for full model or just for actor
        checkpoint. Works with cuda and non-cuda checkppoints, regardless
        what was saved.
        """
        state_dict = torch.load(path, map_location=self.device)
        if "critic.torso.model.0.weight" in state_dict.keys():
            self.model.load_state_dict(state_dict)
        else:
            self.model.actor.load_state_dict(state_dict)

    def noisy_test_step(self, observations, *args, **kwargs):
        return self._step(observations).numpy(force=True)
