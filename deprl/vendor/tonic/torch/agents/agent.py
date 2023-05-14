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

    def save(self, path):
        path = path + ".pt"
        logger.log(f"\nSaving weights to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        path = path + ".pt"
        logger.log(f"\nLoading weights from {path}")
        if torch.cuda.is_available():
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location="cpu")
        try:
            if "critic.torso.model.0.weight" in state_dict.keys():
                logger.log("Loading full model.")
                self._load_weights(state_dict, full=True)
            else:
                logger.log("Loading only actor weights.")
                self._load_weights(state_dict, full=False)
        except RuntimeError as e:
            logger.log(f"Loading failed, policy mismatch with checkpoint: {e}")

    def _load_weights(self, state_dict, full=False):
        if full:
            self.model.load_state_dict(state_dict)
        else:
            self.model.actor.load_state_dict(state_dict)
