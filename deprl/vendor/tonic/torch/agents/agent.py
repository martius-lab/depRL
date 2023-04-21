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
            self.model.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
