import numpy as np
import torch

from deprl.vendor.tonic.utils import logger

NPARTITIONS = 10


class Buffer:
    """Replay storing a large number of transitions for off-policy learning
    and using n-step returns."""

    def __init__(
        self,
        size=int(1e6),
        return_steps=1,
        batch_iterations=50,
        batch_size=100,
        discount_factor=0.99,
        steps_before_batches=int(1e4),
        steps_between_batches=50,
    ):
        self.full_max_size = size
        self.return_steps = return_steps
        self.batch_iterations = batch_iterations
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.steps_before_batches = steps_before_batches
        self.steps_between_batches = steps_between_batches
        # buffers HAS to be last, we need the others before to load it properly
        self.checkpoint_fields = [
            "index",
            "num_workers",
            "max_size",
            "size",
            "buffers",
        ]

    def initialize(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.buffers = None
        self.index = 0
        self.size = 0
        self.last_steps = 0

    def ready(self, steps):
        if steps < self.steps_before_batches:
            return False
        return (steps - self.last_steps) >= self.steps_between_batches

    def store(self, **kwargs):
        if "terminations" in kwargs:
            continuations = np.float32(1 - kwargs["terminations"])
            kwargs["discounts"] = continuations * self.discount_factor

        # Create the named buffers.
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.max_size = self.full_max_size // self.num_workers
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val

        # Accumulate values for n-step returns.
        if self.return_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def accumulate_n_steps(self, kwargs):
        rewards = kwargs["rewards"]
        next_observations = kwargs["next_observations"]
        discounts = kwargs["discounts"]
        masks = np.ones(self.num_workers, np.float32)

        for i in range(min(self.size, self.return_steps - 1)):
            index = (self.index - i - 1) % self.max_size
            masks *= 1 - self.buffers["resets"][index]
            new_rewards = (
                self.buffers["rewards"][index]
                + self.buffers["discounts"][index] * rewards
            )
            self.buffers["rewards"][index] = (1 - masks) * self.buffers[
                "rewards"
            ][index] + masks * new_rewards
            new_discounts = self.buffers["discounts"][index] * discounts
            self.buffers["discounts"][index] = (1 - masks) * self.buffers[
                "discounts"
            ][index] + masks * new_discounts
            self.buffers["next_observations"][index] = (1 - masks)[
                :, None
            ] * self.buffers["next_observations"][index] + masks[
                :, None
            ] * next_observations

    def get(self, *keys, steps):
        """Get batches from named buffers."""

        for _ in range(self.batch_iterations):
            total_size = self.size * self.num_workers
            indices = self.np_random.randint(total_size, size=self.batch_size)
            rows = indices // self.num_workers
            columns = indices % self.num_workers
            yield {k: self.buffers[k][rows, columns] for k in keys}

        self.last_steps = steps

    def save(self, path):
        logger.log("Saving buffer")
        if hasattr(self, "buffers"):
            for field in self.checkpoint_fields:
                if not field == "buffers":
                    save_path = self.get_path(path, field)
                    torch.save(getattr(self, field), save_path)
                else:
                    self.save_buffer_incrementally(path)

    def save_buffer_incrementally(self, path):
        """
        Helps saving big buffers that would otherwise fill up
        system RAM completely.
        """
        for i in range(NPARTITIONS):
            partition = int(self.max_size / NPARTITIONS)
            save_path = self.get_path(path, f"buffer_{i * partition}")
            buffer_save_dict = {}
            for k, v in self.buffers.items():
                if len(v.shape) > 2:
                    buffer_save_dict[k] = v[
                        i * partition : i * partition + partition, :, :
                    ]
                else:
                    buffer_save_dict[k] = v[
                        i * partition : i * partition + partition, :
                    ]
            torch.save(buffer_save_dict, save_path)

    def load_buffer_incrementally(self, path, device):
        """
        Helps loading big buffers that would otherwise fill up
        system RAM completely.
        """
        for i in range(NPARTITIONS):
            partition = int(self.max_size / NPARTITIONS)
            load_path = self.get_path(path, f"buffer_{i * partition}")
            buffer_load_dict = torch.load(load_path, map_location=device)
            if self.buffers is None:
                self.create_empty_buffer(buffer_load_dict)
            for k, v in buffer_load_dict.items():
                if len(v.shape) > 2:
                    self.buffers[k][
                        i * partition : i * partition + partition, :, :
                    ] = v
                else:
                    self.buffers[k][
                        i * partition : i * partition + partition, :
                    ] = v

    def create_empty_buffer(self, buffer_load_dict):
        self.buffers = {}
        for k, v in buffer_load_dict.items():
            if len(v.shape) > 2:
                self.buffers[k] = np.full(
                    (self.max_size, v.shape[-2], v.shape[-1]),
                    np.nan,
                    np.float32,
                )
            else:
                self.buffers[k] = np.full(
                    (self.max_size, v.shape[-1]), np.nan, np.float32
                )

    def load(self, path, device="cpu"):
        logger.log("Loading buffer")
        try:
            if hasattr(self, "buffers"):
                for field in self.checkpoint_fields:
                    if not field == "buffers":
                        buffer_path = self.get_path(path, field)
                        setattr(
                            self,
                            field,
                            torch.load(buffer_path, map_location=device),
                        )
                    else:
                        self.load_buffer_incrementally(path, device)
        except Exception as e:
            print(f"Error in buffer loading, it is freshly initialized: {e}")

    def get_path(self, path, post_fix):
        return path.split("step")[0] + post_fix + ".pt"
