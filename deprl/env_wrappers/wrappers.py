from abc import ABC, abstractmethod

import gym
import numpy as np

import deprl  # noqa
from deprl.vendor.tonic import logger


class AbstractWrapper(gym.Wrapper, ABC):
    def merge_args(self, args):
        if args is not None:
            for k, v in args.items():
                setattr(self.unwrapped, k, v)

    def apply_args(self):
        pass

    def render(self, *args, **kwargs):
        pass

    @property
    def force_scale(self):
        if not hasattr(self, "_force_scale"):
            self._force_scale = 0
        return self._force_scale

    @force_scale.setter
    def force_scale(self, force_scale):
        assert force_scale >= 0, f"expected positive value, got {force_scale}"
        self._force_scale = force_scale

    @abstractmethod
    def muscle_lengths(self):
        pass

    @abstractmethod
    def muscle_forces(self):
        pass

    @property
    def muscle_states(self):
        """
        Computes the DEP input. We assume an input
        muscle_length + force_scale * muscle_force
        where force_scale is chosen by the user and the other
        variables are normalized by the encountered max and min
        values.
        """
        lce = self.muscle_lengths()
        f = self.muscle_forces()
        if not hasattr(self, "max_muscle"):
            self.max_muscle = np.zeros_like(lce)
            self.min_muscle = np.ones_like(lce) * 100.0
            self.max_force = -np.ones_like(f) * 100.0
            self.min_force = np.ones_like(f) * 100.0
        if not np.any(np.isnan(lce)):
            self.max_muscle = np.maximum(lce, self.max_muscle)
            self.min_muscle = np.minimum(lce, self.min_muscle)
        if not np.any(np.isnan(f)):
            self.max_force = np.maximum(f, self.max_force)
            self.min_force = np.minimum(f, self.min_force)
        return (
            1.0
            * (
                (
                    (lce - self.min_muscle)
                    / (self.max_muscle - self.min_muscle + 0.1)
                )
                - 0.5
            )
            * 2.0
            + self.force_scale
            * ((f - self.min_force) / (self.max_force - self.min_force + 0.1))
        ).copy()


class ExceptionWrapper(AbstractWrapper):
    """
    Catches MuJoCo related exception thrown mostly by instabilities in the simulation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        if not np.any(np.isnan(observation)):
            self.last_observation = observation.copy()
        else:
            return self.reset(**kwargs)
        return observation

    def step(self, action):
        try:
            observation, reward, done, info = self._inner_step(action)
            if np.any(np.isnan(observation)):
                raise self.error("NaN detected! Resetting.")

        except self.error as e:
            logger.log(f"Simulator exception thrown: {e}")
            observation = self.last_observation
            reward = 0
            done = 1
            info = {}
            self.reset()
        return observation, reward, done, info

    def _inner_step(self, action):
        return super().step(action)
