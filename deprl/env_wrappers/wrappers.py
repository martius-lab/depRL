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
    def tendon_states(self):
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


class GymWrapper(ExceptionWrapper):
    """Wrapper for OpenAI Gym and MuJoCo, compatible with
    gym=0.13.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from mujoco_py.builder import MujocoException

        self.error = MujocoException

    def render(self, *args, **kwargs):
        kwargs["mode"] = "window"
        self.unwrapped.sim.render(*args, **kwargs)

    def muscle_lengths(self):
        length = self.unwrapped.sim.data.actuator_length
        return length

    def muscle_forces(self):
        return self.unwrapped.sim.data.actuator_force

    def muscle_velocities(self):
        return self.unwrapped.sim.data.actuator_velocity

    def muscle_activity(self):
        return self.unwrapped.sim.data.act

    @property
    def _max_episode_steps(self):
        return self.unwrapped.max_episode_steps


class CustomSconeException(Exception):
    """
    Custom exception class for Scone. The SconePy Interface doesn't define a scone exception at the moment and I have never encountered a simulator-based exception until now. Will update when that is the case.
    """
    pass


class SconeWrapper(ExceptionWrapper):
    """Wrapper for SconeRL, compatible with
    gym=0.13.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error = CustomSconeException

<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 0de73ee (fixed wrapper for sconerl)
=======
>>>>>>> 8eac176 (starting compatibility with sconegym-dev)
    def render(self, *args, **kwargs):
        pass

    def muscle_lengths(self):
<<<<<<< HEAD
        length = self.unwrapped.model.muscle_fiber_length_array()
        return length

    def muscle_forces(self):
        force = self.unwrapped.model.muscle_force_array()
        return force

    def muscle_velocities(self):
        velocity = self.unwrapped.model.muscle_fiber_velocity_array()
        return velocity

    def muscle_activity(self):
        return self.unwrapped.model.muscle_activation_array()

    def write_now(self):
        if self.unwrapped.store_next:
            self.model.write_results(
                self.output_dir, f"{self.episode:05d}_{self.total_reward:.3f}"
            )
        self.episode += 1
        self.unwrapped.store_next = False

    def _inner_step(self, action):
        """
        takes an action and advances environment by 1 step.
        Changed to allow for correct sto saving.
        """
        if not self.unwrapped.has_reset:
<<<<<<< HEAD
<<<<<<< HEAD
            raise Exception("You have to call reset() once before step()")
=======
            raise Exception('You have to call reset() once before step()')
>>>>>>> d9c3989 (made everything compatible with default sconerl and similar environments)
=======
            raise Exception("You have to call reset() once before step()")
>>>>>>> 8eac176 (starting compatibility with sconegym-dev)

        if self.use_delayed_actuators:
            self.unwrapped.model.set_delayed_actuator_inputs(action)
        else:
            self.unwrapped.model.set_actuator_inputs(action)

        self.unwrapped.model.advance_simulation_to(self.time + self.step_size)
        reward = self.unwrapped._get_rew()
        obs = self.unwrapped._get_obs()
        done = self.unwrapped._get_done()
        self.unwrapped.time += self.step_size
        self.unwrapped.total_reward += reward
        return obs, reward, done, {}
<<<<<<< HEAD
=======
        length = self.model.muscle_fiber_length_array()
        return length

    def muscle_forces(self):
        force = self.model.muscle_force_array()
        return force

    def muscle_velocities(self):
        velocity = self.model.muscle_fiber_velocity_array()
        return velocity

    def muscle_activity(self):
        return self.model.muscle_activation_array()
>>>>>>> 0de73ee (fixed wrapper for sconerl)
=======
>>>>>>> d9c3989 (made everything compatible with default sconerl and similar environments)

    @property
    def _max_episode_steps(self):
        return 1000


class DMWrapper(ExceptionWrapper):
    """
    Wrapper for general DeepMind ControlSuite environments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from dm_control.rl.control import PhysicsError

        self.error = PhysicsError

    def muscle_lengths(self):
        length = self.unwrapped.environment.physics.data.actuator_length
        return length

    def muscle_forces(self):
        return self.unwrapped.environment.physics.data.actuator_force

    def muscle_velocities(self):
        return self.unwrapped.environment.physics.data.actuator_velocity

    def muscle_activity(self):
        return self.unwrapped.environment.physics.data.act

    @property
    def _max_episode_steps(self):
        return self.unwrapped.max_episode_steps


class OstrichDMWrapper(DMWrapper):
    """
    Wrapper explicitly for the OstrichRL environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def muscle_lengths(self):
        return self.unwrapped.environment.physics.muscle_lengths().copy()

    def muscle_forces(self):
        return self.unwrapped.environment.physics.muscle_forces().copy()

    def muscle_velocities(self):
        return self.unwrapped.environment.physics.muscle_velocities().copy()

    def muscle_activity(self):
        return self.unrapped.environment.physics.muscle_activations().copy()


def apply_wrapper(env):
<<<<<<< HEAD
    if "control" in str(env).lower():
        if env.name == "ostrich-run":
            return OstrichDMWrapper(env)
        return DMWrapper(env)
    elif "scone" in str(env).lower():
=======
    print(type(env))
    if "control" in str(type(env)).lower():
        if env.name == "ostrich-run":
            return OstrichDMWrapper(env)
        return DMWrapper(env)
    elif "scone" in str(type(env)).lower():
>>>>>>> 0de73ee (fixed wrapper for sconerl)
        return SconeWrapper(env)
    else:
        return GymWrapper(env)


def env_tonic_compat(env, preid=5, parallel=1, sequential=1):
    """
    Applies wrapper for tonic and passes random seed.
    """
    return lambda identifier=0: apply_wrapper(eval(env))
