import numpy as np

from .wrappers import ExceptionWrapper


class CustomSconeException(Exception):
    """
    Custom exception class for Scone. The SconePy Interface doesn't define a scone exception
    at the moment and I have never encountered a simulator-based exception until now.
    Will update when that is the case.
    """

    pass


class SconeWrapper(ExceptionWrapper):
    """Wrapper for SconeRL, compatible with
    gym=0.13.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error = CustomSconeException

    def render(self, *args, **kwargs):
        pass

    def muscle_lengths(self):
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
            raise Exception("You have to call reset() once before step()")

        if self.unwrapped.clip_actions:
            action = np.clip(action, 0, 0.5)
        else:
            action = np.clip(action, 0, 1.0)

        if self.unwrapped.use_delayed_actuators:
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

    @property
    def _max_episode_steps(self):
        return 1000

    @property
    def results_dir(self):
        return self.unwrapped.results_dir
