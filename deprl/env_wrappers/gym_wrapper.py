from deprl.vendor.tonic import logger

from .wrappers import ExceptionWrapper


class DummyException(Exception):
    pass


class GymWrapper(ExceptionWrapper):
    """Wrapper for OpenAI Gym and MuJoCo, compatible with
    gym=0.13.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dummy_counter = 2
        try:
            from mujoco_py.builder import MujocoException

            error_mjpy = MujocoException
            dummy_counter = 0
        except ModuleNotFoundError:
            error_mjpy = DummyException

        try:
            from dm_control.rl.control import PhysicsError

            error_mj = PhysicsError
            dummy_counter = 1

        except ModuleNotFoundError:
            error_mj = DummyException

        if dummy_counter == 2:
            logger.log(
                "Neither mujoco nor mujoco_py has been detected. GymWrapper is not catching exceptions correctly."
            )
        self.error = (error_mjpy, error_mj, DummyException)[dummy_counter]

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
