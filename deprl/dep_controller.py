import json
import os
from collections import deque

import gym
import jax.numpy as jnp
import numpy as onp

# jax.config.update('jax_platform_name', 'cpu')


class DEP:
    """
    DEP Implementation from Der et al.(2015).
    Jax is used instead of numpy to speed up computation, GPU strongly
    recommended.
    In the future, proper JAX features such as jit, vmap, etc
    should be used.
    """

    def __init__(self, params_path="default_path"):
        """
        Load default parameters. Should get overwritten by the experiment
        json.
        """
        if params_path == "default_path":
            dirname = os.path.dirname(__file__)
            params_path = os.path.join(
                dirname, "param_files/default_agents.json"
            )

        with open(params_path, "r") as f:
            self.params = json.load(f)["DEP"]

    def initialize(self, observation_space, action_space, seed=None):
        """
        Tonic function that saves action and observation spaces.
        """
        action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_space.shape)
        )
        self.num_sensors = action_space.shape[0]
        self.num_motors = action_space.shape[0]
        self.n_env = 1

        self.act_scale = self.act_high = action_space.high

        self.action_space = action_space
        self.obs_spec = observation_space.shape
        self.set_params(self.params)

    def step(self, observations, steps=None):
        """
        Main step function. Takes in an observation consisting of
        muscle_lengths + alpha * muscle_forces. Alpha can also be 0.
        """
        if observations.shape != self.obs_spec:
            self.obs_spec = observations.shape
            self._reset(observations.shape)
        if len(observations.shape) == 1:
            observations = observations[jnp.newaxis, :]
        return onp.array(self._get_action(observations)).copy()

    def set_params(self, param_dict):
        for k, v in param_dict.items():
            setattr(self, k, v)
        self._reset()

    def _reset(self, obs_shape=None):
        """
        Reset the controller and setup the buffers and matrices.
        We use a shape of [n_env, num_motors, num_sensors] to have
        a separate DEP matrix per parallel environment, increasing the
        diversity of the exploration data. At the moment,
        it is assumed that num_motors = num_sensors, where the sensors
        directly mirror an actuator state. E.g. joint angle or muscle
        length.
        """
        if obs_shape:
            self.n_env = [obs_shape[0] if len(obs_shape) > 1 else 1][0]
        # Identity model matrix
        self.M = jnp.broadcast_to(
            -jnp.eye(self.num_motors, self.num_sensors),
            (self.n_env, self.num_motors, self.num_sensors),
        )
        # Unnormalized controller matrix
        self.C = jnp.zeros((self.n_env, self.num_motors, self.num_sensors))
        # Normalized controller matrix
        self.C_norm = jnp.zeros(
            (self.n_env, self.num_motors, self.num_sensors)
        )
        # Controller biases
        self.Cb = jnp.zeros((self.n_env, self.num_motors))
        # Filtered observation
        self.obs_smoothed = jnp.zeros((self.n_env, self.num_sensors))
        # Observation and action buffer
        self.buffer = deque(maxlen=self.buffer_size)
        # smoothed_observation
        self.obs_smoothed = jnp.zeros(self.obs_spec)
        # time
        self.t = 0

    def _get_action(self, obs):
        """
        Performs rolling average smoothing on the observation and
        computes a DEP learning step.
        """
        # smoothing
        if self.s4avg > 1 and self.t > 0:
            self.obs_smoothed += (obs - self.obs_smoothed) / self.s4avg
        else:
            self.obs_smoothed = obs

        self.buffer.append([self.obs_smoothed.copy(), None])
        # learning step
        if self.with_learning and len(self.buffer) > (2 + self.time_dist):
            self._learn_controller()
        # new action
        y = self._compute_action()
        self.buffer[-1][1] = y.copy()
        self.t += 1
        return y

    def _q_norm(self, q):
        """
        Normalization function for intermediate action
        obtained by applying the controller matrix to the
        input q = C @ x.
        """
        reg = 10.0 ** (-self.regularization)
        if self.q_norm_selector == "l2":
            q_norm = 1.0 / (jnp.linalg.norm(q, axis=-1) + reg)
        elif self.q_norm_selector == "max":
            q_norm = 1.0 / (max(abs(q), axis=-1) + reg)
        elif self.q_norm_selector == "none":
            q_norm = 1.0
        else:
            raise NotImplementedError(
                "q normalization {self.q_norm_selector} not implemented."
            )

        return q_norm

    def _compute_action(
        self,
    ):
        """
        Compute a DEP action from the current C matrix
        """
        q = jnp.einsum("ijk, ik->ij", self.C_norm, self.obs_smoothed)

        q = jnp.einsum(
            "ij, i->ij",
            q,
            self._q_norm(q),
        )
        y = jnp.maximum(-1, jnp.minimum(1, jnp.tanh(q * self.kappa + self.Cb)))
        y = jnp.einsum("ij, j->ij", y, self.act_scale)
        return y

    def _learn_controller(self):
        """
        Update DEP by one learning step.
        """
        self.C = self._compute_C()
        # linear response in motor space (action -> action)
        R = jnp.einsum("ijk, imk->ijm", self.C, self.M)
        reg = 10.0 ** (-self.regularization)
        # controller normalization c.f. Der et al (2015).
        if self.normalization == "independent":
            factor = self.kappa / (jnp.linalg.norm(R, axis=-1) + reg)
            self.C_norm = jnp.einsum("ijk,ik->ijk", self.C, factor)
        elif self.normalization == "none":
            self.C_norm = self.C
        elif self.normalization == "global":
            norm = jnp.linalg.norm(R)
            self.C_norm = self.C * self.kappa / (norm + reg)
        else:
            raise NotImplementedError(
                f"Controller matrix normalization {self.normalization} not implemented."
            )

        if self.bias_rate >= 0:
            yy = self.buffer[-2][1]
            self.Cb -= (
                jnp.clip(yy * self.bias_rate, -0.05, 0.05) + self.Cb * 0.001
            )
        else:
            self.Cb *= 0

    def _compute_C(self):
        """
        Recompute the controller matrix C from the
        buffer of recent transitions. This is similar
        to the rolling average shown in the publication, but without
        recency weighting.
        """
        C = jnp.zeros_like(self.C)
        for s in range(2, min(self.t - self.time_dist, self.tau)):
            x = self.buffer[-s][0][:, : self.num_sensors]
            xx = self.buffer[-s - 1][0][:, : self.num_sensors]
            xx_t = (
                x
                if self.time_dist == 0
                else self.buffer[-s - self.time_dist][0][:, : self.num_sensors]
            )
            xxx_t = self.buffer[-s - 1 - self.time_dist][0][
                :, : self.num_sensors
            ]

            chi = x - xx
            v = xx_t - xxx_t
            mu = jnp.einsum("ijk, ik->ij", self.M, chi)

            C += jnp.einsum("ij, ik->ijk", mu, v)
        return C
