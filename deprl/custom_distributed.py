"""Builders for distributed training."""
import multiprocessing

import numpy as np


class Sequential:
    """A group of environments used in sequence."""

    def __init__(
        self, build_dict, max_episode_steps, workers, index=0, env_args=None
    ):
        if hasattr(build_env_from_dict(build_dict)().unwrapped, "environment"):
            # its a deepmind env
            self.environments = [
                build_env_from_dict(build_dict)() for i in range(workers)
            ]
        else:
            # its a gym env
            self.environments = [
                build_env_from_dict(build_dict)(identifier=index * workers + i)
                for i in range(workers)
            ]
        if env_args is not None:
            [x.merge_args(env_args) for x in self.environments]
            [x.apply_args() for x in self.environments]
        self._max_episode_steps = max_episode_steps
        self.observation_space = self.environments[0].observation_space
        self.action_space = self.environments[0].action_space
        self.name = self.environments[0].name
        self.num_workers = workers

    def initialize(self, seed):
        for i, environment in enumerate(self.environments):
            environment.seed(seed + i)

    def start(self):
        """Used once to get the initial observations."""
        observations = [env.reset() for env in self.environments]
        tendon_states = [env.tendon_states for env in self.environments]
        self.lengths = np.zeros(len(self.environments), int)
        return np.array(observations, np.float32), np.array(
            tendon_states, np.float32
        )

    def step(self, actions):
        next_observations = []  # Observations for the transitions.
        rewards = []
        resets = []
        terminations = []
        observations = []  # Observations for the actions selection.
        tendon_states = []

        for i in range(len(self.environments)):
            ob, rew, term, env_info = self.environments[i].step(actions[i])
            muscle = self.environments[i].tendon_states
            self.lengths[i] += 1
            # Timeouts trigger resets but are not true terminations.
            reset = term or self.lengths[i] == self._max_episode_steps
            next_observations.append(ob)
            rewards.append(rew)
            resets.append(reset)

            terminations.append(term)

            if reset:
                ob = self.environments[i].reset()
                muscle = self.environments[i].tendon_states
                self.lengths[i] = 0

            observations.append(ob)
            tendon_states.append(muscle)

        observations = np.array(observations, np.float32)
        tendon_states = np.array(tendon_states, np.float32)
        infos = dict(
            observations=np.array(next_observations, np.float32),
            rewards=np.array(rewards, np.float32),
            resets=np.array(resets, bool),
            terminations=np.array(terminations, bool),
        )
        return observations, tendon_states, infos

    def render(self, mode="human", *args, **kwargs):
        outs = []
        for env in self.environments:
            out = env.render(mode=mode, *args, **kwargs)
            outs.append(out)
        if mode != "human":
            return np.array(outs)

    def render_substep(self):
        for env in self.environments:
            env.render_substep()


class Parallel:
    """A group of sequential environments used in parallel."""

    def __init__(
        self,
        build_dict,
        worker_groups,
        workers_per_group,
        max_episode_steps,
        env_args=None,
    ):
        self.build_dict = build_dict
        self.worker_groups = worker_groups
        self.workers_per_group = workers_per_group
        self._max_episode_steps = max_episode_steps
        self.env_args = env_args

    def initialize(self, seed):
        def proc(action_pipe, index, seed):
            """Process holding a sequential group of environments."""
            envs = Sequential(
                self.build_dict,
                self._max_episode_steps,
                self.workers_per_group,
                index,
                self.env_args,
            )
            envs.initialize(seed)

            observations = envs.start()
            self.output_queue.put((index, observations))

            while True:
                actions = action_pipe.recv()
                out = envs.step(actions)
                self.output_queue.put((index, out))

        dummy_environment = build_env_from_dict(self.build_dict)()
        dummy_environment.merge_args(self.env_args)
        dummy_environment.apply_args()

        self.observation_space = dummy_environment.observation_space
        self.action_space = dummy_environment.action_space
        del dummy_environment
        self.started = False

        # fork is default on linux, but not mac
        multiprocessing.set_start_method("fork")
        self.output_queue = multiprocessing.Queue()
        self.action_pipes = []

        for i in range(self.worker_groups):
            pipe, worker_end = multiprocessing.Pipe()
            self.action_pipes.append(pipe)
            group_seed = (
                seed * (self.worker_groups * self.workers_per_group)
                + i * self.workers_per_group
            )
            process = multiprocessing.Process(
                target=proc, args=(worker_end, i, group_seed)
            )
            process.daemon = True
            process.start()

    def start(self):
        """Used once to get the initial observations."""
        assert not self.started
        self.started = True
        observations_list = [None for _ in range(self.worker_groups)]
        tendon_states_list = [None for _ in range(self.worker_groups)]

        for _ in range(self.worker_groups):
            index, (observations, tendon_states) = self.output_queue.get()
            observations_list[index] = observations
            tendon_states_list[index] = tendon_states

        self.observations_list = np.array(observations_list)
        self.tendon_states_list = np.array(tendon_states_list)
        self.next_observations_list = np.zeros_like(self.observations_list)
        self.rewards_list = np.zeros(
            (self.worker_groups, self.workers_per_group), np.float32
        )
        self.resets_list = np.zeros(
            (self.worker_groups, self.workers_per_group), bool
        )
        self.terminations_list = np.zeros(
            (self.worker_groups, self.workers_per_group), bool
        )
        self.terminations_list = np.zeros(
            (self.worker_groups, self.workers_per_group), bool
        )

        return np.concatenate(self.observations_list), np.concatenate(
            self.tendon_states_list
        )

    def step(self, actions):
        actions_list = np.split(actions, self.worker_groups)
        for actions, pipe in zip(actions_list, self.action_pipes):
            pipe.send(actions)

        for _ in range(self.worker_groups):
            index, (
                observations,
                tendon_state,
                infos,
            ) = self.output_queue.get()
            self.observations_list[index] = observations
            self.next_observations_list[index] = infos["observations"]
            self.rewards_list[index] = infos["rewards"]
            self.resets_list[index] = infos["resets"]
            self.terminations_list[index] = infos["terminations"]
            self.tendon_states_list[index] = tendon_state

        observations = np.concatenate(self.observations_list)
        tendon_states = np.concatenate(self.tendon_states_list)
        infos = dict(
            observations=np.concatenate(self.next_observations_list),
            rewards=np.concatenate(self.rewards_list),
            resets=np.concatenate(self.resets_list),
            terminations=np.concatenate(self.terminations_list),
        )
        return observations, tendon_states, infos


def distribute(
    build_dict, worker_groups=1, workers_per_group=1, env_args=None
):
    """Distributes workers over parallel and sequential groups."""

    dummy_environment = build_env_from_dict(build_dict)()
    max_episode_steps = dummy_environment._max_episode_steps
    del dummy_environment

    if worker_groups < 2:
        return Sequential(
            build_dict,
            max_episode_steps=max_episode_steps,
            workers=workers_per_group,
            env_args=env_args,
        )
    return Parallel(
        build_dict,
        worker_groups=worker_groups,
        workers_per_group=workers_per_group,
        max_episode_steps=max_episode_steps,
        env_args=env_args,
    )


def build_env_from_dict(build_dict):
    if type(build_dict) == dict:
        from deprl import env_tonic_compat

        return env_tonic_compat(**build_dict)
    else:
        return lambda identifier=0: build_dict()
