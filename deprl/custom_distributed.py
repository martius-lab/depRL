"""Builders for distributed training."""
import multiprocessing

import numpy as np

from deprl.utils import stdout_suppression


def proc(
    action_pipe,
    output_queue,
    group_seed,
    build_dict,
    max_episode_steps,
    index,
    workers,
    env_args,
    header,
):
    """Process holding a sequential group of environments."""
    envs = Sequential(build_dict, max_episode_steps, workers, env_args, header)
    envs.initialize(group_seed)

    observations = envs.start()
    output_queue.put((index, observations))

    while True:
        actions = action_pipe.recv()
        out = envs.step(actions)
        output_queue.put((index, out))


class Sequential:
    """A group of environments used in sequence."""

    def __init__(
        self, build_dict, max_episode_steps, workers, env_args, header
    ):
        if header is not None:
            with stdout_suppression():
                exec(header)
        if hasattr(build_env_from_dict(build_dict).unwrapped, "environment"):
            # its a deepmind env
            self.environments = [
                build_env_from_dict(build_dict)() for i in range(workers)
            ]
        else:
            # its a gym env
            self.environments = [
                build_env_from_dict(build_dict) for i in range(workers)
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
        # group seed is given, the others are determined from it
        for i, environment in enumerate(self.environments):
            environment.seed(seed + i)

    def start(self):
        """Used once to get the initial observations."""
        observations = [env.reset() for env in self.environments]
        muscle_states = [env.muscle_states for env in self.environments]
        self.lengths = np.zeros(len(self.environments), int)
        return np.array(observations, np.float32), np.array(
            muscle_states, np.float32
        )

    def step(self, actions):
        next_observations = []  # Observations for the transitions.
        rewards = []
        resets = []
        terminations = []
        observations = []  # Observations for the actions selection.
        muscle_states = []

        for i in range(len(self.environments)):
            ob, rew, term, env_info = self.environments[i].step(actions[i])
            muscle = self.environments[i].muscle_states
            self.lengths[i] += 1
            # Timeouts trigger resets but are not true terminations.
            reset = term or self.lengths[i] == self._max_episode_steps
            next_observations.append(ob)
            rewards.append(rew)
            resets.append(reset)

            terminations.append(term)

            if reset:
                ob = self.environments[i].reset()
                muscle = self.environments[i].muscle_states
                self.lengths[i] = 0

            observations.append(ob)
            muscle_states.append(muscle)

        observations = np.array(observations, np.float32)
        muscle_states = np.array(muscle_states, np.float32)
        infos = dict(
            observations=np.array(next_observations, np.float32),
            rewards=np.array(rewards, np.float32),
            resets=np.array(resets, bool),
            terminations=np.array(terminations, bool),
        )
        return observations, muscle_states, infos

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
        env_args,
        header,
    ):
        self.build_dict = build_dict
        self.worker_groups = worker_groups
        self.workers_per_group = workers_per_group
        self._max_episode_steps = max_episode_steps
        self.env_args = env_args
        self.header = header

    def initialize(self, seed):
        dummy_environment = build_env_from_dict(self.build_dict)
        dummy_environment.merge_args(self.env_args)
        dummy_environment.apply_args()

        self.observation_space = dummy_environment.observation_space
        self.action_space = dummy_environment.action_space
        del dummy_environment
        self.started = False
        # this prevents issues with GH actions and multiple start method inits
        # spawn works across all operating systems
        context = multiprocessing.get_context("spawn")
        self.output_queue = context.Queue()
        self.action_pipes = []
        self.processes = []

        for i in range(self.worker_groups):
            pipe, worker_end = context.Pipe()
            self.action_pipes.append(pipe)
            group_seed = (
                seed * self.workers_per_group + i * self.workers_per_group
            )

            # required for spawnstart_method for macos and windows
            proc_kwargs = {
                "action_pipe": worker_end,
                "output_queue": self.output_queue,
                "group_seed": group_seed,
                "build_dict": self.build_dict,
                "max_episode_steps": self._max_episode_steps,
                "index": i,
                "workers": self.workers_per_group,
                "env_args": self.env_args
                if hasattr(self, "env_args")
                else None,
                "header": self.header,
            }

            self.processes.append(
                context.Process(target=proc, kwargs=proc_kwargs)
            )
            self.processes[-1].daemon = True
            self.processes[-1].start()

    def start(self):
        """Used once to get the initial observations."""
        assert not self.started
        self.started = True
        observations_list = [None for _ in range(self.worker_groups)]
        muscle_states_list = [None for _ in range(self.worker_groups)]

        for _ in range(self.worker_groups):
            index, (observations, muscle_states) = self.output_queue.get()
            observations_list[index] = observations
            muscle_states_list[index] = muscle_states

        self.observations_list = np.array(observations_list)
        self.muscle_states_list = np.array(muscle_states_list)
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
            self.muscle_states_list
        )

    def step(self, actions):
        actions_list = np.split(actions, self.worker_groups)
        for actions, pipe in zip(actions_list, self.action_pipes):
            pipe.send(actions)

        for _ in range(self.worker_groups):
            index, (
                observations,
                muscle_states,
                infos,
            ) = self.output_queue.get()
            self.observations_list[index] = observations
            self.next_observations_list[index] = infos["observations"]
            self.rewards_list[index] = infos["rewards"]
            self.resets_list[index] = infos["resets"]
            self.terminations_list[index] = infos["terminations"]
            self.muscle_states_list[index] = muscle_states

        observations = np.concatenate(self.observations_list)
        muscle_states = np.concatenate(self.muscle_states_list)
        infos = dict(
            observations=np.concatenate(self.next_observations_list),
            rewards=np.concatenate(self.rewards_list),
            resets=np.concatenate(self.resets_list),
            terminations=np.concatenate(self.terminations_list),
        )
        return observations, muscle_states, infos

    def close(self):
        self.proc.terminate()


def distribute(
    environment,
    tonic_conf,
    env_args,
    parallel=None,
    sequential=None,
):
    """Distributes workers over parallel and sequential groups."""
    parallel = tonic_conf["parallel"] if parallel is None else parallel
    sequential = tonic_conf["sequential"] if sequential is None else sequential
    build_dict = dict(
        env=environment, parallel=parallel, sequential=sequential
    )

    dummy_environment = build_env_from_dict(build_dict)
    max_episode_steps = dummy_environment._max_episode_steps
    del dummy_environment

    if parallel < 2:
        return Sequential(
            build_dict=build_dict,
            max_episode_steps=max_episode_steps,
            workers=sequential,
            env_args=env_args,
            header=tonic_conf["header"],
        )
    return Parallel(
        build_dict,
        worker_groups=parallel,
        workers_per_group=sequential,
        max_episode_steps=max_episode_steps,
        env_args=env_args,
        header=tonic_conf["header"],
    )


def build_env_from_dict(build_dict):
    assert build_dict["env"] is not None
    if type(build_dict) == dict:
        from deprl import env_tonic_compat

        return env_tonic_compat(**build_dict)
    else:
        return build_dict()
