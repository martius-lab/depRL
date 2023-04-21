import numpy as np

from .dep_controller import DEP
from .vendor.tonic import logger


def dep_factory(mix, instance):
    """
    Class factory returning a class that switches between DEP and the RL agent
    based on one of several heuristics.
    0: UnmixedAgent -> Just RL Agent
    1: InitExploreDEP -> Just DEP exploration to initially fill the buffer.
    2: DetSwitchDEP -> Deterministic switching between DEP and RL agent.
    3: StochSwitchDEP -> Stochastic switching between DEP and RL agent.
    """

    class UnmixedAgent(instance.__class__):
        def step(
            self, observations, steps, tendon_states=None, greedy_episode=None
        ):
            return super().step(observations, steps)

        def test_step(self, observations, steps, tendon_states=None):
            return super().test_step(observations, steps)

        def reset(self):
            pass

    class InitExploreDep(instance.__class__):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.expl = DEP()

        def initialize(self, observation_space, action_space, seed=None):
            super().initialize(observation_space, action_space, seed)
            self.expl.initialize(observation_space, action_space, seed)

        def step(
            self, observations, steps, tendon_states=None, greedy_episode=None
        ):
            if steps > (self.replay.steps_before_batches / 1):
                return super().step(observations, steps)
            actions = self.dep_step(tendon_states, steps)
            self.last_observations = observations.copy()
            self.last_actions = actions.copy()

            return actions

        def test_step(
            self, observations, steps, tendon_states=None, greedy_episode=None
        ):
            return super().test_step(observations, steps)

        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)

        def reset(self):
            pass

        def dep_step(self, tendon_states, steps):
            return self.expl.step(tendon_states, steps)

    class DetSwitchDep(InitExploreDep):
        def __init__(self, *args, **kwargs):
            self.switch = 0
            self.since_switch = 1
            return super().__init__(*args, **kwargs)

        def step(
            self, observations, steps, tendon_states=None, greedy_episode=None
        ):
            if steps > (self.replay.steps_before_batches / 1):
                if (
                    self.switch
                    and not self.since_switch % self.expl.intervention_length
                ):
                    self.switch = 0
                    self.since_switch = 1
                if (
                    not self.switch
                    and not self.since_switch % self.expl.rl_length
                ):
                    self.switch = 1
                    self.since_switch = 1
                self.since_switch += 1
                if not self.switch:
                    self.dep_step(tendon_states, steps)
                    return super().step(observations, steps)
            actions = self.dep_step(tendon_states, steps)

            self.last_observations = observations.copy()
            self.last_actions = actions.copy()
            return actions

        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)

    class StochSwitchDep(DetSwitchDep):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # want policy to start first
            self.since_switch = 500

        def step(
            self, observations, steps, tendon_states=None, greedy_episode=False
        ):
            if steps > (self.replay.steps_before_batches / 1):
                if greedy_episode:
                    return super(DetSwitchDep, self).step(
                        observations, steps, tendon_states
                    )
                if self.since_switch > self.expl.intervention_length:
                    # important for Dep to keep learning
                    self.dep_step(tendon_states, steps)
                    if np.random.uniform() < self.expl.intervention_proba:
                        self.since_switch = 0
                    self.since_switch += 1
                    return super(DetSwitchDep, self).step(
                        observations, steps, tendon_states
                    )
            actions = self.dep_step(tendon_states, steps)
            self.last_observations = observations.copy()
            self.last_actions = actions.copy()
            self.since_switch += 1
            return actions

    class PureDep(InitExploreDep):
        def step(
            self, observations, steps, tendon_states=None, greedy_episode=False
        ):
            if np.any(np.isnan(tendon_states)):
                print("tendon nan!")
            return self.dep_step(tendon_states, steps)

        def update(self, *args, **kwargs):
            pass

        def test_update(self, *args, **kwargs):
            pass

        def test_step(
            self, observations, steps, tendon_states=None, greedy_episode=False
        ):
            # return self.dep_step(tendon_states, steps)[0, :]
            return self.dep_step(tendon_states, steps)

    if mix == 1:
        logger.log("Initial exploration DEP")
        return InitExploreDep
    elif mix == 2:
        logger.log("Deterministic Switch-DEP.")
        return DetSwitchDep
    elif mix == 3:
        logger.log("Stochastic Switch-DEP. Paper version.")
        return StochSwitchDep
    elif mix == 4:
        logger.log("Pure DEP.")
        return PureDep
    elif mix == 0:
        logger.log("No DEP")
        return UnmixedAgent
    else:
        raise Exception("Invalid agent specified")
