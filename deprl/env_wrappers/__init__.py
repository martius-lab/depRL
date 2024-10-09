import deprl  # noqa
from deprl.env_wrappers.dm_wrapper import DMWrapper, OstrichDMWrapper
from deprl.env_wrappers.gym_wrapper import GymWrapper
from deprl.env_wrappers.scone_wrapper import SconeWrapper


def apply_wrapper(env):
    if "control" in str(env).lower():
        if env.unwrapped.name == "ostrich-run":
            return OstrichDMWrapper(env)
        return DMWrapper(env)
    elif "scone" in str(env).lower():
        return SconeWrapper(env)
    else:
        return GymWrapper(env)


def env_tonic_compat(env, id=5, parallel=1, sequential=1):
    """
    Applies wrapper for tonic and passes random seed.
    """
    return apply_wrapper(eval(env))


__all__ = [env_tonic_compat, apply_wrapper]
