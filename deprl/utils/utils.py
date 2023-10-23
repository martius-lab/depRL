import json
import os
import sys
from contextlib import contextmanager



def prepare_params():
    f = open(sys.argv[-1], "r")
    config = json.load(f)
    return config


def mujoco_render(env, *args, **kwargs):
    if "mujoco_py" in str(type(env.sim)):
        env.render(*args, **kwargs)
    else:
        env.sim.renderer.render_to_window(*args, **kwargs)


@contextmanager
def stdout_suppression():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
