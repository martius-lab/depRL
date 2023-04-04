import json
import os
import sys
from types import SimpleNamespace


def prepare_files(orig_params):
    params = get_params(orig_params)
    os.makedirs(params.working_dir, exist_ok=True)
    return params


def get_params(orig_params):
    params = orig_params.copy()
    for key, val in params.items():
        if type(params[key]) == dict:
            params[key] = SimpleNamespace(**val)
    params = SimpleNamespace(**params)
    return params


def prepare_params():
    f = open(sys.argv[-1], "r")
    orig_params = json.load(f)
    params = prepare_files(orig_params)
    return orig_params, params
