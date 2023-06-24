# DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 [![PyPI](https://img.shields.io/pypi/v/deprl)](https://pypi.org/project/deprl/)
 [![Downloads](https://pepy.tech/badge/deprl)](https://pepy.tech/project/deprl)
 
 This repo contains the code for the paper [DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems](https://openreview.net/forum?id=C-xa_D3oTj6) paper, published at ICLR 2023 with perfect review scores (8, 8, 8, 10) and a notable-top-25% rating. See [here](https://sites.google.com/view/dep-rl) for videos.

The work was performed by Pierre Schumacher, Daniel F.B. Haeufle, Dieter Büchler, Syn Schmitt and Georg Martius.

If you just want to see the code for DEP, take a look at `deprl/dep_controller.py`, `deprl/custom_agents.py` and `deprl/env_wrapper/wrappers.py`

<p align="center">
<img src=https://user-images.githubusercontent.com/24903880/229783811-44c422e9-3cc3-42e4-b657-d21be9af6458.gif width=250>
<img src=https://user-images.githubusercontent.com/24903880/229783729-d068e87c-cb0b-43c7-91d5-ff2ba836f05b.gif width=214>

 <img src=https://user-images.githubusercontent.com/24903880/229783370-ee95b9c3-06a0-4ef3-9b60-78e88c4eae38.gif width=214>
</p>


### MyoLeg
If you are coming here for the MyoLeg, take a look at this [tutorial](https://github.com/facebookresearch/myosuite/blob/main/docs/source/tutorials.rst#load-dep-rl-baseline). It will show you how to run the pre-trained baseline. We also explain how to train the walking agent in the MyoSuite  [documentation](https://myosuite.readthedocs.io/en/latest/baselines.html#dep-rl-baseline).
<p align="center">
<img src=https://github.com/martius-lab/depRL/assets/24903880/d06200ae-ad35-484c-9d55-83b5235269bc width=350
</p>

## Abstract
Muscle-actuated organisms are capable of learning an unparalleled diversity of
dexterous movements despite their vast amount of muscles. Reinforcement learning (RL) on large musculoskeletal models, however, has not been able to show
similar performance. We conjecture that ineffective exploration in large overactuated action spaces is a key problem. This is supported by our finding that common
exploration noise strategies are inadequate in synthetic examples of overactuated
systems. We identify differential extrinsic plasticity (DEP), a method from the
domain of self-organization, as being able to induce state-space covering exploration within seconds of interaction. By integrating DEP into RL, we achieve fast
learning of reaching and locomotion in musculoskeletal systems, outperforming
current approaches in all considered tasks in sample efficiency and robustness.


## Installation

We provide a python package for easy installation:
```
pip install deprl
```
### CUDA
If you would like to use `jax` with CUDA support, which is recommended for DEP-RL training, we recommend:
```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

 ### CPU only
 As the current version of PyTorch (2.0.1.) defaults to CUDA, on CPU-only machines you might want to run:
```
 pip install torch --index-url https://download.pytorch.org/whl/cpu
```
after installation.

 ### Environments

The humanreacher environment can be installed with
```
pip install git+https://github.com/P-Schumacher/warmup.git
```

OstrichRL can be installed from [here](https://github.com/vittorione94/ostrichrl).


## Experiments

The major experiments (humanreacher reaching and ostrich running) can be repeated with the config files.
Simply run from the root folder:
```
python -m deprl.main experiments/ostrich_running_dep.json
python -m deprl.main experiments/humanreacher.json
```
to train an agent. Model checkpoints will be saved in the `output` directory.
The progress can be monitored with:
```
python -m tonic.plot --path output/folder/
```

To execute a trained policy, use:

```
python -m deprl.play --path output/folder/
```

See the [TonicRL](https://github.com/fabiopardo/tonic) documentation for details.

Be aware that ostrich training can be seed-dependant, as seen in the plots of the original publication.

## Pure DEP
If you want to see pure DEP in action, just run the following bash files after installing the ostrichrl and warmup environments.
```
bash play_files/play_dep_humanreacher.sh
bash play_files/play_dep_ostrich.sh
bash play_files/play_dep_dmcontrol_quadruped.sh
```

You might see a more interesting ostrich behavior by disabling episode resets in the ostrich environment first.
## Environments

The ostrich environment can be found [here](https://github.com/vittorione94/ostrichrl) and is installed automatically via poetry.

The arm-environment [warmup](https://github.com/P-Schumacher/warmup) is also automatically installed by poetry and can be used like any other gym environment:

```
import gym
import warmup

env = gym.make("humanreacher-v0")

for ep in range(5):
     ep_steps = 0
     state = env.reset()
     while True:
         next_state, reward, done, info = env.step(env.action_space.sample())
         env.render()
         if done or (ep_steps >= env.max_episode_steps):
             break
         ep_steps += 1

```

The humanoid environments were simulated with [SCONE](https://scone.software/doku.php?id=start). A ready-to-use RL package will be released in cooperation with GOATSTREAM at a later date.

## Source Code Installation
We recommend an installation with [poetry](https://python-poetry.org/) to ensure reproducibility.
While [TonicRL](https://github.com/fabiopardo/tonic) with PyTorch is used for the RL algorithms, DEP itself is implemented in `jax`. We *strongly* recommend GPU-usage to speed up the computation of DEP. On systems without GPUs, give the tensorflow version of TonicRL a try! We also provide a requirements file for pip. Please check the instructions for GPU and CPU versions of `torch` and `jax` above.

### Pip
Just clone the repository and install locally:

```
git clone https://github.com/martius-lab/depRL.git
cd depRL
pip install -r requirements.txt
pip install -e ./
```

### Poetry

1. Make sure to install poetry and deactivate all virtual environments.
2. Clone the environment
```
git clone https://github.com/martius-lab/depRL
```

3. Go to the root folder and run
```
poetry install
poetry shell
```

That's it!

The build has been tested with:
```
Ubuntu 20.04 and Ubuntu 22.04
CUDA 12.0
poetry 1.4.0
```
 
### Troubleshooting
* A common error with poetry is a faulty interaction with the python keyring, resulting in a `Failed to unlock the collection!`-error. It could also happen that the dependency solving takes very long (more than 60s), this is caused by the same error.
  If it happens, try to append
```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```
to your bashrc. You can also try to prepend it to the poetry command: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install`.
* If you have an error related to your `ptxas` version, this means that your cuda environment is not setup correctly and you should install the cuda-toolkit. The easiest way is to do this via conda if you don't have admin rights on your workstation.
  I recommend running
```
conda install -c conda-forge cudatoolkit-dev
```
* In any other case, first try to delete the `poetry.lock` file and the virtual env `.venv`, then run `poetry install` again.


Feel free to open an issue if you encounter any problems.

## Citation

If you find this repository useful, please consider giving a star ⭐ and cite our [paper](https://openreview.net/forum?id=C-xa_D3oTj6)  by using the following BibTeX entrys.

```
@inproceedings{schumacher2023:deprl,
  title = {DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems},
  author = {Schumacher, Pierre and Haeufle, Daniel F.B. and B{\"u}chler, Dieter and Schmitt, Syn and Martius, Georg},
  booktitle = {Proceedings of the Eleventh International Conference on Learning Representations (ICLR)},
  month = may,
  year = {2023},
  doi = {},
  url = {https://openreview.net/forum?id=C-xa_D3oTj6},
  month_numeric = {5}
}
```
