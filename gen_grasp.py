# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------
# Ported from IsaacGym to IsaacLab (2026) -- see train.py for why the
# `AppLauncher` bootstrap has to run before any `isaaclab`-touching import.
# --------------------------------------------------------

import sys

from isaaclab.app import AppLauncher


def _cli_override(key: str, default: str) -> str:
    prefix = f"{key}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


_headless = _cli_override("headless", "False").lower() == "true"
_sim_device = _cli_override("sim_device", "cuda:0")

app_launcher = AppLauncher(headless=_headless, device=_sim_device)
simulation_app = app_launcher.app

# ---- everything below may safely import isaaclab ----

import hydra
from omegaconf import DictConfig, OmegaConf

from hora.tasks import isaaclab_task_map
from hora.utils.misc import set_np_formatting, set_seed


# OmegaConf & Hydra Config
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_envs
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)


@hydra.main(config_name="config", config_path="configs")
def main(config: DictConfig):
    set_np_formatting()
    config.seed = set_seed(config.seed)

    env = isaaclab_task_map[config.task_name](
        config=config.task,
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
    )

    env.reset()
    while True:
        actions = env.zero_actions()
        _ = env.step(actions)


if __name__ == "__main__":
    main()
    simulation_app.close()
