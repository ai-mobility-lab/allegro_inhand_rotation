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
# Modified by Wonik Robotics (2025)
# Ported from IsaacGym to IsaacLab (2026): Isaac Sim must be launched via
# `AppLauncher` before any `isaaclab`-touching import happens -- including the
# `from hora.tasks import ...` import this script used to do at module scope
# under IsaacGym. `headless`/`sim_device` are read straight out of `sys.argv`
# here (matching Hydra's `key=value` override syntax) since Hydra itself
# hasn't parsed the config yet at this point and the two CLI styles don't mix.
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

import os
import hydra
import datetime
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from hora.algo.ppo.ppo import PPO
from hora.algo.padapt.padapt import ProprioAdapt
from hora.tasks import isaaclab_task_map
from hora.utils.reformat import omegaconf_to_dict, print_dict
from hora.utils.misc import set_np_formatting, set_seed, git_hash, git_diff_config

import wandb

# OmegaConf & Hydra Config
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_envs
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)

# PATH=./configs, works as a config variable within the main function
@hydra.main(config_name="config", config_path="configs")
def main(config: DictConfig):
    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    config.seed = set_seed(config.seed)

    date = str(datetime.datetime.now().strftime("%m%d%H%M"))

    # Initialize wandb if enabled
    if config.wandb.enabled:
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb_config = {}
        for _ in range(2):
            wandb_config.update(dict(config_dict["task"]))
            wandb_config.update(dict(config_dict["train"]))

        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config={**wandb_config},
            sync_tensorboard=True,
        )
        del config_dict
        del wandb_config

        # set run name
        wandb.run.name = (
            ("stage1" if config.train.algo == "PPO" else "stage2") + "_" + config.task_name + "_" + date
        )
        wandb.run.save()
        cprint(f"Wandb initialized: {config.wandb.entity}/{config.wandb.project}", "cyan", attrs=["bold"])


    cprint("Start Building the Environment", "green", attrs=["bold"])
    env = isaaclab_task_map[config.task_name](
        config=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
    )

    output_dif = os.path.join("outputs", config.train.ppo.output_name)
    os.makedirs(output_dif, exist_ok=True)

    agent = eval(config.train.algo)(env, output_dif, full_config=config)

    if config.test:
        agent.restore_test(config.train.load_path)
        agent.test()
    else:
        # date = str(datetime.datetime.now().strftime("%m%d%H"))
        print(git_diff_config("./"))
        os.system(f"git diff HEAD > {output_dif}/gitdiff.patch")
        with open(
            os.path.join(output_dif, f"config_{date}_{git_hash()}.yaml"), "w"
        ) as f:
            f.write(OmegaConf.to_yaml(config))

        # check whether execute train by mistake:
        best_ckpt_path = os.path.join(
            "outputs",
            config.train.ppo.output_name,
            "stage1_nn" if config.train.algo == "PPO" else "stage2_nn",
            "best.pth",
        )
        if os.path.exists(best_ckpt_path):
            user_input = input(
                f"are you intentionally going to overwrite files in {config.train.ppo.output_name}, type yes to continue \n"
            )
            if user_input != "yes":
                exit()

        agent.restore_train(config.train.load_path)
        agent.train()

    # Clean up wandb
    if config.wandb.enabled:
        wandb.finish()
        cprint("Wandb session finished", "cyan", attrs=["bold"])


if __name__ == "__main__":
    main()
    simulation_app.close()
