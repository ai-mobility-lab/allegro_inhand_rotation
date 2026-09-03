# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Modified by Wonik Robotics (2025)
# Adapts an IsaacLab `DirectRLEnv` to the legacy `hora/tasks/base/vec_task.py`
# `VecTask` interface (`reset()`/`step()` returning a
# `{"obs", "priv_info", "proprio_hist"}` dict, plain `gym.spaces.Box` action/
# observation spaces, `zero_actions()`, `num_envs`, `prop_hist_len`) so that
# `hora/algo/ppo/ppo.py` and `hora/algo/padapt/padapt.py` need no changes at all.
# --------------------------------------------------------

import numpy as np
import torch
from gym import spaces


class HoraDirectEnvWrapper:
    def __init__(self, direct_env, config):
        self.env = direct_env
        self.rl_device = config.get("rl_device", str(direct_env.device))
        self.clip_obs = config["env"].get("clipObservations", np.inf)
        self.clip_actions = config["env"].get("clipActions", np.inf)
        # populated by the task env (`hora.propHistoryLen`), read by ProprioAdapt
        self.prop_hist_len = direct_env.prop_hist_len

        num_obs = config["env"]["numObservations"]
        num_actions = config["env"]["numActions"]
        self.obs_space = spaces.Box(
            np.ones(num_obs, dtype=np.float32) * -np.inf,
            np.ones(num_obs, dtype=np.float32) * np.inf,
        )
        self.act_space = spaces.Box(
            np.ones(num_actions, dtype=np.float32) * -1.0,
            np.ones(num_actions, dtype=np.float32) * 1.0,
        )
        self.obs_dict = {}

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def num_actions(self):
        return self.act_space.shape[0]

    def zero_actions(self):
        return torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.rl_device)

    def _pack_obs(self, obs_dict):
        return {
            "obs": torch.clamp(obs_dict["obs"], -self.clip_obs, self.clip_obs).to(self.rl_device),
            "priv_info": obs_dict["priv_info"].to(self.rl_device),
            "proprio_hist": obs_dict["proprio_hist"].to(self.rl_device),
        }

    def reset(self):
        obs_dict, _ = self.env.reset()
        self.obs_dict = self._pack_obs(obs_dict)
        return self.obs_dict

    def step(self, actions):
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, reward, terminated, truncated, extras = self.env.step(action_tensor)
        self.obs_dict = self._pack_obs(obs_dict)
        dones = (terminated | truncated).to(self.rl_device)
        extras = dict(extras)
        # matches the original VecTask's `timeout_buf`, used for PPO value bootstrapping
        extras["time_outs"] = truncated.to(self.rl_device)
        return self.obs_dict, reward.to(self.rl_device), dones, extras
