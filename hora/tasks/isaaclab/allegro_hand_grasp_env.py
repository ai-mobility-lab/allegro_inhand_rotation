# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Modified by Wonik Robotics (2025)
# IsaacLab port of hora/tasks/allegro_hand_grasp.py. Generates the grasp-pose
# caches under `cache/*.npy` consumed by `AllegroHandHoraEnv._reset_idx`.
#
# The original's fingertip-contact check used `isaacgym.gymapi`'s CPU-only
# `get_env_rigid_contacts`, which is also why the original asserted
# `self.device == "cpu"`. Here it's replaced by an IsaacLab `ContactSensor`
# filtered against the object prim, which works on GPU -- so that assertion
# is dropped.
# --------------------------------------------------------

import os

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import sample_uniform

from .allegro_hand_hora_env import REPO_ROOT, AllegroHandHoraEnv, tensor_clamp

# link names of the four fingertips in assets/allegro/allegro_hora.urdf (and the
# left/right variants) -- confirmed against the URDF, not the deploy-side ROS2 mapping.
# Names use underscores, not dots (`link_3_0_tip`, not `link_3.0_tip`): Isaac Sim's
# URDF importer (isaacsim.asset.importer.urdf-2.4.31) mangles dotted link/joint names
# into a broken internal SdfPath when building the `/visuals/...` sub-prims, raising
# "Used null prim" -- see the URDF files themselves, which were renamed to side-step it.
# Overridable via `env.asset.fingertipLinkNames` in the task yaml -- the DIGIT hand
# variants (assets/allegro/allegro_digit_*.urdf) attach an extra fixed `*_tip_elastomer`
# link past each `*_tip` housing, and the elastomer pad is the surface that actually
# contacts the object, not the rigid housing underneath it.
FINGERTIP_LINK_NAMES = ["link_3_0_tip", "link_7_0_tip", "link_11_0_tip", "link_15_0_tip"]

## ORIGINAL (IsaacGym implementation) ##
# Allegro Hand DOF names: ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0',
# 'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0',
# 'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0',
# 'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0']
#
# FINGER_ORDER = {
#     "index": [0, 1, 2, 3],
#     "thumb": [4, 5, 6, 7],
#     "middle": [8, 9, 10, 11],
#     "ring": [12, 13, 14, 15]
# }
## MODIFIED (IsaacLab implementation) ##
# ['joint_0_0', 'joint_12_0', 'joint_4_0', 'joint_8_0', 
#  'joint_1_0', 'joint_13_0', 'joint_5_0', 'joint_9_0', 
#  'joint_2_0', 'joint_14_0', 'joint_6_0', 'joint_10_0', 
#  'joint_3_0', 'joint_15_0', 'joint_7_0', 'joint_11_0']

# CANONICAL_POSE = [
#     0.082, 1.244, 0.265, 0.298,
#     1.104, 1.163, 0.953, -0.138,
#     0.005, 1.096, 0.080, 0.150,
#     0.029, 1.337, 0.285, 0.317,
# ]
CANONICAL_POSE = [
    0.082, 1.104, 0.005, 0.029,
    1.244, 1.163, 1.096, 1.337,
    0.265, 0.953, 0.080, 0.285,
    0.298, -0.138, 0.150, 0.317,
]


class AllegroHandGraspEnv(AllegroHandHoraEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self.fingertip_link_names = cfg.hora_cfg["asset"].get("fingertipLinkNames", FINGERTIP_LINK_NAMES)
        super().__init__(cfg, render_mode, **kwargs)
        self.saved_grasping_states = torch.zeros((0, 23), dtype=torch.float, device=self.device)
        self.canonical_pose = torch.tensor(CANONICAL_POSE, dtype=torch.float, device=self.device)
        self.max_cache_size = 50000
        self.fingertip_body_ids = [self.hand.body_names.index(n) for n in self.fingertip_link_names]
        self.reset_buffer_steps = 10

    def _setup_scene(self):
        super()._setup_scene()
        # IsaacLab's ContactSensor only supports one-to-many filtering (one sensor body
        # against many filter bodies) -- a single sensor covering all 4 fingertip bodies
        # filtered against the (one) Object silently returns all-zero force_matrix_w once
        # numEnvs > 1, because the sensor-body-major / filter-body-major indexing
        # misaligns across envs. So one sensor per fingertip, each filtered against Object.
        self.fingertip_contact_sensors = []
        for i, link_name in enumerate(self.fingertip_link_names):
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                    filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                    update_period=0.0,
                )
            )
            self.scene.sensors[f"fingertip_contact_{i}"] = sensor
            self.fingertip_contact_sensors.append(sensor)

    def _get_rewards(self) -> torch.Tensor:
        # grasp generation only cares about termination; no learning signal needed.
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        fingertip_pos = self.hand.data.body_pos_w[:, self.fingertip_body_ids]
        obj_pos = self.object.data.root_pos_w.unsqueeze(1)
        near_object = (torch.norm(fingertip_pos - obj_pos, dim=-1) < 0.1).all(dim=-1)

        # each sensor's force_matrix_w: (num_envs, 1 sensor body, 1 filtered body, 3)
        contact_forces = torch.stack(
            [s.data.force_matrix_w[:, 0, 0] for s in self.fingertip_contact_sensors], dim=1
        )  # (num_envs, num_fingertips, 3)
        num_fingertips_in_contact = (torch.norm(contact_forces, dim=-1) > 0.1).sum(dim=-1)
        gripping = num_fingertips_in_contact >= 2

        above_floor = self.object_pos[:, -1] > self.reset_z_threshold
        
        reset_buffer_mask = self.episode_length_buf < self.reset_buffer_steps
        
        good_grasp = ((~reset_buffer_mask) & near_object & gripping & above_floor) | (reset_buffer_mask & above_floor)
        # good_grasp = near_object & gripping & above_floor
        
        # print(torch.norm(fingertip_pos - obj_pos, dim=-1))
        # print(torch.norm(contact_forces, dim=-1))
        # print(near_object, gripping, above_floor, num_fingertips_in_contact)
        # print("=======================================")
            
        terminated = ~good_grasp
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # a full episode length reached (rather than an early failure) means this was a
        # stable grasp worth caching -- must be read before the base class's reset logic
        # runs and clears `reset_time_outs`.
        success_mask = self.reset_time_outs[env_ids].clone()

        if self.randomize_mass:
            new_mass = sample_uniform(
                self.randomize_mass_lower, self.randomize_mass_upper, (len(env_ids), 1), device="cpu"
            )
            masses = self.object.root_physx_view.get_masses()
            masses[env_ids.cpu()] = new_mass
            self.object.root_physx_view.set_masses(masses, env_ids.cpu())
        self._update_priv_buf(env_ids, "obj_mass", self.object.root_physx_view.get_masses()[env_ids.cpu(), 0].to(self.device), lower=0, upper=0.2)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = sample_uniform(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions), device=self.device
            )
            self.d_gain[env_ids] = sample_uniform(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions), device=self.device
            )
        self.rb_forces[env_ids] = 0.0

        # cache the (dof_pos, object_pose) pairs of envs whose grasp survived the full
        # episode, exactly as `AllegroHandGrasp.reset_idx` does.
        all_states = torch.cat(
            [self.hand.data.joint_pos, torch.cat([self.object_pos, self.object.data.root_quat_w], dim=-1)], dim=1
        )
        self.saved_grasping_states = torch.cat([self.saved_grasping_states, all_states[env_ids][success_mask]])
        print("current cache size:", self.saved_grasping_states.shape[0])
        if self.saved_grasping_states.shape[0] >= self.max_cache_size:
            name = os.path.join(
                REPO_ROOT, "cache", f"{self.grasp_cache_name}_grasp_50k_s{str(self.base_obj_scale).replace('.', '')}.npy"
            )
            np.save(name, self.saved_grasping_states[: self.max_cache_size].cpu().numpy())
            raise SystemExit(f"saved {self.max_cache_size} grasp poses to {name}")

        # skip `AllegroHandHoraEnv._reset_idx` (grasp-cache pose lookup -- not applicable
        # here) and go straight to `DirectRLEnv`'s base bookkeeping.
        DirectRLEnv._reset_idx(self, env_ids)

        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)

        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        pos = self.canonical_pose.unsqueeze(0) + 0.25 * rand_floats
        pos = tensor_clamp(pos, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)

        self.prev_targets[env_ids, : self.num_hand_dofs] = pos
        self.cur_targets[env_ids, : self.num_hand_dofs] = pos
        if not self.torque_control:
            self.hand.set_joint_position_target(pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=env_ids)

        self.obs_buf_lag_history[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
        self._compute_intermediate_values()
