"""DIGIT rotation with privileged elastomer/object normal contact forces."""

import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg

from .allegro_hand_hora_env import AllegroHandHoraEnv


class AllegroHandContactHoraEnv(AllegroHandHoraEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        env_cfg = cfg.hora_cfg
        self.fingertip_link_names = env_cfg["asset"]["fingertipLinkNames"]
        if len(self.fingertip_link_names) != 4 or not all(
            name.endswith("_elastomer") for name in self.fingertip_link_names
        ):
            raise ValueError("Contact Hora requires four fingertip elastomer links")
        if env_cfg["hora"]["privInfoDim"] != 21:
            raise ValueError("Contact Hora requires 9 object + 12 contact privileged observations")
        self.contact_force_threshold = float(env_cfg["contact"]["forceThreshold"])
        self.contact_reward_scale = float(env_cfg["reward"]["fingertipContactRewardScale"])
        if self.contact_force_threshold < 0 or self.contact_reward_scale < 0:
            raise ValueError("Contact threshold and reward scale must be non-negative")
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()
        # Filtering supports one sensor body to many targets: use one per pad.
        self.fingertip_contact_sensors = []
        for i, link_name in enumerate(self.fingertip_link_names):
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                update_period=0.0,
            ))
            self.scene.sensors[f"fingertip_contact_{i}"] = sensor
            self.fingertip_contact_sensors.append(sensor)

    def _fingertip_object_forces(self):
        """(env, fingertip, xyz) normal force vectors in world frame, in N.

        Object filtering excludes contacts with the hand and ground. IsaacLab's
        force_matrix_w reports normal forces, excluding tangential friction.
        """
        return torch.stack([
            sensor.data.force_matrix_w[:, 0].sum(dim=1)
            for sensor in self.fingertip_contact_sensors
        ], dim=1)

    def _get_observations(self):
        obs = super()._get_observations()
        # Keep policy observations and proprioceptive history unchanged. Stage 2
        # uses these forces only for its privileged teacher's adaptation target.
        self.priv_info_buf[:, 9:21] = self._fingertip_object_forces().flatten(start_dim=1)
        return obs

    def _get_rewards(self):
        reward = super()._get_rewards()
        forces = self._fingertip_object_forces()
        contact_count = (forces.norm(dim=-1) > self.contact_force_threshold).sum(dim=-1).float()
        contact_reward = self.contact_reward_scale * contact_count
        self.extras["fingertip_contact_count"] = contact_count.mean()
        self.extras["fingertip_contact_reward"] = contact_reward.mean()
        return reward + contact_reward
