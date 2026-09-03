# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Modified by Wonik Robotics (2025)
# IsaacLab port of hora/tasks/allegro_hand_hora.py (originally against
# isaacgym.gymapi). Behavior (reward, observations, domain randomization,
# reset logic) is a direct port; only the simulator API underneath changed.
# --------------------------------------------------------

import os
import xml.etree.ElementTree as ET
from glob import glob

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul, sample_uniform, saturate, scale_transform

# IsaacLab's naming is the inverse of IsaacGym's here: `scale_transform` normalizes a
# value from (lower, upper) into [-1, 1] -- what IsaacGym called `unscale` -- and
# `saturate` is IsaacGym's `tensor_clamp`. Aliased to avoid re-deriving this each read.
unscale = scale_transform
tensor_clamp = saturate

from . import sim_cfg as hora_sim_cfg

# repo root, e.g. ".../allegro_inhand_rotation/" -- this file lives at
# hora/tasks/isaaclab/, three levels below the repo root.
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

_OBJECT_ASSET_CFG = SceneEntityCfg("object")
_ROBOT_ASSET_CFG = SceneEntityCfg("robot")


def _stable_usd_cache_dir(source_path: str) -> str:
    """A fixed, content-hashed cache dir next to `source_path`.

    Passing this as `usd_dir` (instead of leaving it `None`, which defaults to a fresh
    `/tmp/IsaacLab/usd_<timestamp>_<random>` dir every call) is what makes
    `AssetConverterBase`'s lazy-conversion check actually able to hit: it compares an
    MD5 of the source file + converter options against `usd_dir/.asset_hash` and skips
    reconversion when they match, both within one run and across reruns.
    """
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(os.path.dirname(source_path), "usd_cache", stem)


def _parse_primitive_geometry(urdf_path: str) -> dict:
    """Read the single `<box>`/`<cylinder>`/`<sphere>` collision geometry out of one
    of the `assets/{cuboid,cylinder,sphere}/*/*.urdf` object files.

    These objects are spawned as native IsaacLab shape prims (`CuboidCfg`/etc.)
    instead of being round-tripped through the URDF->USD converter -- see
    `AllegroHandHoraEnv._build_object_cfg` for why.
    """
    root = ET.parse(urdf_path).getroot()
    geom = root.find("./link/collision/geometry")
    box = geom.find("box")
    if box is not None:
        x, y, z = (float(v) for v in box.get("size").split())
        return {"kind": "cuboid", "size": (x, y, z)}
    cylinder = geom.find("cylinder")
    if cylinder is not None:
        return {"kind": "cylinder", "radius": float(cylinder.get("radius")), "height": float(cylinder.get("length"))}
    sphere = geom.find("sphere")
    if sphere is not None:
        return {"kind": "sphere", "radius": float(sphere.get("radius"))}
    raise ValueError(f"{urdf_path}: no <box>/<cylinder>/<sphere> collision geometry found")


@configclass
class EventCfg:
    """Domain randomization applied once at env creation (`mode="startup"`).

    The original only ever samples object mass/COM/friction once, in `_create_envs`;
    `reset_idx` never re-samples them (unlike the PD gains, which HORA does re-sample
    every episode -- ported directly in `AllegroHandHoraEnv._reset_idx` instead, since
    they drive a hand-rolled torque law rather than an IsaacLab actuator model).
    """

    object_mass: EventTerm | None = None
    object_com: EventTerm | None = None
    object_friction: EventTerm | None = None
    # the original samples ONE friction value shared by hand and object; IsaacLab's
    # per-asset material-randomization term samples each asset independently, so this
    # is a reasonable but not bit-exact stand-in (see EventCfg docstring above).
    hand_friction: EventTerm | None = None


@configclass
class AllegroHandHoraEnvCfg(DirectRLEnvCfg):
    # populated by `build_hora_env_cfg` below
    decimation: int = 6
    episode_length_s: float = 20.0
    action_space: int = 16
    observation_space: int = 96
    state_space: int = 0

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg()
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.25,
        # object geometry/scale varies per env (see `_build_object_cfg`); PhysX's fast
        # replicate-from-source-0 path assumes identical per-env assets, so it must
        # stay off for the whole scene (this also applies to the -- otherwise
        # identical -- hand articulations, at some perf cost).
        replicate_physics=False,
    )
    robot_cfg: ArticulationCfg = None
    events: EventCfg = EventCfg()

    # the raw `config["env"]` dict from configs/task/*.yaml -- everything HORA-specific
    # (reward scales, randomization ranges, priv-info toggles, object type list, grasp
    # cache name, controller gains) is read from here rather than re-modeled as typed
    # configclass fields one by one.
    hora_cfg: dict = None
    # the raw `config["sim"]` dict, needed for the per-asset PhysX property translation
    # in `sim_cfg.py` (solver iterations, contact offsets, depenetration velocity).
    hora_sim_dict: dict = None


def build_hora_env_cfg(config: dict, sim_device: str, graphics_device_id: int, headless: bool) -> AllegroHandHoraEnvCfg:
    # strip any leftover OmegaConf `ListConfig`/`DictConfig` nodes before this dict gets
    # attached to `AllegroHandHoraEnvCfg` -- see `to_plain_container`'s docstring.
    config = hora_sim_cfg.to_plain_container(config)
    env_cfg = config["env"]
    sim_dict = config["sim"]
    controller_cfg = env_cfg["controller"]
    decimation = controller_cfg["controlFrequencyInv"]
    dt = sim_dict["dt"]

    hand_asset_file = os.path.join(REPO_ROOT, env_cfg["asset"]["handAsset"])
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=hand_asset_file,
            usd_dir=_stable_usd_cache_dir(hand_asset_file),
            fix_base=True,
            merge_fixed_joints=False,
            activate_contact_sensors=True,
            # `target_type="none"` documents that it forces stiffness/damping to 0.0 at
            # conversion time, but `gains.stiffness` still has no default and fails
            # `cfg.validate()` if left unset -- so set it explicitly too.
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            articulation_props=hora_sim_cfg.build_articulation_root_props(sim_dict),
            collision_props=hora_sim_cfg.build_collision_props(sim_dict),
            rigid_props=hora_sim_cfg.build_rigid_body_props(sim_dict, disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # matches `_init_object_pose`'s `allegro_hand_start_pose`: pos (0, 0, 0.5),
            # rot = Quat.from_axis_angle((0,1,0), -pi/2) * Quat.from_axis_angle((1,0,0), pi/2)
            # re-expressed in IsaacLab's (w, x, y, z) quaternion convention.
            pos=(0.0, 0.0, 0.5),
            rot=(0.5, 0.5, -0.5, 0.5),
            # this spawn-time default is immediately overwritten by `_reset_idx`'s
            # grasp-cache (or canonical-pose, for the Grasp env) pose on the very first
            # reset -- but IsaacLab validates it against the joint's own limits at
            # articulation init regardless. `joint_12_0` (thumb base/CMC) is the only
            # one of the 16 joints whose range, [0.263, 1.396], excludes 0; reusing the
            # same value the repo's own `CANONICAL_POSE[4]` uses for it.
            # a wildcard alongside an exact-name key both matching the same joint is an
            # ambiguous "multiple matches" error in IsaacLab -- needs a negative
            # lookahead to exclude "joint_12_0" from the wildcard, not just a more
            # specific second entry (same trick `isaaclab_assets`' own `ALLEGRO_HAND_CFG`
            # uses for its equivalent "thumb_joint_0" special case).
            joint_pos={"^(?!joint_12_0$).*$": 0.0, "joint_12_0": 1.104},
        ),
        actuators={
            # torque_control=True: PhysX applies pure effort commands, the P/D law
            # itself runs in Python every physics sub-step (`_apply_action`), matching
            # the original's `DOF_MODE_EFFORT` branch in `_create_envs`. torque_control
            # =False: PhysX's own implicit PD drive does the work instead, matching the
            # original's `DOF_MODE_POS` branch -- `_apply_action` just forwards
            # `cur_targets` via `set_joint_position_target` every sub-step there.
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=0.5,
                stiffness=0.0 if controller_cfg["torque_control"] else controller_cfg["pgain"],
                damping=0.0 if controller_cfg["torque_control"] else controller_cfg["dgain"],
                friction=0.01,
                armature=0.001,
            ),
        },
    )

    return AllegroHandHoraEnvCfg(
        decimation=decimation,
        # ceil()'d by DirectRLEnv into `max_episode_length`; the tiny epsilon guards
        # against float rounding pushing the ceil up by one step.
        episode_length_s=env_cfg["episodeLength"] * dt * decimation - 1e-6,
        action_space=env_cfg["numActions"],
        observation_space=env_cfg["numObservations"],
        state_space=0,
        sim=hora_sim_cfg.build_simulation_cfg(sim_dict, sim_device),
        scene=InteractiveSceneCfg(num_envs=env_cfg["numEnvs"], env_spacing=env_cfg["envSpacing"], replicate_physics=False),
        robot_cfg=robot_cfg,
        events=_build_event_cfg(env_cfg["randomization"]),
        hora_cfg=env_cfg,
        hora_sim_dict=sim_dict,
    )


def _build_event_cfg(rand_cfg: dict) -> EventCfg:
    events = EventCfg()
    if rand_cfg["randomizeMass"]:
        events.object_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": _OBJECT_ASSET_CFG,
                "mass_distribution_params": (rand_cfg["randomizeMassLower"], rand_cfg["randomizeMassUpper"]),
                "operation": "abs",
            },
        )
    if rand_cfg["randomizeCOM"]:
        events.object_com = EventTerm(
            func=hora_sim_cfg.randomize_rigid_object_com,
            mode="startup",
            params={
                "asset_cfg": _OBJECT_ASSET_CFG,
                "com_range": {
                    "x": (rand_cfg["randomizeCOMLower"], rand_cfg["randomizeCOMUpper"]),
                    "y": (rand_cfg["randomizeCOMLower"], rand_cfg["randomizeCOMUpper"]),
                    "z": (rand_cfg["randomizeCOMLower"], rand_cfg["randomizeCOMUpper"]),
                },
            },
        )
    if rand_cfg["randomizeFriction"]:
        friction_range = (rand_cfg["randomizeFrictionLower"], rand_cfg["randomizeFrictionUpper"])
        events.object_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": _OBJECT_ASSET_CFG,
                "static_friction_range": friction_range,
                "dynamic_friction_range": friction_range,
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
        events.hand_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": _ROBOT_ASSET_CFG,
                "static_friction_range": friction_range,
                "dynamic_friction_range": friction_range,
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
    return events


class AllegroHandHoraEnv(DirectRLEnv):
    cfg: AllegroHandHoraEnvCfg

    def __init__(self, cfg: AllegroHandHoraEnvCfg, render_mode: str | None = None, **kwargs):
        # ---- everything `_setup_scene` needs must be ready before `super().__init__()`,
        # since that call is what triggers `_setup_scene()` -- mirrors the original's own
        # "before calling init in VecTask, need to do: 1) randomization, 2) priv info,
        # 3) object assets, 4) reward" ordering in `AllegroHandHora.__init__`.
        env_cfg = cfg.hora_cfg
        self.debug_viz = env_cfg["enableDebugVis"]
        self.reset_z_threshold = env_cfg["reset_height_threshold"]
        self.grasp_cache_name = env_cfg["grasp_cache_name"]
        self.base_obj_scale = env_cfg["baseObjScale"]
        self.save_init_pose = env_cfg["genGrasps"]
        self.evaluate = env_cfg.get("on_evaluation", False)
        self.torque_control = env_cfg["controller"]["torque_control"]
        self.control_freq_inv = env_cfg["controller"]["controlFrequencyInv"]
        self.p_gain_scalar = env_cfg["controller"]["pgain"]
        self.d_gain_scalar = env_cfg["controller"]["dgain"]
        self.priv_info_dict = {
            "obj_position": (0, 3),
            "obj_scale": (3, 4),
            "obj_mass": (4, 5),
            "obj_friction": (5, 6),
            "obj_com": (6, 9),
        }
        self._setup_priv_option_config(env_cfg["privInfo"])
        self._setup_domain_rand_config(env_cfg["randomization"])
        self._setup_reward_config(env_cfg["reward"])
        self._setup_object_info(env_cfg["object"])
        self.prop_hist_len = env_cfg["hora"]["propHistoryLen"]
        self.num_env_factors = env_cfg["hora"]["privInfoDim"]
        self.force_scale = env_cfg.get("forceScale", 0.0)
        self.random_force_prob_scalar = env_cfg.get("randomForceProbScalar", 0.0)
        self.force_decay = env_cfg.get("forceDecay", 0.99)
        self.force_decay_interval = env_cfg.get("forceDecayInterval", 0.08)

        super().__init__(cfg, render_mode, **kwargs)

        # ---- everything below needs `self.device`/`self.hand`/`self.object`, which only
        # exist once `super().__init__()` (and thus `_setup_scene()`) has returned.
        self.num_hand_dofs = self.hand.num_joints
        self.num_dofs = self.num_hand_dofs
        self.num_actions = env_cfg["numActions"]
        self.num_obs = env_cfg["numObservations"]
        # `max_episode_length` is a read-only property on `DirectRLEnv` itself, derived
        # from `episode_length_s`/`decimation`/`sim.dt` -- which `build_hora_env_cfg`
        # already set to reproduce `env_cfg["episodeLength"]` exactly, so no assignment
        # is needed (or possible) here.

        # `get_dof_limits()` is per-env batched, shape (num_envs, num_dofs, 2) -- unlike
        # the original's flat (num_dofs,) `allegro_hand_dof_lower/upper_limits` (built
        # once from the asset properties under IsaacGym, identical across envs). Every
        # env shares the same joint limits here too (one hand asset type), so take env
        # 0's slice: keeping these flat is what lets them broadcast correctly against
        # dof_pos tensors indexed down to an env *subset* elsewhere in this file (e.g.
        # `at_reset_env_ids` in `_get_observations`), rather than needing every such
        # indexing site to also subset-index the limits.
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.allegro_hand_dof_lower_limits = joint_pos_limits[0, :, 0]
        self.allegro_hand_dof_upper_limits = joint_pos_limits[0, :, 1]

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        self.torques = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device)

        assert isinstance(self.p_gain_scalar, (int, float)) and isinstance(self.d_gain_scalar, (int, float))
        self.p_gain = torch.full((self.num_envs, self.num_actions), float(self.p_gain_scalar), device=self.device)
        self.d_gain = torch.full((self.num_envs, self.num_actions), float(self.d_gain_scalar), device=self.device)

        self.priv_info_buf = torch.zeros((self.num_envs, self.num_env_factors), device=self.device, dtype=torch.float)
        self.proprio_hist_buf = torch.zeros((self.num_envs, self.prop_hist_len, 32), device=self.device, dtype=torch.float)
        self.obs_buf_lag_history = torch.zeros((self.num_envs, 80, self.num_obs // 3), device=self.device, dtype=torch.float)
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)

        if self.randomize_scale and self.scale_list_init:
            self.saved_grasping_states = {}
            for s in self.randomize_scale_list:
                cache_path = os.path.join(
                    REPO_ROOT, "cache", f"{self.grasp_cache_name}_grasp_50k_s{str(s).replace('.', '')}.npy"
                )
                self.saved_grasping_states[str(s)] = torch.from_numpy(np.load(cache_path)).float().to(self.device)
        else:
            assert self.save_init_pose

        self._compute_intermediate_values()
        self.object_rot_prev = self.object_rot.clone()
        self.object_pos_prev = self.object_pos.clone()

        # Read back the values the startup randomization events (mass/COM/friction)
        # actually applied, and record them into priv_info -- this way priv_info is
        # guaranteed to match what physics is using rather than duplicating the
        # sampling. Matches the original's un-normalized raw storage (no lower/upper
        # passed to `_update_priv_buf` for these three fields in the base Hora task).
        all_envs = slice(None)
        self._update_priv_buf(all_envs, "obj_scale", self._object_scales)
        self._update_priv_buf(all_envs, "obj_mass", self.object.root_physx_view.get_masses()[:, 0].to(self.device))
        # `RigidObject.root_physx_view.get_coms()` is `(count, 7)` -- no per-body
        # dimension, unlike `Articulation`'s `(count, max_links, 7)`.
        self._update_priv_buf(all_envs, "obj_com", self.object.root_physx_view.get_coms()[:, :3].to(self.device))
        friction = self.object.root_physx_view.get_material_properties()[:, 0, 0].to(self.device)
        self._update_priv_buf(all_envs, "obj_friction", friction)

    # ------------------------------------------------------------------
    # scene / asset setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self._build_object_cfg())

        sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _build_object_cfg(self) -> RigidObjectCfg:
        num_envs = self.cfg.scene.num_envs
        num_scales = len(self.randomize_scale_list) if self.randomize_scale else 1
        per_env_scale = np.zeros(num_envs, dtype=np.float32)
        per_env_asset_cfgs = []
        object_rigid_props = hora_sim_cfg.build_rigid_body_props(self.cfg.hora_sim_dict, disable_gravity=False)
        object_collision_props = hora_sim_cfg.build_collision_props(self.cfg.hora_sim_dict)
        # matches the URDFs' own `<mass value="0.05"/>` -- only a placeholder default
        # anyway, since the `object_mass` startup event overwrites it when enabled.
        object_mass_props = sim_utils.MassPropertiesCfg(mass=0.05)

        # Cuboid/cylinder/sphere objects are spawned as native IsaacLab shape prims
        # (not round-tripped through the URDF->USD converter): Isaac Sim 5.1's URDF
        # importer splits its output into a top usd that *payloads* a
        # `configuration/*_physics.usd` sublayer, whose own geometry-defining
        # `/visuals`/`/colliders` scopes only get pulled in via an internal
        # (same-layer, empty-assetPath) reference from within the payloaded subtree --
        # confirmed empty/invisible both via raw `Usd.Stage.Open` and by opening the
        # converted file directly in Kit on an empty stage. `Articulation` (the hand)
        # renders fine through this same pattern; `RigidObject` (the object) doesn't --
        # spawning shapes natively sidesteps the whole broken composition path.
        for i in range(num_envs):
            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                bucket = self.randomize_scale_list[i % num_scales]
                obj_scale = float(np.random.uniform(bucket - 0.025, bucket + 0.025))
            per_env_scale[i] = obj_scale

            # object shape/size variety (approximates `sampleProb`'s per-type weighting
            # -- `MultiAssetSpawnerCfg` only supports a flat, order-preserving list, so
            # each env's chosen shape is resolved into its own spawn cfg up front).
            type_id = int(np.random.choice(len(self.object_type_list), p=self.object_type_prob))
            spec = self.object_shape_specs[self.object_type_list[type_id]]
            shape_kwargs = dict(rigid_props=object_rigid_props, collision_props=object_collision_props, mass_props=object_mass_props)
            if spec["kind"] == "cuboid":
                sx, sy, sz = spec["size"]
                shape_cfg = sim_utils.CuboidCfg(size=(sx * obj_scale, sy * obj_scale, sz * obj_scale), **shape_kwargs)
            elif spec["kind"] == "cylinder":
                shape_cfg = sim_utils.CylinderCfg(
                    radius=spec["radius"] * obj_scale, height=spec["height"] * obj_scale, **shape_kwargs
                )
            else:
                shape_cfg = sim_utils.SphereCfg(radius=spec["radius"] * obj_scale, **shape_kwargs)
            per_env_asset_cfgs.append(shape_cfg)
        self._object_scales = torch.from_numpy(per_env_scale).to(self.device)

        _, _, obj_pos = self._init_object_pose()
        return RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=MultiAssetSpawnerCfg(assets_cfg=per_env_asset_cfgs, random_choice=False),
            init_state=RigidObjectCfg.InitialStateCfg(pos=obj_pos),
        )

    def _init_object_pose(self):
        # direct port of `_init_object_pose` in the original allegro_hand_hora.py
        hand_pos = (0.0, 0.0, 0.5)
        hand_rot = (0.5, 0.5, -0.5, 0.5)
        pose_dx = -0.01
        pose_dy = -0.01
        obj_x = hand_pos[0] + pose_dx
        obj_y = hand_pos[1] + pose_dy  # overrides the `+ pose_dy` computation, as in the original
        object_z = 0.66 if self.save_init_pose else 0.65
        if "internal" not in self.grasp_cache_name:
            object_z -= 0.02
        return hand_pos, hand_rot, (obj_x, obj_y, object_z)

    # ------------------------------------------------------------------
    # low-level control (runs once per physics sub-step, decimation times per env step)
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().to(self.device)
        targets = self.prev_targets + (1.0 / 24.0) * self.actions
        self.cur_targets[:] = tensor_clamp(targets, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.prev_targets[:] = self.cur_targets.clone()

        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos

        if self.force_scale > 0.0:
            self.rb_forces *= self.force_decay ** (self.physics_dt / self.force_decay_interval)
            obj_mass = self.object.root_physx_view.get_masses()[:, 0].to(self.device)
            prob = self.random_force_prob_scalar
            force_env_ids = (torch.rand(self.num_envs, device=self.device) < prob).nonzero(as_tuple=False).squeeze(-1)
            self.rb_forces[force_env_ids, 0, :] = (
                torch.randn((len(force_env_ids), 3), device=self.device) * obj_mass[force_env_ids, None] * self.force_scale
            )
            # `forces`/`torques` must stay 3D -- (len(env_ids), len(body_ids), 3) -- even
            # for a single-body RigidObject; slicing off the body dim (`rb_forces[:, 0]`)
            # collapses it to 2D, which `wp.from_torch(..., dtype=wp.vec3f)` then turns
            # into a 1D array of vec3s instead of the 2D one the kernel expects.
            self.object.set_external_force_and_torque(self.rb_forces, torch.zeros_like(self.rb_forces))

    def _apply_action(self) -> None:
        if not self.torque_control:
            # PhysX's own implicit PD drive (configured via the `fingers` actuator's
            # stiffness/damping in `build_hora_env_cfg`) does the P/D law here, so
            # `self.torques`/`self.dof_vel_finite_diff` are never touched in this mode
            # -- matching the original, where they likewise stay at their zero-init
            # value for the lifetime of a `torque_control=False` run (only the
            # `if self.torque_control:` branch of `update_low_level_control` ever
            # wrote them), so `torque_penalty`/`work_penalty` are always zero here.
            self.hand.set_joint_position_target(self.cur_targets)
            return

        previous_dof_pos = self.hand.data.joint_pos.clone()
        dof_pos = self.hand.data.joint_pos
        dof_vel = (dof_pos - previous_dof_pos) / self.physics_dt
        self.dof_vel_finite_diff = dof_vel.clone()

        pos_error = self.cur_targets - dof_pos
        torques = self.p_gain * pos_error - self.d_gain * dof_vel
        self.torques = torch.clip(torques, -0.5, 0.5)
        self.hand.set_joint_effort_target(self.torques)

    # ------------------------------------------------------------------
    # observations / reward / termination
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self):
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w
        self.allegro_hand_dof_pos = self.hand.data.joint_pos
        self.allegro_hand_dof_vel = self.hand.data.joint_vel

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise = (torch.rand_like(self.allegro_hand_dof_pos) * 2.0 - 1.0) * self.joint_noise_scale
        cur_dof_normalized = unscale(
            joint_noise + self.allegro_hand_dof_pos, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits
        ).unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_dof_normalized, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.at_reset_buf[at_reset_env_ids] = 0
        # `at_reset_env_ids` is empty on essentially every step after the first --
        # indexing down to a zero-length tensor and feeding it into the TorchScript-
        # jitted `scale_transform` (`unscale`) trips "Global alloc not supported yet"
        # in that kernel. Nothing to update anyway when it's empty, so just skip.
        if len(at_reset_env_ids) > 0:
            self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = unscale(
                self.allegro_hand_dof_pos[at_reset_env_ids], self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits
            ).unsqueeze(1)
            self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = self.allegro_hand_dof_pos[at_reset_env_ids].unsqueeze(1)

        t_buf = self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1).clone()
        obs = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        obs[:, : t_buf.shape[1]] = t_buf
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len :].clone()

        self._update_priv_buf(slice(None), "obj_position", self.object_pos.clone())

        return {"obs": obs, "policy": obs, "priv_info": self.priv_info_buf, "proprio_hist": self.proprio_hist_buf}

    def _get_rewards(self) -> torch.Tensor:
        self.rot_axis_buf[:, -1] = -1
        pose_diff_penalty = ((self.allegro_hand_dof_pos - self.init_pose_buf) ** 2).sum(-1)
        torque_penalty = (self.torques**2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2

        angdiff = quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.physics_dt)
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)

        object_linvel = (self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.physics_dt)
        object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)

        reward = (
            self.rotate_reward_scale * rotate_reward
            + self.object_linvel_penalty_scale * object_linvel_penalty
            + self.pose_diff_penalty_scale * pose_diff_penalty
            + self.torque_penalty_scale * torque_penalty
            + self.work_penalty_scale * work_penalty
        )

        self.extras["rotation_reward"] = rotate_reward.mean()
        self.extras["object_linvel_penalty"] = object_linvel_penalty.mean()
        self.extras["pose_diff_penalty"] = pose_diff_penalty.mean()
        self.extras["work_done"] = work_penalty.mean()
        self.extras["torques"] = torque_penalty.mean()
        self.extras["roll"] = object_angvel[:, 0].mean()
        self.extras["pitch"] = object_angvel[:, 1].mean()
        self.extras["yaw"] = object_angvel[:, 2].mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        fell = self.object_pos[:, -1] < self.reset_z_threshold
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fell, time_out

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = sample_uniform(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions), device=self.device
            )
            self.d_gain[env_ids] = sample_uniform(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions), device=self.device
            )
        self.rb_forces[env_ids] = 0.0

        num_scales = len(self.randomize_scale_list)
        dof_pos = self.hand.data.joint_pos[env_ids].clone()
        object_pose = torch.cat([self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids], self.object.data.root_quat_w[env_ids]], dim=-1)
        for n_s in range(num_scales):
            scale_key = str(self.randomize_scale_list[n_s])
            bucket_mask = (env_ids % num_scales) == n_s
            if not bucket_mask.any():
                continue
            n_ids = int(bucket_mask.sum().item())
            sampled_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=n_ids)
            sampled_pose = self.saved_grasping_states[scale_key][sampled_idx].clone()
            dof_pos[bucket_mask] = sampled_pose[:, :16]
            object_pose[bucket_mask] = sampled_pose[:, 16:]
            self.init_pose_buf[env_ids[bucket_mask]] = sampled_pose[:, :16].clone()

        dof_vel = torch.zeros_like(dof_pos)
        world_pos = object_pose[:, :3] + self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(torch.cat([world_pos, object_pose[:, 3:7]], dim=-1), env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)

        self.prev_targets[env_ids, : self.num_hand_dofs] = dof_pos
        self.cur_targets[env_ids, : self.num_hand_dofs] = dof_pos
        if not self.torque_control:
            self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.obs_buf_lag_history[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
        self._compute_intermediate_values()

    # ------------------------------------------------------------------
    # config parsing (direct ports of the original's `_setup_*` helpers)
    # ------------------------------------------------------------------
    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config["enableObjPos"]
        self.enable_priv_obj_mass = p_config["enableObjMass"]
        self.enable_priv_obj_scale = p_config["enableObjScale"]
        self.enable_priv_obj_com = p_config["enableObjCOM"]
        self.enable_priv_obj_friction = p_config["enableObjFriction"]

    def _setup_domain_rand_config(self, rand_config):
        # mass/COM/friction are randomized once at env creation for this (base Hora)
        # task -- see `EventCfg` -- but `AllegroHandGraspEnv` re-randomizes mass on every
        # reset, so these bounds are kept here for both subclasses to share.
        self.randomize_mass = rand_config["randomizeMass"]
        self.randomize_mass_lower = rand_config["randomizeMassLower"]
        self.randomize_mass_upper = rand_config["randomizeMassUpper"]
        self.randomize_com = rand_config["randomizeCOM"]
        self.randomize_com_lower = rand_config["randomizeCOMLower"]
        self.randomize_com_upper = rand_config["randomizeCOMUpper"]
        self.randomize_friction = rand_config["randomizeFriction"]
        self.randomize_friction_lower = rand_config["randomizeFrictionLower"]
        self.randomize_friction_upper = rand_config["randomizeFrictionUpper"]
        self.randomize_scale = rand_config["randomizeScale"]
        self.scale_list_init = rand_config["scaleListInit"]
        self.randomize_scale_list = rand_config["randomizeScaleList"]
        self.randomize_pd_gains = rand_config["randomizePDGains"]
        self.randomize_p_gain_lower = rand_config["randomizePGainLower"]
        self.randomize_p_gain_upper = rand_config["randomizePGainUpper"]
        self.randomize_d_gain_lower = rand_config["randomizeDGainLower"]
        self.randomize_d_gain_upper = rand_config["randomizeDGainUpper"]
        self.joint_noise_scale = rand_config["jointNoiseScale"]

    def _setup_reward_config(self, r_config):
        self.angvel_clip_min = r_config["angvelClipMin"]
        self.angvel_clip_max = r_config["angvelClipMax"]
        self.rotate_reward_scale = r_config["rotateRewardScale"]
        self.object_linvel_penalty_scale = r_config["objLinvelPenaltyScale"]
        self.pose_diff_penalty_scale = r_config["poseDiffPenaltyScale"]
        self.torque_penalty_scale = r_config["torquePenaltyScale"]
        self.work_penalty_scale = r_config["workPenaltyScale"]

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        s, e = self.priv_info_dict[name]
        if getattr(self, f"enable_priv_{name}"):
            if isinstance(value, list):
                value = torch.tensor(value, dtype=torch.float, device=self.device)
            # a flat (N,) tensor (one scalar per env, e.g. obj_scale/obj_mass/
            # obj_friction) doesn't broadcast into a width-1 `priv_info_buf[:, s:e]`
            # slice on its own -- needs an explicit trailing dim to become (N, 1).
            if torch.is_tensor(value) and value.dim() == 1 and (e - s) == 1:
                value = value.unsqueeze(-1)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0

    def _setup_object_info(self, o_config):
        self.object_type = o_config["type"]
        raw_prob = o_config["sampleProb"]
        assert sum(raw_prob) == 1

        primitive_list = self.object_type.split("+")
        self.object_type_prob = []
        self.object_type_list = []
        self.object_shape_specs = {"simple_tennis_ball": {"kind": "sphere", "radius": 0.04}}

        for p_id, prim in enumerate(primitive_list):
            for shape in ("cuboid", "cylinder", "sphere"):
                if shape in prim:
                    subset_name = self.object_type.split("_")[-1]
                    files = sorted(glob(os.path.join(REPO_ROOT, f"assets/{shape}/{subset_name}/*.urdf")))
                    names = [f"{shape}_{i}" for i in range(len(files))]
                    self.object_type_list += names
                    for i, f in enumerate(files):
                        self.object_shape_specs[f"{shape}_{i}"] = _parse_primitive_geometry(f)
                    self.object_type_prob += [raw_prob[p_id] / len(names) for _ in names]
                    break
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]

        assert len(self.object_type_list) == len(self.object_type_prob)


def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """Direct port of `quat_to_axis_angle` from the original allegro_hand_hora.py.

    That version indexed [..., :3] for the vector part and [..., 3:] for the real part,
    matching IsaacGym's (x, y, z, w) quaternion convention. IsaacLab's `quat_mul`/
    `quat_conjugate` return (w, x, y, z) instead, so the indexing here is [..., 1:4]
    (vector) / [..., 0:1] (real) -- same algebra, different component order.
    """
    norms = torch.norm(quaternions[..., 1:4], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 0:1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:4] / sin_half_angles_over_angles
