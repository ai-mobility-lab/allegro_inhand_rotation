# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Modified by Wonik Robotics (2025)
# Translates the repo's IsaacGym-flavored `sim` config dict (configs/task/*.yaml,
# the `sim:` block) into the IsaacLab configclasses that carry the equivalent
# settings. Every helper here takes that `sim` sub-dict directly (not the full
# hydra config), since it's the only part these translations need.
# --------------------------------------------------------

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform


def randomize_rigid_object_com(env, env_ids: torch.Tensor | None, com_range: dict, asset_cfg: SceneEntityCfg):
    """`isaaclab.envs.mdp.randomize_rigid_body_com`, but for a `RigidObject`.

    That mdp term indexes `coms[env_ids[:, None], body_ids, :3]`, assuming the
    `(count, max_links, 7)` shape `Articulation.root_physx_view.get_coms()` returns.
    `RigidObject.root_physx_view.get_coms()` instead returns `(count, 7)` -- there's no
    per-body dimension for a single-rigid-body asset -- so that indexing raises
    "too many indices for tensor of dimension 2". Same sampling logic, 2D indexing.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")

    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids, :3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)


def to_plain_container(obj):
    """Recursively strip any OmegaConf `DictConfig`/`ListConfig` out of `obj`.

    `hora/utils/reformat.py`'s `omegaconf_to_dict` only converts nested `DictConfig`
    values, so list-valued YAML fields (`randomizeScaleList`, `sampleProb`, `gravity`,
    ...) survive as `ListConfig` objects even after that pass. That was harmless under
    IsaacGym -- nothing ever deep-walked the config -- but IsaacLab's `configclass`
    machinery does exactly that (`DirectRLEnvCfg.validate()`), and a `ListConfig` isn't
    a plain `list`, so it recurses into OmegaConf's internal (self-referential via
    parent pointers) node structure instead of the list's elements -- a `RecursionError`.
    Every config dict that ends up stored on an IsaacLab configclass field (`hora_cfg`,
    `hora_sim_dict`) must be sanitized through this first.
    """
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: to_plain_container(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_container(v) for v in obj]
    return obj


def build_simulation_cfg(sim_dict: dict, sim_device: str) -> sim_utils.SimulationCfg:
    """Build the top-level `SimulationCfg` from `config["sim"]`.

    Only the knobs that map onto IsaacLab's *scene-wide* PhysX settings live here.
    Per-asset knobs (solver iteration counts, contact/rest offset, max depenetration
    velocity) are handled by `build_articulation_root_props`/`build_rigid_body_props`/
    `build_collision_props` below and applied on the hand/object spawn configs instead,
    since IsaacLab moved those from being sim-wide defaults to per-asset properties.

    IsaacGym-only knobs with no IsaacLab equivalent (`num_threads`, `num_subscenes`,
    `contact_collection`, `max_gpu_contact_pairs`, `default_buffer_size_multiplier`)
    are intentionally dropped rather than mapped. GPU vs. CPU pipeline is controlled
    by IsaacLab through `sim.device` rather than a separate `use_gpu_pipeline` flag.
    """
    physx_cfg = sim_dict.get("physx", {})
    return sim_utils.SimulationCfg(
        device=sim_device if sim_dict["use_gpu_pipeline"] else "cpu",
        dt=sim_dict["dt"],
        gravity=tuple(sim_dict["gravity"]),
        physx=sim_utils.PhysxCfg(
            solver_type=physx_cfg.get("solver_type", 1),
            bounce_threshold_velocity=physx_cfg.get("bounce_threshold_velocity", 0.2),
        ),
    )


def build_articulation_root_props(sim_dict: dict) -> sim_utils.ArticulationRootPropertiesCfg:
    """Solver iteration counts: sim-wide in IsaacGym, per-articulation in IsaacLab."""
    physx_cfg = sim_dict.get("physx", {})
    return sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=physx_cfg.get("num_position_iterations", 8),
        solver_velocity_iteration_count=physx_cfg.get("num_velocity_iterations", 0),
    )


def build_collision_props(sim_dict: dict) -> sim_utils.CollisionPropertiesCfg:
    physx_cfg = sim_dict.get("physx", {})
    return sim_utils.CollisionPropertiesCfg(
        contact_offset=physx_cfg.get("contact_offset", 0.002),
        rest_offset=physx_cfg.get("rest_offset", 0.0),
    )


def build_rigid_body_props(sim_dict: dict, disable_gravity: bool) -> sim_utils.RigidBodyPropertiesCfg:
    physx_cfg = sim_dict.get("physx", {})
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=disable_gravity,
        max_depenetration_velocity=physx_cfg.get("max_depenetration_velocity", 1000.0),
    )
