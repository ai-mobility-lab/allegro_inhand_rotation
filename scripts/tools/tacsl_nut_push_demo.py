# --------------------------------------------------------
# Standalone smoke test for the DIGIT `VisuoTactileSensor`s wired onto this repo's Allegro-DIGIT
# hand, mirroring the sibling `inhand_rotation` repo's own `scripts/tools/tacsl_sensor_demo.py`
# test *design*, not just its intent.
#
# `scripts/tools/tacsl_sensor_demo.py` (this repo's other tactile smoke test) drives the full
# `DatasetAllegroHandHoraEnv` through `env.reset()`, which samples a pre-formed grasp from the
# task's own grasp cache -- fingertips are typically already touching the object at reset, but
# "typically" is doing a lot of work: a bad grasp sample, a stale cache, or a mismatched object
# scale can leave a finger with no contact at all, and there is no independent way to tell that
# apart from an actual sensor/USD-wiring bug from that script's output alone.
#
# This script instead reproduces `inhand_rotation`'s own test design: skip the hand's grasp
# entirely and give each fingertip its own small "nut" (the same `factory_nut_m16.usd` contact
# object `inhand_rotation`'s demo uses), placed directly in front of *that* elastomer and pushed
# straight into it by a constant force, independent of the hand's joint configuration. Each of the
# 4 sensors gets an independent, guaranteed-contact readout regardless of hand pose -- so if a
# finger's tactile RGB/depth still looks wrong here, the bug is in the sensor/USD wiring itself
# (shared with `tacsl_sensor_demo.py`, and with `inhand_rotation`'s own demo), not in the grasp
# cache or `AllegroHandHoraEnv`'s task-specific reset logic.
#
# Unlike `inhand_rotation`'s demo (a bare `InteractiveScene` built around its own vendored
# `ALLEGRO_HAND_DIGIT_CFG` USD asset), this one builds the hand the way this repo actually does at
# train/dataset-collection time: `hora.tasks.isaaclab.allegro_hand_hora_env.build_hora_env_cfg`
# converts `configs/task/*.yaml`'s `handAsset` URDF the same way `AllegroHandHoraEnv` does, so the
# `ArticulationCfg` under test here is byte-for-byte what training/dataset collection uses -- not a
# hand-rolled stand-in. Everything downstream (elastomer link names, DIGIT camera intrinsics/
# offsets, Taxim render config) is pulled from `scripts/dataset_collection/sensors.py`, the same
# helpers `DatasetAllegroHandHoraEnv` itself uses, so this is a strict subset of that env's sensor
# rig with the grasp/task machinery removed -- not a separate reimplementation to keep in sync.
#
# Requires `--enable_cameras` (the DIGIT sensors are tiled cameras) and real DIGIT Taxim
# calibration data (see `--digit_calib_dir`, default `assets/digit_data/`) -- without it,
# `enable_camera_tactile=True` sensor init raises `FileNotFoundError`.
#
# Usage (run from the `allegro_inhand_rotation` repo root):
#
#   ./isaaclab.sh -p scripts/tools/tacsl_nut_push_demo.py --enable_cameras --headless \
#       --steps 100 --save_every 20
#
# Extra Hydra-style overrides (e.g. to point at a different hand asset) can be appended after a
# bare `--`, e.g. `-- task=LeftAllegroHandHora`.
# --------------------------------------------------------

import argparse
import gc
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

# Same resolvers `train.py`/`collect_stage1_feelsight_dataset.py`/`tacsl_sensor_demo.py` register
# before composing any Hydra config under `configs/` -- required for `hydra.compose` below.
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)

# this file lives at scripts/tools/, two levels below the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DIGIT_CALIB_DIR = REPO_ROOT / "assets/digit_data"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task", default="LeftAllegroHandDigitHora",
        help="task config to pull the hand's `asset.handAsset` URDF + sim settings from (only the "
        "hand/sim config is used -- no object, reward, or grasp-cache config from this task is "
        "touched).",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
    parser.add_argument("--env_spacing", type=float, default=0.5, help="Distance (m) between env origins.")
    parser.add_argument("--steps", type=int, default=100, help="Number of physics steps to run.")
    parser.add_argument("--save_every", type=int, default=20, help="Save tactile images every N steps (and at step 0).")
    parser.add_argument("--save_viz_dir", default=str(REPO_ROOT / "outputs/tacsl_nut_push_demo"), help="Directory to save PNGs into.")
    parser.add_argument("--digit_calib_dir", default=str(DEFAULT_DIGIT_CALIB_DIR), help="dir with bg.jpg + polycalib.npz + real_bg.npy (Taxim calibration)")
    parser.add_argument("--digit_depth_scale", type=float, default=33333.34, help="gel-deformation-meters -> 8-bit pixel scale for the saved deformation frames")
    parser.add_argument("--tactile_mask_eps", type=float, default=3e-4, help="gel deformation (m) above which a tactile pixel counts as 'contact'")
    parser.add_argument(
        "--nut_standoff", type=float, default=0.01, help="Initial gap (m) between each nut and its elastomer's centroid."
    )
    parser.add_argument("--push_force", type=float, default=1.0, help="Constant force (N) pushing each nut onto its pad.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        extra_overrides = argv[split + 1:]
        argv = argv[:split]
    else:
        extra_overrides = []

    parser = build_arg_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(argv)

    if not args_cli.enable_cameras:
        raise SystemExit(
            "This script needs Isaac Sim's offscreen render pipeline for the DIGIT tactile "
            "cameras -- rerun with --enable_cameras."
        )

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # ---- everything below may safely import isaaclab / hora.tasks ----
    sys.path.insert(0, str(REPO_ROOT))  # so `import hora...` resolves regardless of cwd
    sys.path.insert(0, str(SCRIPTS_DIR))  # so `import dataset_collection...` resolves

    import hydra
    import cv2
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

    from hora.tasks.isaaclab.allegro_hand_hora_env import build_hora_env_cfg
    from hora.utils.misc import set_seed
    from hora.utils.reformat import omegaconf_to_dict

    from dataset_collection.sensors import (
        ELASTOMER_LINKS,
        FINGER_NAMES,
        HOUSING_LINKS,
        build_digit_camera_cfg,
        build_digit_render_cfg,
    )

    set_seed(args_cli.seed)

    # ---- pull just the hand asset + sim settings out of the task config -- no object/reward/
    # grasp-cache config from this task is used (there is no object or grasp cache in this test).
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        overrides = [
            f"task={args_cli.task}",
            f"headless={args_cli.headless}",
            f"sim_device={args_cli.device}",
            "wandb.enabled=False",
        ]
        overrides += extra_overrides
        cfg = hydra.compose(config_name="config", overrides=overrides)

    task_cfg_dict = omegaconf_to_dict(cfg.task)
    hora_env_cfg = build_hora_env_cfg(task_cfg_dict, cfg.sim_device, cfg.graphics_device_id, cfg.headless)
    robot_cfg = hora_env_cfg.robot_cfg
    print(f"[INFO]: task={args_cli.task} hand_asset={task_cfg_dict['env']['asset']['handAsset']}")

    digit_render_cfg = build_digit_render_cfg(args_cli.digit_calib_dir)

    # ---- one nut per fingertip -- same contact object as inhand_rotation's own demo's
    # `--contact_object_type nut` case. Placeholder pose here (below the ground, out of the way);
    # each nut gets repositioned at its own elastomer at runtime in `push_nuts_onto_elastomers`,
    # once the elastomer's actual world pose is known.
    nut_spawn = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1,
            max_angular_velocity=180.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        scale=[0.5, 0.5, 0.5],
    )
    nut_init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0), rot=(1.0, 0.0, 0.0, 0.0))

    @configclass
    class TacslNutPushSceneCfg(InteractiveSceneCfg):
        """This repo's Allegro-DIGIT hand with all four tactile sensors, plus one pushable nut
        per fingertip -- the nut-push equivalent of `DatasetAllegroHandHoraEnv`'s sensor rig
        (`scripts/dataset_collection/sensors.py`), without the grasp-cache-dependent hand/object."""

        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        robot = robot_cfg

        index_nut = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/index_nut", spawn=nut_spawn, init_state=nut_init_state)
        middle_nut = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/middle_nut", spawn=nut_spawn, init_state=nut_init_state)
        ring_nut = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/ring_nut", spawn=nut_spawn, init_state=nut_init_state)
        thumb_nut = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/thumb_nut", spawn=nut_spawn, init_state=nut_init_state)

        index_tactile = VisuoTactileSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{ELASTOMER_LINKS['index']}/tactile_sensor",
            render_cfg=digit_render_cfg,
            enable_camera_tactile=True,
            enable_force_field=False,
            tactile_array_size=(16, 19),
            tactile_margin=0.002,
            camera_cfg=build_digit_camera_cfg(HOUSING_LINKS["index"], digit_render_cfg),
        )
        middle_tactile = index_tactile.replace(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{ELASTOMER_LINKS['middle']}/tactile_sensor",
            camera_cfg=build_digit_camera_cfg(HOUSING_LINKS["middle"], digit_render_cfg),
        )
        ring_tactile = index_tactile.replace(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{ELASTOMER_LINKS['ring']}/tactile_sensor",
            camera_cfg=build_digit_camera_cfg(HOUSING_LINKS["ring"], digit_render_cfg),
        )
        thumb_tactile = index_tactile.replace(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{ELASTOMER_LINKS['thumb']}/tactile_sensor",
            camera_cfg=build_digit_camera_cfg(HOUSING_LINKS["thumb"], digit_render_cfg),
        )

    # (tactile sensor name, elastomer link name, nut asset name) -- one triple per fingertip.
    FINGER_TRIPLES = [(f"{finger}_tactile", ELASTOMER_LINKS[finger], f"{finger}_nut") for finger in FINGER_NAMES]

    def push_nuts_onto_elastomers(scene: InteractiveScene, standoff: float, force: float):
        """Place each nut just outside its elastomer's pad and press it inward with a constant
        force -- direct port of `inhand_rotation`'s own `tacsl_sensor_demo.push_nuts_onto_elastomers`.

        The elastomer's local +X axis is the pad's own outward normal (the gel pad bulges out of
        the housing's local +X face -- see `dataset_collection/sensors.py`'s `DIGIT_CAMERA_OFFSET_POS`
        docstring, which notes this repo's `allegro_digit_left_elastomer.urdf` shares
        `inhand_rotation`'s elastomer geometry); the elastomer link's own authored orientation
        matches the tip housing's, so rotating local +X by the elastomer's own world quaternion
        gives that outward direction directly, with no dependency on the hand's current joint pose.
        """
        robot = scene["robot"]
        for _, elastomer_link_name, nut_name in FINGER_TRIPLES:
            body_idx = robot.find_bodies(elastomer_link_name)[0][0]
            nut = scene[nut_name]

            pad_pos_w = robot.data.body_pos_w[:, body_idx]
            pad_quat_w = robot.data.body_quat_w[:, body_idx]
            local_x = torch.tensor([1.0, 0.0, 0.0], device=scene.device).expand(scene.num_envs, 3)
            outward_w = math_utils.quat_apply(pad_quat_w, local_x)

            adj_pos = torch.tensor([0.0, 0.0, 0.02], device=scene.device).expand(scene.num_envs, 3)
            adj_pos_w = math_utils.quat_apply(pad_quat_w, adj_pos)
            nut_pos_w = pad_pos_w + outward_w * standoff + adj_pos_w
            adj_quat = torch.tensor([0.0, 0.7071068, 0.0, 0.7071068], device=scene.device).expand(scene.num_envs, 4)
            nut_quat_w = math_utils.quat_mul(pad_quat_w, adj_quat)
            nut.write_root_pose_to_sim(torch.cat([nut_pos_w, nut_quat_w], dim=-1))
            nut.write_root_velocity_to_sim(torch.zeros(scene.num_envs, 6, device=scene.device))

            push_dir_w = -outward_w
            force_tensor = (push_dir_w * force).unsqueeze(1)  # (num_envs, 1, 3)
            torque_tensor = torch.zeros_like(force_tensor)
            nut.permanent_wrench_composer.set_forces_and_torques(force_tensor, torque_tensor, is_global=True)

    def depth_to_visual(depth: torch.Tensor) -> np.ndarray:
        """Normalize a (H, W, 1) depth tensor to an 8-bit grayscale image for viewing."""
        depth_np = depth.squeeze(-1).cpu().numpy()
        lo, hi = np.percentile(depth_np, 1), np.percentile(depth_np, 99)
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        normalized = np.clip((depth_np - lo) / (hi - lo), 0.0, 1.0)
        return (normalized * 255).astype(np.uint8)

    def save_step(scene: InteractiveScene, nominal_depth: dict, output_dir: str, step: int):
        for tactile_name, _, _ in FINGER_TRIPLES:
            finger = tactile_name.split("_")[0]
            data = scene[tactile_name].data
            for env_i in range(min(scene.num_envs, 2)):
                rgb = data.tactile_rgb_image[env_i].cpu().numpy()
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, f"step{step:03d}_env{env_i}_{finger}_rgb.png"), bgr)

                depth = data.tactile_depth_image[env_i]
                cv2.imwrite(
                    os.path.join(output_dir, f"step{step:03d}_env{env_i}_{finger}_depth.png"),
                    depth_to_visual(depth),
                )

                deformation = (nominal_depth[finger][env_i, ..., 0] - depth[..., 0]).detach().cpu().numpy()
                deform_u8 = np.clip(deformation * args_cli.digit_depth_scale, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"step{step:03d}_env{env_i}_{finger}_deform.png"), deform_u8)

                print(
                    f"[INFO] step {step} env {env_i} {finger}: depth min/max = "
                    f"{depth.min().item():.5f}/{depth.max().item():.5f} m, "
                    f"deformation min/max = {deformation.min():.6f}/{deformation.max():.6f} m"
                )

    sim = sim_utils.SimulationContext(hora_env_cfg.sim)
    sim.set_camera_view(eye=[0.5, 0.0, 0.8], target=[0.0, -0.15, 0.5])

    scene = InteractiveScene(TacslNutPushSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing))

    sim.reset()
    print("[INFO]: Setup complete...")

    push_nuts_onto_elastomers(scene, standoff=args_cli.nut_standoff, force=args_cli.push_force)

    nominal_depth = {}
    for tactile_name, _, _ in FINGER_TRIPLES:
        finger = tactile_name.split("_")[0]
        baseline = scene[tactile_name].get_initial_render()
        nominal_depth[finger] = baseline["distance_to_image_plane"].clone()

    os.makedirs(args_cli.save_viz_dir, exist_ok=True)
    print(f"[INFO]: Saving tactile images to: {os.path.abspath(args_cli.save_viz_dir)}")

    save_step(scene, nominal_depth, args_cli.save_viz_dir, 0)

    max_deformation = {finger: 0.0 for finger in FINGER_NAMES}
    sim_dt = sim.get_physics_dt()
    with torch.inference_mode():
        for step in range(1, args_cli.steps + 1):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            for tactile_name, _, _ in FINGER_TRIPLES:
                finger = tactile_name.split("_")[0]
                depth = scene[tactile_name].data.tactile_depth_image[..., 0]
                deformation = (nominal_depth[finger][..., 0] - depth).max().item()
                max_deformation[finger] = max(max_deformation[finger], deformation)

            if step % args_cli.save_every == 0:
                save_step(scene, nominal_depth, args_cli.save_viz_dir, step)

    print("[INFO]: Done. Per-finger contact summary (guaranteed-contact nut push):")
    any_no_contact = False
    for finger in FINGER_NAMES:
        touched = max_deformation[finger] > args_cli.tactile_mask_eps
        print(f"  {finger:6s}: max_deformation={max_deformation[finger]:.6f} m {'(contact)' if touched else '(NO CONTACT)'}")
        any_no_contact = any_no_contact or not touched
    if any_no_contact:
        print(
            "[WARN]: at least one finger never registered contact above --tactile_mask_eps "
            f"({args_cli.tactile_mask_eps} m) even with a nut forced directly onto its pad -- "
            "this points at the sensor/USD wiring for that finger (elastomer geometry, camera "
            "placement, or push_nuts_onto_elastomers' outward-normal assumption), not the grasp "
            "cache or AllegroHandHoraEnv's task logic."
        )

    gc.collect()
    simulation_app.close()


if __name__ == "__main__":
    main()
