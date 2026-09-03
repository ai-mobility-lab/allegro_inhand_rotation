# --------------------------------------------------------
# Rolls out a trained HORA stage-2 (ProprioAdapt) policy on the `LeftAllegroHandDigitHora`
# task and records a NeuralFeels-"feelsight"-style visuotactile dataset: per-finger DIGIT
# tactile image/depth/mask, an external RGB-D + object-mask camera, and robot/object pose
# tracks -- see `scripts/dataset_collection/feelsight_writer.py` for the exact on-disk
# layout and `scripts/dataset_collection/sensors.py` for how the DIGIT tactile sensors
# (VisuoTactileSensor, attached to `assets/allegro/allegro_digit_left_elastomer.urdf`'s
# `link_*_tip_elastomer` bodies -- see `scripts/tools/tacsl_sensor_demo.py` and
# `inhand_rotation_env_cfg.py` in the sibling `inhand_rotation` repo, which this mirrors)
# and the scene camera are bolted onto `AllegroHandHoraEnv`, which has neither by default.
#
# Mirrors `vis_s2.sh`'s stage-2 test invocation (`train.algo=ProprioAdapt
# train.ppo.priv_info=True train.ppo.proprio_adapt=True test=True`) but drives the env/
# policy directly instead of going through `train.py`'s hydra `main()`, so a custom
# per-step data-collection loop (and the extra camera/tactile sensors) can be added.
#
# Usage (run from the `allegro_inhand_rotation` repo root, matching `train.py`'s own
# convention -- IsaacLab's tiled cameras need `--enable_cameras` even when `--headless`):
#
#   ./isaaclab.sh -p scripts/collect_stage2_feelsight_dataset.py --enable_cameras --headless \
#       --checkpoint outputs/LeftAllegroHandDigitHora/baseline/stage2_nn/best.pth \
#       --num_episodes 5 --episode_steps 300 --output_dir data/feelsight_sim
#
# Extra Hydra-style overrides (e.g. to change the manipulated object) can be appended
# after a bare `--`, e.g. `-- task.env.object.type=cylinder_default`.
# --------------------------------------------------------

import argparse
import sys
from pathlib import Path
import gc
from omegaconf import DictConfig, OmegaConf

from isaaclab.app import AppLauncher

# OmegaConf & Hydra Config
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_envs
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIGIT_CALIB_DIR = REPO_ROOT / "assets/digit_data"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="LeftAllegroHandDigitHora")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "outputs/LeftAllegroHandDigitHora/baseline/stage2_nn/best.pth"),
        help="stage-2 (ProprioAdapt) checkpoint, e.g. outputs/<task>/<run>/stage2_nn/best.pth",
    )
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--episode_steps", type=int, default=300, help="control steps per episode (HORA default episodeLength=400)")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data/feelsight_sim"))
    parser.add_argument("--object_name", default=None, help="label for object.name / the output subfolder; defaults to task.env.object.type")
    parser.add_argument("--camera_name", default="front", help="name of the external RGB-D camera, mirrors feelsight's 'realsense/<camera_name>'")
    parser.add_argument("--scene_cam_eye", type=float, nargs=3, default=(0.55, -0.45, 0.85), metavar=("X", "Y", "Z"))
    parser.add_argument("--scene_cam_target", type=float, nargs=3, default=(0.0, 0.0, 0.5), metavar=("X", "Y", "Z"), help="default matches the hand's fixed base position")
    parser.add_argument("--scene_cam_width", type=int, default=640)
    parser.add_argument("--scene_cam_height", type=int, default=480)
    parser.add_argument("--tactile_width", type=int, default=320, help="tactile images are rendered at the DIGIT's native 640x480 and resized to this for saving")
    parser.add_argument("--tactile_height", type=int, default=240)
    parser.add_argument("--digit_calib_dir", default=str(DEFAULT_DIGIT_CALIB_DIR), help="dir with bg.jpg + polycalib.npz + real_bg.npy (Taxim calibration)")
    parser.add_argument("--digit_depth_scale", type=float, default=33333.34, help="gel-deformation-meters -> 8-bit pixel scale for the saved tactile depth/mask frames")
    parser.add_argument("--tactile_mask_eps", type=float, default=3e-4, help="gel deformation (m) above which a tactile pixel counts as 'contact' in the mask frame")
    parser.add_argument("--gt_sdf_voxel_size", type=float, default=5e-4, help="voxel pitch (m) for the ground-truth object gt_sdf_voxel=<...>.npz, matches neuralfeels' default gt_voxel_size")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        extra_overrides = argv[split + 1 :]
        argv = argv[:split]
    else:
        extra_overrides = []

    parser = build_arg_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(argv)

    if not args_cli.enable_cameras:
        raise SystemExit(
            "This script needs Isaac Sim's offscreen render pipeline for the DIGIT tactile "
            "cameras and the external RGB-D camera -- rerun with --enable_cameras."
        )

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # ---- everything below may safely import isaaclab / hora.tasks ----
    sys.path.insert(0, str(REPO_ROOT))  # so `import hora...` resolves regardless of cwd
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import dataset_collection...` resolves

    import hydra
    import numpy as np
    import torch
    from PIL import Image

    from hora.algo.models.models import ActorCritic
    from hora.algo.models.running_mean_std import RunningMeanStd
    from hora.tasks.isaaclab.allegro_hand_hora_env import build_hora_env_cfg
    from hora.tasks.isaaclab.wrapper import HoraDirectEnvWrapper
    from hora.utils.misc import set_seed
    from hora.utils.reformat import omegaconf_to_dict

    from dataset_collection.feelsight_writer import EpisodeWriter, scene_segmentation_image
    from dataset_collection.gt_sdf import trimesh_from_shape_cfg, write_gt_sdf
    from dataset_collection.sensors import (
        DIGIT_HORIZONTAL_APERTURE_MM,
        DIGIT_VERTICAL_APERTURE_MM,
        DIGIT_FOCAL_LENGTH_MM,
        DIGIT_WORKING_DISTANCE_MM,
        ELASTOMER_LINKS,
        FINGER_NAMES,
        DatasetAllegroHandHoraEnv,
        build_digit_render_cfg,
        build_scene_camera_cfg,
        pinhole_intrinsics,
        pose_to_matrix,
    )

    set_seed(args_cli.seed)

    checkpoint_path = Path(args_cli.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"stage2 checkpoint not found: {checkpoint_path}")

    # ---- compose the same Hydra config train.py would build for `vis_s2.sh`'s stage-2
    # test invocation, but through the compose API (no need for a hydra-clean argv here).
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        overrides = [
            f"task={args_cli.task}",
            f"headless={args_cli.headless}",
            f"sim_device={args_cli.device}",
            "test=True",
            "train.algo=ProprioAdapt",
            "train.ppo.priv_info=True",
            "train.ppo.proprio_adapt=True",
            "wandb.enabled=False",
            "task.env.numEnvs=1",
            f"task.env.object.type={args_cli.object_name}",
            "task.env.object.sampleProb=[1.0]",
            f"checkpoint={checkpoint_path}",
        ] + extra_overrides
        cfg = hydra.compose(config_name="config", overrides=overrides)

    task_cfg_dict = omegaconf_to_dict(cfg.task)

    digit_render_cfg = build_digit_render_cfg(args_cli.digit_calib_dir)
    scene_camera_cfg = build_scene_camera_cfg(
        eye=args_cli.scene_cam_eye,
        target=args_cli.scene_cam_target,
        width=args_cli.scene_cam_width,
        height=args_cli.scene_cam_height,
    )

    env_cfg = build_hora_env_cfg(task_cfg_dict, cfg.sim_device, cfg.graphics_device_id, cfg.headless)
    # tag the hand so the scene segmentation can tell it apart from the (otherwise
    # unlabeled) background -- mirrors DatasetAllegroHandHoraEnv._build_object_cfg's own
    # ("class", "object") tag on the manipulated object.
    env_cfg.robot_cfg.spawn.semantic_tags = [("class", "robot")]
    raw_env = DatasetAllegroHandHoraEnv(
        env_cfg,
        render_mode=None if cfg.headless else "human",
        digit_render_cfg=digit_render_cfg,
        scene_camera_cfg=scene_camera_cfg,
    )
    env = HoraDirectEnvWrapper(raw_env, task_cfg_dict)

    # ---- build + load the stage-2 (ProprioAdapt) policy -- exact same construction as
    # `hora/algo/padapt/padapt.py`'s `ProprioAdapt.__init__`/`restore_test`.
    device = cfg.rl_device
    net_config = {
        "actor_units": cfg.train.network.mlp.units,
        "priv_mlp_units": cfg.train.network.priv_mlp.units,
        "actions_num": env.action_space.shape[0],
        "input_shape": env.observation_space.shape,
        "priv_info": cfg.train.ppo["priv_info"],
        "proprio_adapt": cfg.train.ppo["proprio_adapt"],
        "priv_info_dim": cfg.train.ppo["priv_info_dim"],
    }
    model = ActorCritic(net_config).to(device)
    running_mean_std = RunningMeanStd(env.observation_space.shape).to(device)
    sa_mean_std = RunningMeanStd((env.prop_hist_len, 32)).to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    running_mean_std.load_state_dict(checkpoint["running_mean_std"])
    model.load_state_dict(checkpoint["model"])
    sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
    model.eval()
    running_mean_std.eval()
    sa_mean_std.eval()

    object_name = args_cli.object_name or str(task_cfg_dict["env"]["object"]["type"])
    object_spawn_cfg = getattr(raw_env.object.cfg, "spawn", None)
    object_mesh = getattr(object_spawn_cfg, "usd_path", None) or getattr(object_spawn_cfg, "asset_path", None)
    # `numEnvs=1`, so the multi-asset spawner's single per-env shape cfg *is* the object.
    gt_mesh = trimesh_from_shape_cfg(object_spawn_cfg.assets_cfg[0])

    output_root = Path(args_cli.output_dir) / object_name
    output_root.mkdir(parents=True, exist_ok=True)

    scene_intrinsics = pinhole_intrinsics(
        args_cli.scene_cam_width, args_cli.scene_cam_height, 24.0, 20.955, 20.955 * args_cli.scene_cam_height / args_cli.scene_cam_width
    )
    digit_intrinsics = pinhole_intrinsics(
        args_cli.tactile_width, args_cli.tactile_height, DIGIT_FOCAL_LENGTH_MM, DIGIT_HORIZONTAL_APERTURE_MM, DIGIT_VERTICAL_APERTURE_MM
    )
    digit_info = {
        "depth_scale": args_cli.digit_depth_scale,
        "cam_dist": -DIGIT_WORKING_DISTANCE_MM / 1000.0,
        "intrinsics": digit_intrinsics,
    }

    control_dt = raw_env.step_dt
    body_ids = {f: raw_env.hand.find_bodies(ELASTOMER_LINKS[f])[0][0] for f in FINGER_NAMES}
    tactile_size = (args_cli.tactile_width, args_cli.tactile_height)

    print(f"[collect] task={args_cli.task} object={object_name} control_dt={control_dt:.4f}s output={output_root}")

    for ep in range(args_cli.num_episodes):
        writer = EpisodeWriter(output_root, ep, camera_name=args_cli.camera_name)
        obs_dict = env.reset()
        raw_env.capture_initial_tactile_render()

        base_pos = raw_env.hand.data.root_pos_w[0].detach().cpu().numpy()
        base_quat = raw_env.hand.data.root_quat_w[0].detach().cpu().numpy()
        base_pose = pose_to_matrix(base_pos, base_quat)

        for step in range(args_cli.episode_steps):
            with torch.no_grad():
                input_dict = {
                    "obs": running_mean_std(obs_dict["obs"]),
                    "proprio_hist": sa_mean_std(obs_dict["proprio_hist"].detach()),
                }
                mu = model.act_inference(input_dict)
                mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, reward, dones, info = env.step(mu)

            # -- object / hand pose --
            obj_pos = raw_env.object.data.root_pos_w[0].detach().cpu().numpy()
            obj_quat = raw_env.object.data.root_quat_w[0].detach().cpu().numpy()
            object_pose = pose_to_matrix(obj_pos, obj_quat)
            joint_state = raw_env.hand.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            finger_poses = np.stack(
                [
                    pose_to_matrix(
                        raw_env.hand.data.body_pos_w[0, body_ids[f]].detach().cpu().numpy(),
                        raw_env.hand.data.body_quat_w[0, body_ids[f]].detach().cpu().numpy(),
                    )
                    for f in FINGER_NAMES
                ],
                axis=0,
            )

            # -- DIGIT tactile: image (Taxim RGB), depth + mask (gel deformation) --
            tactile = {}
            for finger in FINGER_NAMES:
                sensor_data = raw_env.tactile_sensors[finger].data
                rgb = np.clip(sensor_data.tactile_rgb_image[0].detach().cpu().numpy(), 0, 255).astype(np.uint8)
                depth = sensor_data.tactile_depth_image[0, ..., 0].detach().cpu().numpy()
                nominal_depth = raw_env.nominal_tactile_depth[finger][0, ..., 0].detach().cpu().numpy()
                deformation = nominal_depth - depth  # positive = gel pushed in
                depth_u8 = np.clip(deformation * args_cli.digit_depth_scale, 0, 255).astype(np.uint8)
                mask_u8 = np.where(deformation > args_cli.tactile_mask_eps, np.uint8(255), np.uint8(0))
                img = np.array(Image.fromarray(rgb, mode="RGB").resize(tactile_size, Image.BILINEAR))
                depth_img = np.array(Image.fromarray(depth_u8, mode="L").resize(tactile_size, Image.BILINEAR))
                mask_img = np.array(Image.fromarray(mask_u8, mode="L").resize(tactile_size, Image.NEAREST))
                tactile[finger] = (img, depth_img, mask_img)

            # -- external RGB-D + segmentation camera --
            scene_data = raw_env.scene_cam.data
            scene_rgb = scene_data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
            scene_depth = scene_data.output["distance_to_image_plane"][0, ..., 0].detach().cpu().numpy()
            seg_ids = scene_data.output["instance_segmentation_fast"][0].detach().cpu().numpy()
            seg_info = scene_data.info.get("instance_segmentation_fast") if scene_data.info else None
            scene_seg = scene_segmentation_image(seg_ids, seg_info)
            cam_pos = scene_data.pos_w[0].detach().cpu().numpy()
            cam_quat = scene_data.quat_w_world[0].detach().cpu().numpy()
            scene_cam_pose = pose_to_matrix(cam_pos, cam_quat)

            writer.add_step(
                t=step * control_dt,
                object_pose=object_pose,
                finger_poses=finger_poses,
                joint_state=joint_state,
                base_pose=base_pose,
                tactile=tactile,
                scene_rgb=scene_rgb,
                scene_depth=scene_depth,
                scene_seg=scene_seg,
                scene_cam_pose=scene_cam_pose,
            )

            if bool(dones[0]):
                print(f"[collect] episode {ep}: env reset early at control step {step}")
                break

        writer.finalize(object_name=object_name, object_mesh=object_mesh, digit_info=digit_info, realsense_intrinsics=scene_intrinsics)
        gt_sdf_path = write_gt_sdf(writer.dir, gt_mesh, voxel_size=args_cli.gt_sdf_voxel_size)
        print(f"[collect] episode {ep}: wrote {writer.frame_idx} frames to {writer.dir} (+ {gt_sdf_path.name})")

    raw_env.close()
    gc.collect()
    simulation_app.close()


if __name__ == "__main__":
    main()
