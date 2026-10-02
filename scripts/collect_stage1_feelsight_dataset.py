# --------------------------------------------------------
# Rolls out a trained HORA stage-1 (PPO, ground-truth privileged-info) policy on the
# `LeftAllegroHandDigitHora` task and records a NeuralFeels-"feelsight"-style visuotactile
# dataset: per-finger DIGIT tactile image/depth/mask, an external RGB-D + object-mask camera,
# and robot/object pose tracks -- see `scripts/dataset_collection/feelsight_writer.py` for the
# exact on-disk layout and `scripts/dataset_collection/sensors.py` for how the DIGIT tactile
# sensors (VisuoTactileSensor, attached to `assets/allegro/allegro_digit_left_elastomer.urdf`'s
# `link_*_tip_elastomer` bodies -- see `scripts/tools/tacsl_sensor_demo.py` and
# `inhand_rotation_env_cfg.py` in the sibling `inhand_rotation` repo, which this mirrors)
# and the scene camera are bolted onto `AllegroHandHoraEnv`, which has neither by default.
#
# This is the stage-1 counterpart of `collect_stage2_feelsight_dataset.py`: it loads a plain
# PPO checkpoint (`stage1_nn/best.pth`) instead of a ProprioAdapt one, and at inference feeds
# the policy the environment's ground-truth privileged info directly (`obs_dict["priv_info"]`)
# rather than routing through the proprioceptive-history adaptation module -- mirroring
# `ActorCritic._actor_critic`'s "Stage 1" branch (`hora/algo/models/models.py`) and
# `PPO.test()` (`hora/algo/ppo/ppo.py`). There is no `sa_mean_std`/`proprio_hist` normalizer
# involved, since stage-1 checkpoints don't have one.
#
# Mirrors `vis_s1.sh`'s stage-1 test invocation (`train.algo=PPO train.ppo.priv_info=True
# test=True`) but drives the env/policy directly instead of going through `train.py`'s hydra
# `main()`, so a custom per-step data-collection loop (and the extra camera/tactile sensors)
# can be added.
#
# Usage (run from the `allegro_inhand_rotation` repo root, matching `train.py`'s own
# convention -- IsaacLab's tiled cameras need `--enable_cameras` even when `--headless`):
#
#   ./isaaclab.sh -p scripts/collect_stage1_feelsight_dataset.py --enable_cameras --headless \
#       --checkpoint outputs/LeftAllegroHandDigitHora/baseline/stage1_nn/best.pth \
#       --num_episodes 5 --episode_steps 300 --output_dir data/feelsight_sim
#
# Also supports --task LeftAllegroHandDigitContactHora with a matching checkpoint.
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
# mirrors `dataset_collection.sensors.SCENE_CAMERA_VIEW_NAMES` -- duplicated here (rather than
# imported) since this module's arg parser runs before `AppLauncher`, and `sensors.py` pulls
# in `isaaclab`/`isaaclab_contrib`, which aren't safe to import until the sim app exists.
DEFAULT_CAMERA_VIEWS = ("front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right", "top-down")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="LeftAllegroHandDigitHora")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="defaults to outputs/<task>/baseline/stage1_nn/best.pth; stage-1 (PPO) checkpoint, e.g. outputs/<task>/<run>/stage1_nn/best.pth",
    )
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--episode_steps", type=int, default=300, help="control steps per episode (HORA default episodeLength=400)")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data/feelsight_sim"))
    parser.add_argument("--object_name", default=None, help="label for object.name / the output subfolder; defaults to task.env.object.type")
    parser.add_argument(
        "--camera_views", nargs="+", default=list(DEFAULT_CAMERA_VIEWS), choices=list(DEFAULT_CAMERA_VIEWS),
        help="external RGB-D scene cameras to record, each written to its own 'realsense/<view>' "
        "subfolder (mirrors feelsight's 'realsense/<camera_name>'); pass e.g. `--camera_views front` "
        "to record only the single original view",
    )
    parser.add_argument(
        "--scene_cam_eye", type=float, nargs=3, default=(0.35, -0.25, 0.85), metavar=("X", "Y", "Z"),
        help="eye position of the 'front' view; every other --camera_views entry orbits around "
        "--scene_cam_target at this same radius/height (see build_scene_camera_cfgs)",
    )
    parser.add_argument("--scene_cam_target", type=float, nargs=3, default=(0.0, 0.0, 0.5), metavar=("X", "Y", "Z"), help="default matches the hand's fixed base position")
    parser.add_argument("--scene_cam_width", type=int, default=640)
    parser.add_argument("--scene_cam_height", type=int, default=480)
    parser.add_argument("--tactile_width", type=int, default=320, help="pre-portrait width: saved tactile height (native render is 640x480)")
    parser.add_argument("--tactile_height", type=int, default=240)
    parser.add_argument("--digit_calib_dir", default=str(DEFAULT_DIGIT_CALIB_DIR), help="dir with bg.jpg + polycalib.npz + real_bg.npy (Taxim calibration)")
    parser.add_argument("--neuralfeels_urdf", default=str(REPO_ROOT.parent / "neuralfeels/data/assets/allegro/allegro_digit_left_ball.urdf"),
                        help="Read-only reference URDF used by the target NeuralFeels loader")
    parser.add_argument("--digit_depth_scale", type=float, default=33333.34, help="maximum 8-bit depth units per meter; reduced per episode if needed to avoid clipping")
    parser.add_argument("--tactile_mask_eps", type=float, default=3e-4, help="gel deformation (m) above which a tactile pixel counts as 'contact' in the mask frame")
    parser.add_argument("--gt_sdf_voxel_size", type=float, default=5e-4, help="voxel pitch (m) for the ground-truth object gt_sdf_voxel=<...>.npz, matches neuralfeels' default gt_voxel_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug_baseline_dir", default=str(REPO_ROOT / "outputs/collect_stage1_baseline_debug"),
        help="dump hand pose + per-finger tactile RGB/depth to this dir right when "
        "capture_initial_tactile_render() runs, so the no-contact baseline can be inspected "
        "visually; pass an empty string to skip",
    )
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

    from dataset_collection.feelsight_writer import EpisodeWriter
    from dataset_collection.neuralfeels_export import NeuralFeelsCapture
    from dataset_collection.gt_sdf import trimesh_from_shape_cfg, write_gt_sdf
    from dataset_collection.sensors import (
        ELASTOMER_LINKS,
        FINGER_NAMES,
        dataset_env_class,
        build_digit_render_cfg,
        build_scene_camera_cfgs,
    )

    set_seed(args_cli.seed)

    checkpoint_path = Path(
        args_cli.checkpoint or REPO_ROOT / "outputs" / args_cli.task / "baseline/stage1_nn/best.pth"
    ).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"stage1 checkpoint not found: {checkpoint_path}")

    # ---- compose the same Hydra config train.py would build for `vis_s1.sh`'s stage-1
    # test invocation, but through the compose API (no need for a hydra-clean argv here).
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        overrides = [
            f"task={args_cli.task}",
            f"headless={args_cli.headless}",
            f"sim_device={args_cli.device}",
            "test=True",
            "train.algo=PPO",
            "train.ppo.priv_info=True",
            "wandb.enabled=False",
            "task.env.numEnvs=1",
            "task.env.object.sampleProb=[1.0]",
            "task.env.randomization.randomizeMass=False",
            "task.env.randomization.randomizeCOM=False",
            "task.env.randomization.randomizeFriction=False",
            "task.env.randomization.randomizePDGains=False",
            f"checkpoint={checkpoint_path}",
        ]
        if args_cli.object_name is not None:
            overrides.append(f"task.env.object.type={args_cli.object_name}")
        overrides.extend(extra_overrides)
        cfg = hydra.compose(config_name="config", overrides=overrides)

    task_cfg_dict = omegaconf_to_dict(cfg.task)

    digit_render_cfg = build_digit_render_cfg(args_cli.digit_calib_dir)
    scene_camera_cfgs = build_scene_camera_cfgs(
        front_eye=args_cli.scene_cam_eye,
        target=args_cli.scene_cam_target,
        views=tuple(args_cli.camera_views),
        width=args_cli.scene_cam_width,
        height=args_cli.scene_cam_height,
    )

    env_cfg = build_hora_env_cfg(task_cfg_dict, cfg.sim_device, cfg.graphics_device_id, cfg.headless)
    # tag the hand so the scene segmentation can tell it apart from the (otherwise
    # unlabeled) background -- mirrors DatasetAllegroHandHoraEnv._build_object_cfg's own
    # ("class", "object") tag on the manipulated object.
    env_cfg.robot_cfg.spawn.semantic_tags = [("class", "robot")]
    env_cfg.sim.render_interval = 1
    raw_env = dataset_env_class(task_cfg_dict["name"])(
        env_cfg,
        render_mode=None if cfg.headless else "human",
        digit_render_cfg=digit_render_cfg,
        scene_camera_cfgs=scene_camera_cfgs,
    )
    env = HoraDirectEnvWrapper(raw_env, task_cfg_dict)

    # ---- build + load the stage-1 (PPO) policy -- exact same construction as
    # `hora/algo/ppo/ppo.py`'s `PPO.__init__`/`restore_test`. Unlike stage-2, there is no
    # `sa_mean_std` (proprio-history normalizer) to load, and inference feeds the policy
    # `priv_info` straight from the env instead of an adapted proprio history.
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

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    running_mean_std.load_state_dict(checkpoint["running_mean_std"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    running_mean_std.eval()

    object_name = args_cli.object_name or str(task_cfg_dict["env"]["object"]["type"])
    object_spawn_cfg = getattr(raw_env.object.cfg, "spawn", None)
    object_mesh = getattr(object_spawn_cfg, "usd_path", None) or getattr(object_spawn_cfg, "asset_path", None)
    # `numEnvs=1`, so the multi-asset spawner's single per-env shape cfg *is* the object.
    gt_mesh = trimesh_from_shape_cfg(object_spawn_cfg.assets_cfg[0])

    output_root = Path(args_cli.output_dir) / object_name
    output_root.mkdir(parents=True, exist_ok=True)

    control_dt = raw_env.step_dt
    body_ids = {f: raw_env.hand.find_bodies(ELASTOMER_LINKS[f])[0][0] for f in FINGER_NAMES}
    tactile_size = (args_cli.tactile_width, args_cli.tactile_height)

    print(f"[collect] task={args_cli.task} object={object_name} control_dt={control_dt:.4f}s output={output_root}")

    # No-contact tactile baseline is a fixed sensor property, not a per-episode one -- capture
    # it once here, before the *first* `env.reset()` ever samples a (contacting) grasp-cache
    # pose, rather than re-capturing it (contaminated by that episode's grasp contact) on
    # every reset. See `capture_initial_tactile_render`'s docstring.
    raw_env.capture_initial_tactile_render()
    capture = NeuralFeelsCapture(raw_env, tactile_size, args_cli.digit_depth_scale, args_cli.tactile_mask_eps, args_cli.neuralfeels_urdf)

    if args_cli.debug_baseline_dir:
        debug_dir = Path(args_cli.debug_baseline_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        base_pos = raw_env.hand.data.root_pos_w[0].detach().cpu().numpy()
        base_quat = raw_env.hand.data.root_quat_w[0].detach().cpu().numpy()
        joint_pos = raw_env.hand.data.joint_pos[0].detach().cpu().numpy()
        print(f"[debug] baseline hand base pos={base_pos} quat={base_quat}")
        print(f"[debug] baseline hand joint_pos={np.array2string(joint_pos, precision=4, suppress_small=True)}")

        for finger in FINGER_NAMES:
            fingertip_pos = raw_env.hand.data.body_pos_w[0, body_ids[finger]].detach().cpu().numpy()
            nominal_depth = raw_env.nominal_tactile_depth[finger][0, ..., 0].detach().cpu().numpy()
            print(
                f"[debug] baseline {finger}: fingertip world pos={fingertip_pos}, "
                f"nominal depth min/max={nominal_depth.min():.6f}/{nominal_depth.max():.6f} m"
            )

            rgb = np.clip(raw_env.tactile_sensors[finger].data.tactile_rgb_image[0].detach().cpu().numpy(), 0, 255).astype(np.uint8)
            Image.fromarray(rgb, mode="RGB").save(debug_dir / f"baseline_{finger}_rgb.png")

            depth_range = max(nominal_depth.max() - nominal_depth.min(), 1e-9)
            depth_u8 = np.clip((nominal_depth - nominal_depth.min()) / depth_range * 255, 0, 255).astype(np.uint8)
            Image.fromarray(depth_u8, mode="L").save(debug_dir / f"baseline_{finger}_depth.png")

        # -- a wide shot of the whole hand from each external scene camera, so the hand's
        # actual pose at capture time (not just each fingertip's close-up tactile view) can
        # be seen -- same camera/format the main per-step loop below saves.
        for view_name, scene_cam in raw_env.scene_cams.items():
            scene_rgb = scene_cam.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
            Image.fromarray(scene_rgb, mode="RGB").save(debug_dir / f"baseline_scene_rgb_{view_name.replace('-', '_')}.png")

        print(f"[debug] saved baseline hand pose + per-finger RGB/depth to {debug_dir}")

    for ep in range(args_cli.num_episodes):
        writer = EpisodeWriter(output_root, ep, camera_names=args_cli.camera_views, joint_names=capture.joint_names)
        obs_dict = env.reset()

        for step in range(args_cli.episode_steps):
            with torch.no_grad():
                input_dict = {
                    "obs": running_mean_std(obs_dict["obs"]),
                    "priv_info": obs_dict["priv_info"],
                }
                mu = model.act_inference(input_dict)
                mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, reward, dones, info = env.step(mu)

            # DirectRLEnv has already reset on done: never mix the new grasp into
            # this episode, or pair reset poses with a previous rendered image.
            if bool(dones[0]):
                print(f"[collect] episode {ep}: stopped before auto-reset frame at step {step}")
                break

            writer.add_step(t=step * control_dt, **capture.capture())

        if not writer.frame_idx:
            print(f"[collect] episode {ep}: no valid pre-reset frames; skipping")
            continue
        writer.finalize(object_name=object_name, object_mesh=object_mesh,
                        digit_info=capture.digit_info, realsense_intrinsics=capture.scene_intrinsics)
        gt_sdf_path = write_gt_sdf(writer.dir, gt_mesh, voxel_size=args_cli.gt_sdf_voxel_size)
        print(f"[collect] episode {ep}: wrote {writer.frame_idx} frames to {writer.dir} (+ {gt_sdf_path.name})")

    raw_env.close()
    gc.collect()
    simulation_app.close()


if __name__ == "__main__":
    main()
