# --------------------------------------------------------
# Standalone smoke test for the DIGIT `VisuoTactileSensor`s wired onto `DatasetAllegroHandHoraEnv`
# -- the exact sensor rig `scripts/collect_stage1_feelsight_dataset.py` /
# `collect_stage2_feelsight_dataset.py` use (see `scripts/dataset_collection/sensors.py`).
#
# Mirrors the sibling `inhand_rotation` repo's own `scripts/tools/tacsl_sensor_demo.py` in intent
# -- exercise the tactile sensors directly, without a trained policy checkpoint or the full
# feelsight-dataset writer, so sensor wiring can be re-checked in isolation right after
# regenerating the hand USD or DIGIT calibration data, before it's exercised through the full
# stage-1/stage-2 collection scripts.
#
# Unlike that reference demo (a bare `InteractiveScene` built from an `InteractiveSceneCfg`, with
# one independently-pushed "nut" per fingertip guaranteeing contact regardless of hand pose), this
# project's `AllegroHandHoraEnv` builds its scene by hand rather than through `InteractiveSceneCfg`
# field reflection (`Articulation`/`RigidObject` constructed directly in `_setup_scene`, see
# `hora/tasks/isaaclab/allegro_hand_hora_env.py`), so there is no equivalent lightweight scene to
# assemble standalone here. Instead, this script drives the real `DatasetAllegroHandHoraEnv`,
# first calling `capture_initial_tactile_render()` -- while the hand/object still sit at their
# raw spawn defaults (fingers open, not yet any grasp-cache contact -- see that method's own
# docstring for why this timing matters) -- to record each sensor's true no-contact depth
# baseline, then `env.reset()`, which samples a pre-formed grasp from the task's own grasp cache,
# typically already touching the object at every fingertip. It then holds that grasp (zero
# actions, so `_pre_physics_step` just re-commands the same joint targets every step) while
# saving each finger's tactile RGB/depth and the gel deformation relative to that true no-contact
# baseline, computed exactly as `collect_stage1_feelsight_dataset.py` does. No trained checkpoint
# is required.
#
# Requires `--enable_cameras` (the DIGIT sensors are tiled cameras) and real DIGIT Taxim
# calibration data (see `--digit_calib_dir`, default `assets/digit_data/`) -- without it,
# `enable_camera_tactile=True` sensor init raises `FileNotFoundError`.
#
# Usage (run from the `allegro_inhand_rotation` repo root):
#
#   ./isaaclab.sh -p scripts/tools/tacsl_sensor_demo.py --enable_cameras --headless \
#       --steps 100 --save_every 20
#
# Extra Hydra-style overrides (e.g. to change the manipulated object) can be appended after a
# bare `--`, e.g. `-- task.env.object.type=cuboid_default`.
# --------------------------------------------------------

import argparse
import gc
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

# Same resolvers `train.py`/`collect_stage1_feelsight_dataset.py` register before composing any
# Hydra config under `configs/` -- required for `hydra.compose` below to resolve them.
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)

# this file lives at scripts/tools/, two levels below the repo root (collect_stage1_feelsight_
# dataset.py's own REPO_ROOT/DEFAULT_DIGIT_CALIB_DIR sit one level up from there).
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DIGIT_CALIB_DIR = REPO_ROOT / "assets/digit_data"
print(DEFAULT_DIGIT_CALIB_DIR)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="LeftAllegroHandDigitHora")
    parser.add_argument(
        "--object_name", default=None,
        help="task.env.object.type override, e.g. cuboid_default; defaults to the task config's own object.type "
        "(needs a matching grasp cache under cache/<type>/ -- see AllegroHandHoraEnv.__init__)",
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of control steps to hold the grasp for.")
    parser.add_argument("--save_every", type=int, default=20, help="Save tactile images every N control steps (and at step 0).")
    parser.add_argument("--save_viz_dir", default=str(REPO_ROOT / "outputs/tacsl_sensor_demo"), help="Directory to save PNGs into.")
    parser.add_argument("--digit_calib_dir", default=str(DEFAULT_DIGIT_CALIB_DIR), help="dir with bg.jpg + polycalib.npz + real_bg.npy (Taxim calibration)")
    parser.add_argument("--digit_depth_scale", type=float, default=33333.34, help="gel-deformation-meters -> 8-bit pixel scale for the saved deformation/mask frames")
    parser.add_argument("--tactile_mask_eps", type=float, default=3e-4, help="gel deformation (m) above which a tactile pixel counts as 'contact'")
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

    from hora.tasks.isaaclab.allegro_hand_hora_env import build_hora_env_cfg
    from hora.tasks.isaaclab.wrapper import HoraDirectEnvWrapper
    from hora.utils.misc import set_seed
    from hora.utils.reformat import omegaconf_to_dict

    from dataset_collection.sensors import (
        FINGER_NAMES,
        DatasetAllegroHandHoraEnv,
        build_digit_render_cfg,
    )

    set_seed(args_cli.seed)

    # ---- compose the same task/sim config `collect_stage1_feelsight_dataset.py` would, minus
    # the train/checkpoint-only overrides (no policy is loaded here).
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        overrides = [
            f"task={args_cli.task}",
            f"headless={args_cli.headless}",
            f"sim_device={args_cli.device}",
            "wandb.enabled=False",
            "task.env.numEnvs=1",
        ]
        if args_cli.object_name is not None:
            overrides += [f"task.env.object.type={args_cli.object_name}", "task.env.object.sampleProb=[1.0]"]
        overrides += extra_overrides
        cfg = hydra.compose(config_name="config", overrides=overrides)

    task_cfg_dict = omegaconf_to_dict(cfg.task)

    digit_render_cfg = build_digit_render_cfg(args_cli.digit_calib_dir)
    env_cfg = build_hora_env_cfg(task_cfg_dict, cfg.sim_device, cfg.graphics_device_id, cfg.headless)

    raw_env = DatasetAllegroHandHoraEnv(
        env_cfg,
        render_mode=None if cfg.headless else "human",
        digit_render_cfg=digit_render_cfg,
        scene_camera_cfg=None,
    )
    env = HoraDirectEnvWrapper(raw_env, task_cfg_dict)

    object_name = args_cli.object_name or str(task_cfg_dict["env"]["object"]["type"])
    print(f"[INFO]: task={args_cli.task} object={object_name}")

    raw_env.capture_initial_tactile_render()
    env.reset()
    print("[INFO]: Setup complete, true no-contact baseline captured, initial grasp sampled from grasp cache...")

    os.makedirs(args_cli.save_viz_dir, exist_ok=True)
    print(f"[INFO]: Saving tactile images to: {os.path.abspath(args_cli.save_viz_dir)}")

    def finger_deformation(finger: str) -> torch.Tensor:
        """(gel-deformation-meters, H, W) for env 0 -- positive where the pad is pushed in,
        relative to the true no-contact rest baseline `capture_initial_tactile_render` set."""
        depth = raw_env.tactile_sensors[finger].data.tactile_depth_image[0, ..., 0]
        nominal = raw_env.nominal_tactile_depth[finger][0, ..., 0]
        return nominal - depth

    def save_step(step: int):
        for finger in FINGER_NAMES:
            sensor_data = raw_env.tactile_sensors[finger].data
            rgb = sensor_data.tactile_rgb_image[0].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args_cli.save_viz_dir, f"step{step:03d}_{finger}_rgb.png"), bgr)

            deformation = finger_deformation(finger).detach().cpu().numpy()
            deform_u8 = np.clip(deformation * args_cli.digit_depth_scale, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args_cli.save_viz_dir, f"step{step:03d}_{finger}_deform.png"), deform_u8)

            print(
                f"[INFO] step {step} {finger}: deformation min/max = "
                f"{deformation.min():.6f}/{deformation.max():.6f} m"
            )

    # step 0's deformation already reflects the grasp cache's initial contact (the baseline
    # is now genuinely no-contact, not contaminated by that contact -- see
    # `capture_initial_tactile_render`'s docstring), so seed `max_deformation` from it instead
    # of starting from 0.
    max_deformation = {finger: finger_deformation(finger).max().item() for finger in FINGER_NAMES}
    save_step(0)

    zero_actions = env.zero_actions()
    for step in range(1, args_cli.steps + 1):
        _, _, dones, _ = env.step(zero_actions)

        for finger in FINGER_NAMES:
            max_deformation[finger] = max(max_deformation[finger], finger_deformation(finger).max().item())

        if step % args_cli.save_every == 0:
            save_step(step)

        if bool(dones[0]):
            print(f"[WARN]: env reset early at control step {step} (object fell?) -- stopping.")
            break

    print("[INFO]: Done. Per-finger contact summary (max gel deformation vs. the true no-contact baseline):")
    any_no_contact = False
    for finger in FINGER_NAMES:
        touched = max_deformation[finger] > args_cli.tactile_mask_eps
        print(
            f"  {finger:6s}: max_deformation={max_deformation[finger]:.6f} m "
            f"{'(contact)' if touched else '(NO CONTACT)'}"
        )
        any_no_contact = any_no_contact or not touched
    if any_no_contact:
        print(
            "[WARN]: at least one finger never registered contact above --tactile_mask_eps "
            f"({args_cli.tactile_mask_eps} m) -- check the grasp cache / elastomer geometry / "
            "sensor wiring for that finger."
        )

    raw_env.close()
    gc.collect()
    simulation_app.close()


if __name__ == "__main__":
    main()
