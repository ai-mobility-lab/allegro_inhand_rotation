# --------------------------------------------------------
# Interactive tool for hand-tuning the four DIGIT tactile-camera extrinsics wired onto
# `DatasetAllegroHandHoraEnv` (see `scripts/dataset_collection/sensors.py`'s
# `DIGIT_CAMERA_OFFSET_POS`/`DIGIT_CAMERA_OFFSET_ROT`) and for freely repositioning the
# task's own sample object in front of the fingertips, so both can be checked against the
# live tactile RGB feed instead of guessing from geometry alone.
#
# Spawns the real `DatasetAllegroHandHoraEnv` (same sensor rig `tacsl_sensor_demo.py`
# exercises) and opens an `omni.ui` window with:
#   - one collapsible per finger with sliders for that DIGIT camera's local position/
#     rotation offset from the fingertip housing (on top of the shared
#     `DIGIT_CAMERA_OFFSET_POS`/`_ROT` baseline every finger starts from),
#   - sliders for the sample object's position/rotation, relative to wherever
#     `env.reset()`'s sampled grasp first placed it,
#   - a live-updating tactile RGB preview per finger (a small `ByteImageProvider`-backed
#     widget -- see `LiveImageView` below, since `isaaclab.ui.widgets.ImagePlot` is broken
#     in this IsaacLab checkout),
#   - "Print / Save" buttons that dump the current camera offsets (as a ready-to-paste
#     snippet for `sensors.py`) and/or the four current tactile RGB/depth frames to disk.
#
# The hand is held frozen at `env.reset()`'s sampled grasp pose the whole time (joint
# state/target rewritten every step, bypassing the actuators -- same trick
# `tune_grasp_joint_pos.py` in the sibling `inhand_rotation` repo uses for its own
# GRASP_JOINT_POS sliders) so the only things moving are the object and the cameras you
# are actively tuning. `env.step()` (and the reward/done/auto-reset machinery that comes
# with it) is never called -- the sim is stepped directly, so a "falling" object never
# triggers an env reset out from under you.
#
# Requires a GUI session (no --headless) and --enable_cameras (the DIGIT sensors are
# tiled cameras), plus real DIGIT Taxim calibration data (see --digit_calib_dir, default
# assets/digit_data/).
#
# Run from the allegro_inhand_rotation repo root, e.g.:
#
#   bash ~/lib/IsaacLab/isaaclab.sh -p scripts/tools/tune_tactile_camera_pose.py --enable_cameras
#   bash ~/lib/IsaacLab/isaaclab.sh -p scripts/tools/tune_tactile_camera_pose.py --enable_cameras \
#       -- task.env.object.type=cuboid_default
#
# Extra Hydra-style overrides (e.g. to change the manipulated object type) can be
# appended after a bare `--`, exactly as in tacsl_sensor_demo.py.
# --------------------------------------------------------

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

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
    parser.add_argument("--task", default="LeftAllegroHandDigitHora")
    parser.add_argument(
        "--object_name", default=None,
        help="task.env.object.type override, e.g. cuboid_default; defaults to the task config's own object.type "
        "(needs a matching grasp cache under cache/<type>/ -- see AllegroHandHoraEnv.__init__).",
    )
    parser.add_argument("--digit_calib_dir", default=str(DEFAULT_DIGIT_CALIB_DIR), help="dir with bg.jpg + polycalib.npz + real_bg.npy (Taxim calibration)")
    parser.add_argument("--object_pos_range", type=float, default=0.05, help="+/- range (m) of the object position sliders, around its reset pose.")
    parser.add_argument("--cam_pos_range", type=float, default=0.01, help="+/- range (m) of each camera's local position-offset sliders.")
    parser.add_argument("--cam_rot_range_deg", type=float, default=45.0, help="+/- range (deg) of each camera's local rotation-offset sliders.")
    parser.add_argument("--save_dir", default=str(REPO_ROOT / "outputs/tune_tactile_camera_pose"), help="Directory the save buttons write into.")
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

    if args_cli.headless:
        raise ValueError("tune_tactile_camera_pose.py needs a GUI window to be interactive -- do not pass --headless.")
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

    import os
    import time

    import cv2
    import hydra
    import numpy as np
    import omni.ui as ui
    import torch
    from isaacsim.gui.components.ui_utils import btn_builder, combo_floatfield_slider_builder

    from isaaclab.utils.math import convert_camera_frame_orientation_convention, quat_from_euler_xyz, quat_mul

    from hora.tasks.isaaclab.allegro_hand_hora_env import build_hora_env_cfg
    from hora.tasks.isaaclab.wrapper import HoraDirectEnvWrapper
    from hora.utils.misc import set_seed
    from hora.utils.reformat import omegaconf_to_dict

    from dataset_collection.sensors import (
        DIGIT_CAMERA_OFFSET_POS,
        DIGIT_CAMERA_OFFSET_ROT,
        FINGER_NAMES,
        HOUSING_LINKS,
        DatasetAllegroHandHoraEnv,
        build_digit_render_cfg,
    )

    class LiveImageView:
        """Minimal `ByteImageProvider`-backed live RGB preview.

        `isaaclab.ui.widgets.ImagePlot` would be the natural fit here, but in this IsaacLab
        checkout its `__init__` never sets `_show_min_max`/`_enabled_min_max`/`_min_value`/
        `_max_value`, while `_build_widget`/`_build_mode_frame` read them unconditionally --
        it raises `AttributeError` the instant the mode-selector sub-frame builds. This
        reimplements just the byte-provider display path `ImagePlot.update_image` actually
        needs, skipping that broken mode-dropdown frame entirely.
        """

        def __init__(self, label: str, image: np.ndarray, widget_height: int = 200):
            height, width = image.shape[:2]
            self._provider = ui.ByteImageProvider()
            with ui.VStack(spacing=3):
                ui.Label(label)
                with ui.Frame(width=(width / height) * widget_height, height=widget_height):
                    ui.ImageWithProvider(self._provider)
            self.update_image(image)

        def update_image(self, image: np.ndarray):
            height, width = image.shape[:2]
            if image.ndim == 3 and image.shape[2] == 3:
                image = np.dstack((image, np.full((height, width, 1), 255, dtype=np.uint8)))
            self._provider.set_bytes_data(image.flatten().data, [width, height])

    set_seed(args_cli.seed)

    # ---- compose the same task/sim config tacsl_sensor_demo.py would, minus anything
    # train/checkpoint-only (no policy is loaded here either).
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        overrides = [
            f"task={args_cli.task}",
            "headless=False",
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
    env_cfg = build_hora_env_cfg(task_cfg_dict, cfg.sim_device, cfg.graphics_device_id, headless=False)

    raw_env = DatasetAllegroHandHoraEnv(
        env_cfg, render_mode="human", digit_render_cfg=digit_render_cfg, scene_camera_cfgs=None,
    )
    env = HoraDirectEnvWrapper(raw_env, task_cfg_dict)

    object_name = args_cli.object_name or str(task_cfg_dict["env"]["object"]["type"])
    print(f"[INFO]: task={args_cli.task} object={object_name}")

    raw_env.capture_initial_tactile_render()
    env.reset()
    print("[INFO]: Setup complete, initial grasp sampled from grasp cache -- hand is now held fixed there.")

    device = raw_env.device

    # freeze the hand exactly where `env.reset()` put it -- rewritten every step below,
    # bypassing the actuators entirely (position targets alone would still let a
    # torque-controlled hand drift), same trick `tune_grasp_joint_pos.py` uses.
    hold_joint_pos = raw_env.hand.data.joint_pos.clone()
    hold_joint_vel = torch.zeros_like(hold_joint_pos)

    # the object's reset-sampled grasp pose is the (0, 0, 0) origin the sliders below
    # nudge it away from/back to.
    object_start_pos = raw_env.object.data.root_pos_w.clone()
    object_start_quat = raw_env.object.data.root_quat_w.clone()

    base_cam_pos = torch.tensor(DIGIT_CAMERA_OFFSET_POS, dtype=torch.float32, device=device).unsqueeze(0)
    base_cam_quat = torch.tensor(DIGIT_CAMERA_OFFSET_ROT, dtype=torch.float32, device=device).unsqueeze(0)

    object_pos_models: dict[str, "ui.AbstractValueModel"] = {}
    object_rot_models: dict[str, "ui.AbstractValueModel"] = {}
    cam_pos_models: dict[str, dict[str, "ui.AbstractValueModel"]] = {f: {} for f in FINGER_NAMES}
    cam_rot_models: dict[str, dict[str, "ui.AbstractValueModel"]] = {f: {} for f in FINGER_NAMES}
    image_plots: dict[str, LiveImageView] = {}

    def build_object_section():
        ui.Label(
            "Offsets are relative to the pose env.reset()'s sampled grasp put the object at.",
            word_wrap=True, height=30,
        )
        with ui.CollapsableFrame("Object position (m)", height=0):
            with ui.VStack(spacing=2, height=0):
                for axis in "xyz":
                    model, _ = combo_floatfield_slider_builder(
                        label=f"pos_{axis}", default_val=0.0,
                        min=-args_cli.object_pos_range, max=args_cli.object_pos_range, step=0.001,
                    )
                    object_pos_models[axis] = model
        with ui.CollapsableFrame("Object rotation (deg)", height=0):
            with ui.VStack(spacing=2, height=0):
                for axis in ("roll", "pitch", "yaw"):
                    model, _ = combo_floatfield_slider_builder(
                        label=axis, default_val=0.0, min=-180.0, max=180.0, step=1.0,
                    )
                    object_rot_models[axis] = model

        def on_reset_object():
            for model in list(object_pos_models.values()) + list(object_rot_models.values()):
                model.set_value(0.0)

        btn_builder(label="", text="Reset object to grasp pose", on_clicked_fn=on_reset_object)

    def on_save_snapshot():
        os.makedirs(args_cli.save_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for finger in FINGER_NAMES:
            sensor_data = raw_env.tactile_sensors[finger].data
            rgb = sensor_data.tactile_rgb_image[0].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args_cli.save_dir, f"{stamp}_{finger}_rgb.png"), bgr)
        print(f"[INFO] Saved tactile RGB snapshots to: {os.path.abspath(args_cli.save_dir)}")

    def format_camera_offsets() -> str:
        per_finger_pos = {}
        per_finger_rot = {}
        for finger in FINGER_NAMES:
            m = cam_pos_models[finger]
            dp = torch.tensor([[m["x"].get_value_as_float(), m["y"].get_value_as_float(), m["z"].get_value_as_float()]], device=device)
            r = cam_rot_models[finger]
            droll = torch.tensor([np.deg2rad(r["roll"].get_value_as_float())], device=device)
            dpitch = torch.tensor([np.deg2rad(r["pitch"].get_value_as_float())], device=device)
            dyaw = torch.tensor([np.deg2rad(r["yaw"].get_value_as_float())], device=device)
            dquat = quat_from_euler_xyz(droll, dpitch, dyaw)
            abs_pos = (base_cam_pos + dp)[0].detach().cpu().numpy()
            abs_quat = quat_mul(base_cam_quat, dquat)[0].detach().cpu().numpy()
            per_finger_pos[finger] = tuple(float(v) for v in abs_pos)
            per_finger_rot[finger] = tuple(float(v) for v in abs_quat)

        all_pos_equal = len(set(per_finger_pos.values())) == 1
        all_rot_equal = len(set(per_finger_rot.values())) == 1
        lines = []
        if all_pos_equal and all_rot_equal:
            lines.append("# all four fingers share the same offset -- drop straight into sensors.py's own constants:")
            lines.append(f"DIGIT_CAMERA_OFFSET_POS = {per_finger_pos[FINGER_NAMES[0]]}")
            lines.append(f"DIGIT_CAMERA_OFFSET_ROT = {per_finger_rot[FINGER_NAMES[0]]}")
        else:
            lines.append("# per-finger offsets differ -- sensors.py's build_digit_camera_cfg would need a per-finger lookup:")
            lines.append("DIGIT_CAMERA_OFFSET_POS = {")
            for finger in FINGER_NAMES:
                lines.append(f'    "{finger}": {per_finger_pos[finger]},')
            lines.append("}")
            lines.append("DIGIT_CAMERA_OFFSET_ROT = {")
            for finger in FINGER_NAMES:
                lines.append(f'    "{finger}": {per_finger_rot[finger]},')
            lines.append("}")
        return "\n".join(lines)

    def on_print_and_save_offsets():
        text = format_camera_offsets()
        print(text)
        os.makedirs(args_cli.save_dir, exist_ok=True)
        out_path = os.path.join(args_cli.save_dir, f"digit_camera_offsets_{time.strftime('%Y%m%d_%H%M%S')}.py")
        with open(out_path, "w") as f:
            f.write(text + "\n")
        print(f"[INFO] Saved to: {os.path.abspath(out_path)}")

    def build_camera_section(finger: str):
        with ui.CollapsableFrame(f"{finger.capitalize()} DIGIT camera ({HOUSING_LINKS[finger]})", height=0):
            with ui.VStack(spacing=2, height=0):
                ui.Label("Local offset on top of the shared DIGIT_CAMERA_OFFSET_POS/_ROT baseline.", word_wrap=True, height=24)
                for axis in "xyz":
                    model, _ = combo_floatfield_slider_builder(
                        label=f"pos_{axis} (m)", default_val=0.0,
                        min=-args_cli.cam_pos_range, max=args_cli.cam_pos_range, step=0.0005,
                    )
                    cam_pos_models[finger][axis] = model
                for axis in ("roll", "pitch", "yaw"):
                    model, _ = combo_floatfield_slider_builder(
                        label=f"{axis} (deg)", default_val=0.0,
                        min=-args_cli.cam_rot_range_deg, max=args_cli.cam_rot_range_deg, step=0.5,
                    )
                    cam_rot_models[finger][axis] = model

                def on_reset_finger(finger=finger):
                    for model in list(cam_pos_models[finger].values()) + list(cam_rot_models[finger].values()):
                        model.set_value(0.0)

                btn_builder(label="", text=f"Reset {finger} camera offset", on_clicked_fn=on_reset_finger)

    def build_window():
        window = ui.Window("Tune DIGIT Camera Pose", width=1150, height=820, dockPreference=ui.DockPreference.MAIN)
        window.deferred_dock_in("Property", ui.DockPolicy.DO_NOTHING)
        with window.frame:
            with ui.HStack(spacing=8):
                with ui.ScrollingFrame(width=520):
                    with ui.VStack(spacing=4):
                        build_object_section()
                        ui.Spacer(height=8)
                        for finger in FINGER_NAMES:
                            build_camera_section(finger)
                        ui.Spacer(height=8)
                        btn_builder(label="", text="Print / Save camera offsets", on_clicked_fn=on_print_and_save_offsets)
                        btn_builder(label="", text="Save tactile RGB snapshot", on_clicked_fn=on_save_snapshot)
                with ui.ScrollingFrame():
                    with ui.VStack(spacing=8):
                        placeholder = np.zeros((digit_render_cfg.image_height, digit_render_cfg.image_width, 3), dtype=np.uint8)
                        for finger in FINGER_NAMES:
                            image_plots[finger] = LiveImageView(image=placeholder, label=f"{finger} tactile RGB", widget_height=180)
        return window

    window = build_window()
    window.visible = True

    while simulation_app.is_running():
        # ---- freeze the hand at its sampled grasp pose ----
        raw_env.hand.write_joint_state_to_sim(hold_joint_pos, hold_joint_vel)
        raw_env.hand.set_joint_position_target(hold_joint_pos)

        # ---- object pose from sliders, relative to its grasp-reset pose ----
        dpos = torch.tensor(
            [[object_pos_models["x"].get_value_as_float(), object_pos_models["y"].get_value_as_float(), object_pos_models["z"].get_value_as_float()]],
            device=device,
        )
        droll = torch.tensor([np.deg2rad(object_rot_models["roll"].get_value_as_float())], device=device)
        dpitch = torch.tensor([np.deg2rad(object_rot_models["pitch"].get_value_as_float())], device=device)
        dyaw = torch.tensor([np.deg2rad(object_rot_models["yaw"].get_value_as_float())], device=device)
        obj_dquat = quat_from_euler_xyz(droll, dpitch, dyaw)
        obj_pos = object_start_pos + dpos
        obj_quat = quat_mul(object_start_quat, obj_dquat)
        raw_env.object.write_root_pose_to_sim(torch.cat([obj_pos, obj_quat], dim=-1))
        raw_env.object.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

        # ---- per-finger DIGIT camera offset from sliders, written directly as the "cam"
        # prim's LOCAL pose (relative to its parent fingertip-housing prim). Deliberately
        # NOT going through `Camera.set_world_poses`: that call re-derives a local pose by
        # reading the parent's transform via `UsdGeom.Xformable.ComputeLocalToWorldTransform
        # (Usd.TimeCode.Default())` (see `XformPrimView.set_world_poses`), which reflects the
        # prim's USD-authored rest pose, not the live PhysX-simulated joint pose -- for a
        # camera nested under a moving articulation link, that mismatch parks the camera
        # somewhere unrelated to the fingertip regardless of what "world" pose we ask for,
        # which is why the sliders had no visible effect on the tactile image. Writing the
        # local pose ourselves sidesteps that lookup entirely -- USD parents this prim under
        # the housing link, so it still moves with the hand for free.
        for finger in FINGER_NAMES:
            m = cam_pos_models[finger]
            local_pos = base_cam_pos + torch.tensor([[m["x"].get_value_as_float(), m["y"].get_value_as_float(), m["z"].get_value_as_float()]], device=device)
            r = cam_rot_models[finger]
            c_droll = torch.tensor([np.deg2rad(r["roll"].get_value_as_float())], device=device)
            c_dpitch = torch.tensor([np.deg2rad(r["pitch"].get_value_as_float())], device=device)
            c_dyaw = torch.tensor([np.deg2rad(r["yaw"].get_value_as_float())], device=device)
            local_dquat = quat_from_euler_xyz(c_droll, c_dpitch, c_dyaw)
            # still in the "world" convention (local +X=forward, +Z=up) DIGIT_CAMERA_OFFSET_ROT
            # itself is authored in -- must convert to the USD/opengl convention actually
            # stored on the prim's own `xformOp:orient` before writing it (the one conversion
            # `Camera.__init__`/`set_world_poses` would otherwise have done for us).
            local_quat_world_conv = quat_mul(base_cam_quat.float(), local_dquat.float())
            local_quat_opengl = convert_camera_frame_orientation_convention(local_quat_world_conv.float(), origin="world", target="opengl")

            camera = raw_env.tactile_sensors[finger]._camera_sensor
            camera._view.set_local_poses(
                local_pos.to(camera.device).float(), local_quat_opengl.to(camera.device).float(), indices=[0],
            )

        raw_env.scene.write_data_to_sim()
        raw_env.sim.step()
        raw_env.scene.update(raw_env.physics_dt)

        for finger in FINGER_NAMES:
            rgb = raw_env.tactile_sensors[finger].data.tactile_rgb_image[0].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            image_plots[finger].update_image(rgb)

    raw_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
