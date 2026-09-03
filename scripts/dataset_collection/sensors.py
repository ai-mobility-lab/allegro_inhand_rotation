# --------------------------------------------------------
# Sensor rig for stage-2 dataset collection: an external RGB-D + instance-segmentation
# "scene" camera plus four DIGIT VisuoTactileSensors, bolted onto `AllegroHandHoraEnv`
# (which by itself has no cameras or tactile sensing at all -- see
# `hora/tasks/isaaclab/allegro_hand_hora_env.py`).
#
# `AllegroHandHoraEnv._setup_scene()` builds the hand/object by hand (`self.hand =
# Articulation(...)`, `self.scene.articulations["robot"] = self.hand`, ...) rather than
# through `InteractiveSceneCfg` field reflection, so new sensors are added the same
# manual way here: construct them directly and register them into `scene.sensors[name]`
# *after* `super()._setup_scene()` returns (i.e. after `scene.clone_environments()` has
# already replicated the hand's `link_*_tip_elastomer` bodies into every env -- sensors
# nested under those prim paths need them to already exist).
#
# DIGIT camera placement mirrors `inhand_rotation`'s own
# `inhand_rotation_env_cfg.py`/`scripts/tools/tacsl_sensor_demo.py` (same physical
# sensor, same `allegro_*_digit_left_elastomer.urdf` derived asset).
# --------------------------------------------------------

from __future__ import annotations

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab_contrib.sensors.tacsl_sensor import (
    GelSightRenderCfg,
    VisuoTactileSensor,
    VisuoTactileSensorCfg,
)

from hora.tasks.isaaclab.allegro_hand_hora_env import AllegroHandHoraEnv

FINGER_NAMES = ["index", "middle", "ring", "thumb"]

# raw Allegro joint numbering used by `allegro_digit_left_elastomer.urdf` (and by
# `configs/task/LeftAllegroHandDigitHora.yaml`'s `fingertipLinkNames`): index -> link_3,
# middle -> link_7, ring -> link_11, thumb -> link_15.
HOUSING_LINKS = {"index": "link_3_0_tip", "middle": "link_7_0_tip", "ring": "link_11_0_tip", "thumb": "link_15_0_tip"}
ELASTOMER_LINKS = {f: f"{link}_elastomer" for f, link in HOUSING_LINKS.items()}

# ---- DIGIT camera intrinsics (Lambeta et al. 2020 spec) -- same constants/derivation as
# inhand_rotation's `inhand_rotation_env_cfg.py`/`tacsl_sensor_demo.py`.
DIGIT_FOCAL_LENGTH_MM = 1.15
DIGIT_WORKING_DISTANCE_MM = 12.0
DIGIT_SENSING_FIELD_MM = (16.0, 19.0)  # (horizontal, vertical)
DIGIT_HORIZONTAL_APERTURE_MM = DIGIT_FOCAL_LENGTH_MM * DIGIT_SENSING_FIELD_MM[0] / DIGIT_WORKING_DISTANCE_MM
DIGIT_VERTICAL_APERTURE_MM = DIGIT_FOCAL_LENGTH_MM * DIGIT_SENSING_FIELD_MM[1] / DIGIT_WORKING_DISTANCE_MM
# camera offset from each link_*_tip housing's own origin (meters): the elastomer pad's
# bounding-box center in the tip's local frame, `_DIGIT_WORKING_DISTANCE_MM` behind that
# along local +X. Same numbers as inhand_rotation_env_cfg.py's GRASP_ELASTOMER_LOCAL_OFFSET
# (this repo's allegro_digit_left_elastomer.urdf shares the same elastomer geometry).
DIGIT_CAMERA_OFFSET_POS = (0.0181021569 - DIGIT_WORKING_DISTANCE_MM / 1000.0, 0.0000018431, 0.0172339268)
# local +X -> camera forward, local +Z -> camera up (see tacsl_sensor_demo.py's own comment
# on this same quaternion).
DIGIT_CAMERA_OFFSET_ROT = (0.7071068, -0.7071068, 0.0, 0.0)


def pinhole_intrinsics(width: int, height: int, focal_length_mm: float, h_aperture_mm: float, v_aperture_mm: float) -> dict:
    """Standard pinhole intrinsics for an IsaacLab `PinholeCameraCfg` at a given render resolution."""
    return {
        "w": int(width),
        "h": int(height),
        "fx": float(focal_length_mm * width / h_aperture_mm),
        "fy": float(focal_length_mm * height / v_aperture_mm),
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
    }


def build_digit_render_cfg(calib_dir) -> GelSightRenderCfg:
    """DIGIT Taxim render config, reading real calibration data from `calib_dir`.

    `VisuoTactileSensor` only ever produces tactile depth/RGB when `enable_camera_tactile
    =True` (see `visuotactile_sensor.py:_initialize_camera_tactile`), and that path
    unconditionally constructs a `GelsightRender`, which raises `FileNotFoundError` unless
    `bg.jpg` + `polycalib.npz` (+ optionally `real_bg.npy`) exist under `calib_dir` -- there
    is no depth-only fallback. This repo does not ship that calibration data itself; the
    default caller path points at the sibling `inhand_rotation` project's copy.
    """
    from pathlib import Path

    calib_dir = Path(calib_dir).resolve()
    required = ["bg.jpg", "polycalib.npz", "real_bg.npy"]
    missing = [f for f in required if not (calib_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"DIGIT Taxim calibration files {missing} not found under {calib_dir}. Pass "
            "--digit_calib_dir pointing at a directory with bg.jpg + polycalib.npz + "
            "real_bg.npy (e.g. inhand_rotation/source/inhand_rotation/inhand_rotation/"
            "assets/digit_data)."
        )
    return GelSightRenderCfg(
        base_data_path=str(calib_dir.parent),
        sensor_data_dir_name=calib_dir.name,
        background_path="bg.jpg",
        calib_path="polycalib.npz",
        real_background="real_bg.npy",
        image_height=480,
        image_width=640,
        num_bins=125,
        mm_per_pixel=(19.0 / 640 + 16.0 / 480) / 2,
    )


def build_digit_camera_cfg(tip_link_name: str, render_cfg: GelSightRenderCfg) -> TiledCameraCfg:
    """DIGIT depth camera for one fingertip housing (feeds `VisuoTactileSensorCfg.camera_cfg`)."""
    return TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/Robot/{tip_link_name}/cam",
        height=render_cfg.image_height,
        width=render_cfg.image_width,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=DIGIT_FOCAL_LENGTH_MM,
            horizontal_aperture=DIGIT_HORIZONTAL_APERTURE_MM,
            vertical_aperture=DIGIT_VERTICAL_APERTURE_MM,
            clipping_range=(0.001, 0.05),
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=DIGIT_CAMERA_OFFSET_POS, rot=DIGIT_CAMERA_OFFSET_ROT, convention="world"),
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _matrix_to_quat_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    """Standard (Shepperd's method) rotation-matrix -> (w, x, y, z) quaternion."""
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def quat_from_forward_up(forward, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) rotating local +X -> `forward`, local +Z -> `up` (right-handed).

    Matches the `convention="world"` interpretation this codebase's own DIGIT camera
    offsets already use (local +X = forward, local +Z = up) -- see `DIGIT_CAMERA_OFFSET_ROT`.
    """
    forward = _normalize(np.asarray(forward, dtype=np.float64))
    up = np.asarray(up, dtype=np.float64)
    y_axis = _normalize(np.cross(up, forward))
    z_axis = np.cross(forward, y_axis)
    r = np.stack([forward, y_axis, z_axis], axis=1)  # columns = local X, Y, Z expressed in world frame
    return _matrix_to_quat_wxyz(r)


def pose_to_matrix(pos, quat_wxyz) -> np.ndarray:
    """(pos, quat (w,x,y,z)) -> 4x4 homogeneous world transform."""
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    w, x, y, z = (float(c) for c in quat_wxyz)
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r
    m[:3, 3] = pos
    return m


def build_scene_camera_cfg(
    eye,
    target,
    width: int = 640,
    height: int = 480,
    focal_length_mm: float = 24.0,
    horizontal_aperture_mm: float = 20.955,
    prim_path: str = "/World/envs/env_.*/scene_cam",
) -> TiledCameraCfg:
    """External RGB-D + instance-segmentation camera, aimed from `eye` at `target` (world frame)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    quat = quat_from_forward_up(target - eye)
    vertical_aperture_mm = horizontal_aperture_mm * height / width
    return TiledCameraCfg(
        prim_path=prim_path,
        height=height,
        width=width,
        data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
        colorize_instance_segmentation=False,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length_mm,
            horizontal_aperture=horizontal_aperture_mm,
            vertical_aperture=vertical_aperture_mm,
            clipping_range=(0.05, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye.tolist()), rot=quat, convention="world"),
    )


class DatasetAllegroHandHoraEnv(AllegroHandHoraEnv):
    """`AllegroHandHoraEnv` plus one external scene camera and four DIGIT tactile sensors.

    Pass `digit_render_cfg=None`/`scene_camera_cfg=None` to skip either sensor group (e.g.
    while iterating on the base policy rollout without paying the rendering cost).
    """

    def __init__(self, cfg, render_mode=None, digit_render_cfg=None, scene_camera_cfg=None, **kwargs):
        self._digit_render_cfg = digit_render_cfg
        self._scene_camera_cfg = scene_camera_cfg
        self.tactile_sensors: dict[str, VisuoTactileSensor] = {}
        self.scene_cam: TiledCamera | None = None
        self.nominal_tactile_depth: dict[str, "torch.Tensor"] = {}
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def _build_object_cfg(self):
        # `AllegroHandHoraEnv`'s own object prims carry no USD Semantics schema, so
        # Replicator's instance segmentation collapses all of them into one shared,
        # unlabeled id (`idToLabels` value `"UNLABELLED"`) -- indistinguishable from the
        # ground plane/background. Tag each per-env object shape with a `class=object`
        # semantic label so it gets its own id->label entry `scene_segmentation_image`
        # can actually match against.
        cfg = super()._build_object_cfg()
        for asset_cfg in cfg.spawn.assets_cfg:
            asset_cfg.semantic_tags = [("class", "object")]
        return cfg

    def _setup_scene(self):
        super()._setup_scene()

        if self._scene_camera_cfg is not None:
            self.scene_cam = TiledCamera(self._scene_camera_cfg)
            self.scene.sensors["scene_cam"] = self.scene_cam

        if self._digit_render_cfg is not None:
            for finger in FINGER_NAMES:
                tactile_cfg = VisuoTactileSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{ELASTOMER_LINKS[finger]}/tactile_sensor",
                    render_cfg=self._digit_render_cfg,
                    enable_camera_tactile=True,
                    enable_force_field=False,
                    tactile_array_size=(16, 19),
                    tactile_margin=0.002,
                    camera_cfg=build_digit_camera_cfg(HOUSING_LINKS[finger], self._digit_render_cfg),
                )
                sensor = VisuoTactileSensor(tactile_cfg)
                self.tactile_sensors[finger] = sensor
                self.scene.sensors[f"{finger}_tactile"] = sensor

    def capture_initial_tactile_render(self):
        """Set each tactile sensor's no-contact depth baseline. Call once, right after the
        first `env.reset()`, before the recorded rollout starts.

        Unlike `scripts/tools/tacsl_sensor_demo.py` (which places an independent nut in
        front of each pad specifically to avoid this issue), HORA resets straight into a
        pre-formed grasp (`reset_joints_from_grasp_cache`) -- fingertips are typically
        already touching the object at this point, so this baseline already includes that
        initial contact. Recorded tactile depth/mask therefore reflect deformation *beyond*
        the grasp's resting contact, not absolute contact/no-contact.
        """
        self.nominal_tactile_depth = {}
        for finger, sensor in self.tactile_sensors.items():
            baseline = sensor.get_initial_render()
            self.nominal_tactile_depth[finger] = baseline["distance_to_image_plane"].clone()
