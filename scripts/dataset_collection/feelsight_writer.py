# --------------------------------------------------------
# Writes one rollout episode to disk in the same layout as NeuralFeels'
# `feelsight` dataset (see e.g. neuralfeels/data/feelsight/015_peach/00/):
#
#   <object_name>/<episode_idx>/
#     data.pkl                                    -- object/allegro/digit_info/realsense/time
#     allegro/<finger>/{image,depth,mask}/<i>.jpg  -- per-finger DIGIT tactile frames
#     realsense/<camera_name>/image/<i>.jpg        -- external RGB frames
#     realsense/<camera_name>/seg/<i>.jpg          -- external seg frames (0=bg, 127=hand, 255=object)
#     realsense/<camera_name>/depth.npz            -- stacked external depth, meters
#
# This is a from-scratch generator, not a bit-exact clone of the real dataset -- notably,
# depth here is stored as plain positive meters (camera-to-surface distance), not
# neuralfeels' negative OpenGL-convention `z` (see `neuralfeels/datasets/dataset.py`); adapt
# the sign if feeding this into a loader written against the real feelsight data.
# --------------------------------------------------------

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from .sensors import FINGER_NAMES


# (label substring to match in `idToLabels`, pixel value to paint) -- painted in order, so a
# later entry wins on an (impossible, in practice) id collision. Matches the real feelsight
# `realsense/seg` convention of 3 bands: 0=background, ~127=hand, ~255=object (see
# `neuralfeels/datasets/dataset.py`'s `round to nearest of {0, 127.5, 255}`).
DEFAULT_SEG_CLASSES = (("robot", 127), ("object", 255))


def scene_segmentation_image(seg_ids: np.ndarray, seg_info: dict | None, class_values=DEFAULT_SEG_CLASSES) -> np.ndarray:
    """Labeled uint8 segmentation image from a `TiledCamera`'s raw `instance_segmentation_fast` output.

    `seg_ids`: (H, W) or (H, W, 1) raw integer instance ids for one frame.
    `seg_info`: the corresponding `camera.data.info["instance_segmentation_fast"]` entry --
    a dict carrying an `idToLabels` mapping (Omniverse Replicator's instance-segmentation
    schema: `{"<id>": "<label>", ...}`).

    Unlabeled prims (no USD Semantics schema authored) all collapse into one shared id
    whose label is the literal string `"UNLABELLED"` -- a raw prim-path substring match
    against that never matches. Fixed by having `DatasetAllegroHandHoraEnv._build_object_cfg`
    tag the object's spawn cfg with `semantic_tags=[("class", "object")]` (and the main
    script tag `env_cfg.robot_cfg` with `[("class", "robot")]`), so their ids get real
    `"object"`/`"robot"` labels here to match against.

    Falls back to an all-background image (each distinct failure mode warned once) if the
    expected `idToLabels` key is missing, or if no id's label matches any `class_values`
    entry -- the latter dumps the raw `idToLabels` content so a mismatch on a different
    Isaac Sim/Replicator version is diagnosable from one run instead of guessing blind.
    """
    seg_ids = np.asarray(seg_ids)
    if seg_ids.ndim == 3:
        seg_ids = seg_ids[..., 0]
    out = np.zeros(seg_ids.shape, dtype=np.uint8)
    if not seg_info:
        return out
    id_to_labels = seg_info.get("idToLabels") if isinstance(seg_info, dict) else None
    if not id_to_labels:
        if not scene_segmentation_image._warned_missing:
            print(f"[dataset_collection] WARNING: no 'idToLabels' in segmentation info {list(seg_info.keys()) if isinstance(seg_info, dict) else seg_info!r} -- seg frames will be all-background.")
            scene_segmentation_image._warned_missing = True
        return out
    any_match = False
    for label_substr, pixel_value in class_values:
        ids = [int(k) for k, v in id_to_labels.items() if label_substr.lower() in str(v).lower()]
        if ids:
            any_match = True
            out[np.isin(seg_ids, ids)] = pixel_value
    if not any_match and not scene_segmentation_image._warned_no_match:
        print(f"[dataset_collection] WARNING: no id in idToLabels matched any of {class_values} -- idToLabels={id_to_labels!r} -- seg frames will be all-background.")
        scene_segmentation_image._warned_no_match = True
    return out


scene_segmentation_image._warned_missing = False
scene_segmentation_image._warned_no_match = False


class EpisodeWriter:
    """Accumulates one episode's frames/poses and writes them out in feelsight layout."""

    def __init__(self, object_root: Path, episode_idx: int, camera_name: str = "front", jpg_quality: int = 92):
        self.dir = Path(object_root) / f"{episode_idx:02d}"
        self.camera_name = camera_name
        self.jpg_quality = jpg_quality

        for finger in FINGER_NAMES:
            for sub in ("image", "depth", "mask"):
                (self.dir / "allegro" / finger / sub).mkdir(parents=True, exist_ok=True)
        (self.dir / "realsense" / camera_name / "image").mkdir(parents=True, exist_ok=True)
        (self.dir / "realsense" / camera_name / "seg").mkdir(parents=True, exist_ok=True)

        self.frame_idx = 0
        self.times: list[float] = []
        self.object_poses: list[np.ndarray] = []
        self.finger_poses: list[np.ndarray] = []
        self.joint_states: list[np.ndarray] = []
        self.base_pose: np.ndarray | None = None
        self.realsense_poses: list[np.ndarray] = []
        self.realsense_depth_frames: list[np.ndarray] = []

    @staticmethod
    def _save_gray(arr_uint8: np.ndarray, path: Path, quality: int = 95, rotate=False):
        if rotate:
            Image.fromarray(arr_uint8, mode="L").rotate(90, expand=True).save(path, quality=quality)
        else:
            Image.fromarray(arr_uint8, mode="L").save(path, quality=quality)

    @staticmethod
    def _save_rgb(arr_uint8: np.ndarray, path: Path, quality: int, rotate=False):
        if rotate:
            Image.fromarray(arr_uint8, mode="RGB").rotate(90, expand=True).save(path, quality=quality)
        else:
            Image.fromarray(arr_uint8, mode="RGB").save(path, quality=quality)
            
    def add_step(
        self,
        t: float,
        object_pose: np.ndarray,
        finger_poses: np.ndarray,
        joint_state: np.ndarray,
        base_pose: np.ndarray,
        tactile: dict,
        scene_rgb: np.ndarray,
        scene_depth: np.ndarray,
        scene_seg: np.ndarray,
        scene_cam_pose: np.ndarray,
    ):
        i = self.frame_idx
        self.times.append(t)
        self.object_poses.append(object_pose)
        self.finger_poses.append(finger_poses)
        self.joint_states.append(joint_state)
        if self.base_pose is None:
            self.base_pose = base_pose

        for finger in FINGER_NAMES:
            img, depth_u8, mask_u8 = tactile[finger]
            # rotate images left -> bottom, right -> top
            
            self._save_rgb(img, self.dir / "allegro" / finger / "image" / f"{i}.jpg", self.jpg_quality, rotate=True)
            self._save_gray(depth_u8, self.dir / "allegro" / finger / "depth" / f"{i}.jpg", rotate=True)
            self._save_gray(mask_u8, self.dir / "allegro" / finger / "mask" / f"{i}.jpg", rotate=True)

        self._save_rgb(scene_rgb, self.dir / "realsense" / self.camera_name / "image" / f"{i}.jpg", self.jpg_quality)
        self._save_gray(scene_seg, self.dir / "realsense" / self.camera_name / "seg" / f"{i}.jpg")

        self.realsense_depth_frames.append(np.asarray(scene_depth, dtype=np.float32))
        self.realsense_poses.append(scene_cam_pose)

        self.frame_idx += 1

    def finalize(self, object_name: str, object_mesh, digit_info: dict, realsense_intrinsics: dict):
        depth_stack = np.stack(self.realsense_depth_frames, axis=0)
        np.savez(
            self.dir / "realsense" / self.camera_name / "depth.npz",
            depth=depth_stack,
            depth_scale=np.float32(1.0),
        )

        data = {
            "object": {
                "name": object_name,
                "mesh": object_mesh,
                "pose": np.stack(self.object_poses, axis=0),
            },
            "allegro": {
                "finger_poses": self.finger_poses,
                "joint_state": np.stack(self.joint_states, axis=0).astype(np.float32),
                "base_pose": self.base_pose,
            },
            "digit_info": digit_info,
            "realsense": {
                self.camera_name: {
                    "depth_scale": 1.0,
                    "pose": np.stack(self.realsense_poses, axis=0).astype(np.float32),
                    "intrinsics": realsense_intrinsics,
                }
            },
            "time": np.asarray(self.times, dtype=np.float64),
        }
        with open(self.dir / "data.pkl", "wb") as f:
            pickle.dump(data, f)
