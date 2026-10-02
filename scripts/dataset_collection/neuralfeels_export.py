"""Capture calibrated frames from a single IsaacLab environment for NeuralFeels."""

import numpy as np
from PIL import Image

from .conventions import FINGER_NAMES, resized_intrinsics
from .legacy_tactile import DEFAULT_URDF, EXPORT_JOINT_NUMBERS, LegacyAllegro, SOURCE_FINGERS
from .feelsight_writer import scene_segmentation_image


class NeuralFeelsCapture:
    """Construct after capture_initial_tactile_render(), before collecting episodes.

    Cameras already use the original NeuralFeels FK camera frames. Preserve full
    images and verify the frame match before saving; no reprojection is required.
    """

    def __init__(self, env, tactile_size, depth_scale, mask_eps, neuralfeels_urdf=DEFAULT_URDF):

        self.env = env
        self.size = (tactile_size[1], tactile_size[0])  # retain the saved portrait dimensions
        self.depth_scale = depth_scale
        self.mask_eps = mask_eps
        self.joint_names = [f"joint_{i}.0" for i in range(16)]
        self.joint_ids = [env.hand.joint_names.index(f"joint_{i}_0") for i in EXPORT_JOINT_NUMBERS]
        self.legacy_hand = LegacyAllegro(neuralfeels_urdf)
        intrinsics = [self._intrinsics(env.tactile_sensors[f]._camera_sensor, self.size) for f in FINGER_NAMES]
        if any(k != intrinsics[0] for k in intrinsics[1:]):
            raise ValueError("NeuralFeels requires identical tactile intrinsics for all fingers")
        self.digit_info = dict(depth_scale=depth_scale, cam_dist=-.022, intrinsics=intrinsics[0])
        self.scene_intrinsics = {name: self._intrinsics(cam) for name, cam in env.scene_cams.items()}

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy().copy()

    def _resize_depth(self, depth):
        # Nearest keeps metric depth and the contact mask aligned at silhouettes.
        return np.array(Image.fromarray(depth.astype(np.float32)).resize(self.size, Image.Resampling.NEAREST))

    def _intrinsics(self, camera, target_size=None):
        height, width = camera.data.image_shape
        return resized_intrinsics(self._numpy(camera.data.intrinsic_matrices[0]),
                                  (width, height), target_size or (width, height))

    def capture(self):
        from .sensors import pose_to_matrix

        env = self.env
        def pose(pos, quat):
            return pose_to_matrix(self._numpy(pos), self._numpy(quat))

        base_pose = pose(env.hand.data.root_pos_w[0], env.hand.data.root_quat_w[0])
        joint_state = self._numpy(env.hand.data.joint_pos[0, self.joint_ids]).astype(np.float32)
        finger_poses, camera_poses = self.legacy_hand.poses(joint_state, base_pose)
        tactile = {}
        for i, target in enumerate(FINGER_NAMES):
            finger = SOURCE_FINGERS[target]
            sensor = env.tactile_sensors[finger]
            sensor_data = sensor.data
            rgb = np.clip(self._numpy(sensor_data.tactile_rgb_image[0]), 0, 255).astype(np.uint8)
            depth = self._numpy(sensor_data.tactile_depth_image[0, ..., 0])
            nominal = self._numpy(env.nominal_tactile_depth[finger][0, ..., 0])
            contact = np.isfinite(depth) & (depth > 0) & ((nominal-depth) > self.mask_eps)
            rgb = np.array(Image.fromarray(rgb).resize(self.size, Image.Resampling.BILINEAR))
            contact = np.array(Image.fromarray(contact).resize(self.size, Image.Resampling.NEAREST))
            depth = self._resize_depth(depth)
            camera = sensor._camera_sensor
            source_pose = pose(camera.data.pos_w[0], camera.data.quat_w_opengl[0])
            if not np.allclose(source_pose, camera_poses[i], atol=2e-5, rtol=0):
                raise ValueError(f"{target}: simulator camera differs from NeuralFeels FK. "
                                 "Check the hand URDF, cached USD, joint mapping and camera offset.")
            tactile[target] = dict(rgb=rgb, depth=np.where(contact, -depth, np.nan).astype(np.float32))

        scene_frames = {}
        for name, camera in env.scene_cams.items():
            data = camera.data
            depth = self._numpy(data.output["distance_to_image_plane"][0, ..., 0])
            depth = np.where(np.isfinite(depth) & (depth > 0), -depth, 0).astype(np.float32)
            seg_ids = self._numpy(data.output["instance_segmentation_fast"][0])
            seg_info = data.info.get("instance_segmentation_fast") if data.info else None
            scene_frames[name] = dict(
                rgb=self._numpy(data.output["rgb"][0]).astype(np.uint8)[..., :3], depth=depth,
                seg=scene_segmentation_image(seg_ids, seg_info),
                cam_pose=pose(data.pos_w[0], data.quat_w_opengl[0]),
            )
        return dict(
            object_pose=pose(env.object.data.root_pos_w[0], env.object.data.root_quat_w[0]),
            base_pose=base_pose, joint_state=joint_state, finger_poses=finger_poses,
            tactile=tactile, scene_frames=scene_frames,
        )
