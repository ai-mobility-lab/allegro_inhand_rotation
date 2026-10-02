"""Export geometry for the unmodified NeuralFeels Allegro and image loaders."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .conventions import FINGER_NAMES, NEURALFEELS_TIP_TO_CAMERA

DEFAULT_URDF = Path(__file__).resolve().parents[3] / 'neuralfeels/data/assets/allegro/allegro_digit_left_ball.urdf'
# NeuralFeels labels the two outer fingers opposite to the simulator URDF.
SOURCE_FINGERS = {'index': 'ring', 'middle': 'middle', 'ring': 'index', 'thumb': 'thumb'}
EXPORT_JOINT_NUMBERS = [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15]
TIP_TO_CAMERA = NEURALFEELS_TIP_TO_CAMERA


class LegacyAllegro:
    """NumPy FK equivalent to stock Allegro.get_fk, with numeric URDF joint order.

    In the stock loader, the index/ring swap followed by torchkin's traversal map
    assigns saved q[i] to joint_i.0. Do not repeat either permutation here.
    """
    def __init__(self, urdf=DEFAULT_URDF):
        self.joints = []
        for j in ET.parse(urdf).getroot().findall('joint'):
            origin = j.find('origin')
            transform = np.eye(4)
            if origin is not None:
                transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            axis, number = None, None
            if j.get('type') != 'fixed':
                if j.get('type') != 'revolute':
                    raise ValueError('Expected the NeuralFeels revolute Allegro URDF')
                number = int(float(j.get('name').removeprefix('joint_')))
                axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
            self.joints.append((j.find('parent').get('link'), j.find('child').get('link'), transform, axis, number))

    def poses(self, q, base):
        transforms = {'base_link': base}
        for parent, child, origin, axis, number in self.joints:
            motion = np.eye(4)
            if number is not None:
                motion[:3, :3] = Rotation.from_rotvec(axis * q[number]).as_matrix()
            transforms[child] = transforms[parent] @ origin @ motion
        tips = np.stack([transforms[f'link_{i}.0_tip'] for i in [3, 7, 11, 15]])
        return tips, tips @ TIP_TO_CAMERA


def project_surface(rgb, depth, contact, intrinsics, source_pose, target_pose):
    """Backproject a measured surface into the camera that stock NeuralFeels uses."""
    v, u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    z = -depth[valid]
    xyz = np.column_stack((-(u[valid]-intrinsics['cx'])/intrinsics['fx']*z,
                           (v[valid]-intrinsics['cy'])/intrinsics['fy']*z, z))
    transform = np.linalg.inv(target_pose) @ source_pose
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    contact = contact[valid]
    front = xyz[:, 2] < -1e-4
    if np.any(contact & ~front):
        raise ValueError('A tactile contact lies behind the legacy NeuralFeels camera; cannot export it faithfully')
    return dict(points=xyz[front].astype(np.float32), rgb=rgb[valid][front], contact=contact[front])


def projection_intrinsics(frames, size):
    """Fit one common, fixed K to every recorded finger's virtual-camera field of view."""
    lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
    for frame in frames:
        for sample in frame.values():
            points = sample['points']
            if not len(points):
                continue
            uv = points[:, :2] / -points[:, 2:3]
            uv[:, 1] *= -1
            lower = np.minimum(lower, uv.min(axis=0))
            upper = np.maximum(upper, uv.max(axis=0))
    if not np.isfinite(lower).all():
        raise ValueError('No tactile surface is visible in the legacy cameras')
    w, h = size
    focal = (np.array([w, h]) - 3) / np.maximum(upper-lower, 1e-6)
    center = 1 - lower*focal
    return dict(w=w, h=h, fx=float(focal[0]), fy=float(focal[1]), cx=float(center[0]), cy=float(center[1]))


def rasterize(sample, k):
    """Nearest-pixel splatting with a z-buffer; unobserved pixels remain masked out."""
    points = sample['points']
    u = np.rint(points[:, 0]/-points[:, 2]*k['fx']+k['cx']).astype(int)
    v = np.rint(points[:, 1]/points[:, 2]*k['fy']+k['cy']).astype(int)
    valid = (u >= 0) & (u < k['w']) & (v >= 0) & (v < k['h'])
    ids = np.flatnonzero(valid)
    pixel = v[valid]*k['w']+u[valid]
    order = np.lexsort((-points[ids, 2], pixel))
    _, first = np.unique(pixel[order], return_index=True)
    selected = ids[order[first]]
    pixel = pixel[order[first]]
    rgb = np.zeros((k['h']*k['w'], 3), dtype=np.uint8)
    depth = np.full(k['h']*k['w'], np.nan, dtype=np.float32)
    rgb[pixel] = sample['rgb'][selected]
    contact = sample['contact'][selected]
    depth[pixel[contact]] = points[selected[contact], 2]
    return rgb.reshape(k['h'], k['w'], 3), depth.reshape(k['h'], k['w'])


def depth_encoding(frames, requested_scale):
    """Fit stock depth/scale + cam_dist to the episode, reserving zero for no contact."""
    lower, upper = np.inf, -np.inf
    for frame in frames:
        for sample in frame.values():
            depths = (sample['depth'][np.isfinite(sample['depth'])] if 'depth' in sample
                      else sample['points'][sample['contact'], 2])
            if len(depths):
                lower, upper = min(lower, float(depths.min())), max(upper, float(depths.max()))
    if not np.isfinite(lower):
        return requested_scale, -.022
    scale = min(requested_scale, 253.0/max(upper-lower, 1e-9))
    return scale, lower-1.0/scale


def encode_depth(depth, scale, cam_dist):
    valid = np.isfinite(depth)
    encoded = np.zeros(depth.shape, dtype=np.uint8)
    encoded[valid] = np.clip(np.rint((depth[valid]-cam_dist)*scale), 1, 255).astype(np.uint8)
    return encoded, valid.astype(np.uint8)*255
