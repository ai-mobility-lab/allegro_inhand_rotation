"""CPU export regressions against stock NeuralFeels consumer source.

Extract consumer classes/methods with AST to avoid requiring optional SLAM models
for these I/O tests. A separate smoke run can use the full NeuralFeels environment.
"""
import ast
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
NEURALFEELS = Path(os.environ.get('NEURALFEELS_ROOT', REPO.parent / 'neuralfeels'))
sys.path.insert(0, str(REPO / 'scripts'))
from dataset_collection.conventions import FINGER_NAMES, resized_intrinsics
from dataset_collection.legacy_tactile import (
    EXPORT_JOINT_NUMBERS, SOURCE_FINGERS, LegacyAllegro, project_surface,
    projection_intrinsics, rasterize, depth_encoding, encode_depth,
)
from dataset_collection.feelsight_writer import EpisodeWriter


def consumer(class_name, method=None):
    paths = {'TactileDataset': 'datasets/dataset.py', 'VisionDataset': 'datasets/dataset.py',
             'Allegro': 'modules/allegro.py', 'DigitSensor': 'modules/sensor.py', 'DepthTransform': 'datasets/image_transforms.py'}
    path = NEURALFEELS / 'neuralfeels' / paths[class_name]
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    if method:
        node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    namespace = dict(np=np, torch=torch, cv2=cv2, os=os, Dataset=torch.utils.data.Dataset,
                     FrameData=lambda **kw: SimpleNamespace(**kw))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method or class_name]


def test_finger_and_joint_mapping_agree():
    assert SOURCE_FINGERS == dict(index='ring', middle='middle', ring='index', thumb='thumb')
    for target, source in SOURCE_FINGERS.items():
        offset = FINGER_NAMES.index(source)*4
        block = FINGER_NAMES.index(target)*4
        assert EXPORT_JOINT_NUMBERS[block:block+4] == list(range(offset, offset+4))


def test_fk_handles_thumb_geometry_and_returns_camera_frames():
    hand = LegacyAllegro()
    tips, cams = hand.poses(np.zeros(16), np.eye(4))
    assert tips.shape == cams.shape == (4, 4, 4)
    base = np.eye(4)
    base[:3, 3] = [1, 2, 3]
    _, translated = hand.poses(np.zeros(16), base)
    np.testing.assert_allclose(translated, base @ cams)
    # Thumb calibration is not the same translation-only change as the other fingers.
    assert not np.allclose(cams[0, :3, :3], cams[3, :3, :3])


def test_projection_preserves_world_geometry():
    k = dict(w=8, h=6, fx=12., fy=13., cx=3.2, cy=2.1)
    z = np.full((6, 8), .02, np.float32)
    rgb = np.full((6, 8, 3), 100, np.uint8)
    source, target = np.eye(4), np.eye(4)
    target[:3, 3] = [.004, -.002, .003]
    cloud = project_surface(rgb, z, np.ones_like(z, bool), k, source, target)
    v, u = np.indices(z.shape)
    expected = np.stack(((u-k['cx'])/k['fx']*z, -(v-k['cy'])/k['fy']*z, -z), axis=-1).reshape(-1,3)
    np.testing.assert_allclose(cloud['points'] + target[:3,3], expected, atol=1e-8)
    target[2,3] = -.1
    with pytest.raises(ValueError, match='behind'):
        project_surface(rgb, z, np.ones_like(z, bool), k, source, target)


def test_resize_pixel_centers():
    k = resized_intrinsics(np.array([[480,0,320],[0,480,240],[0,0,1.]]), (640,480), (320,240))
    assert k == dict(w=320,h=240,fx=240.,fy=240.,cx=159.75,cy=119.75)


@pytest.fixture
def episodes(tmp_path):
    v, u = np.indices((12, 16))
    sample = dict(points=np.stack(((u-7.5)*.0005, -(v-5.5)*.0005,
                                  -.02 + u*.00001), axis=-1).reshape(-1,3).astype(np.float32),
                  rgb=np.full((192,3),128,np.uint8), contact=np.ones(192,bool))
    frames = {f: sample for f in FINGER_NAMES}
    info = dict(image_size=(16,12),depth_scale=33333.34,cam_dist=-.022)
    k = dict(w=16,h=12,fx=20.,fy=20.,cx=8.,cy=6.)
    for ep in range(2):
        writer = EpisodeWriter(tmp_path, ep)
        for i in range(2):
            writer.add_step(t=i*.05, object_pose=np.eye(4), finger_poses=np.repeat(np.eye(4)[None],4,axis=0),
                            joint_state=np.zeros(16),base_pose=np.eye(4), tactile=frames,
                            scene_frames={'front':dict(rgb=np.full((12,16,3),100,np.uint8),
                                                       depth=-np.ones((12,16)),
                                                       seg=np.tile(np.array([0,127,255,0],np.uint8),(12,4)),
                                                       cam_pose=np.eye(4))})
        writer.finalize('fixture',None,info,{'front':k})
    assert info == dict(image_size=(16,12),depth_scale=33333.34,cam_dist=-.022)
    return tmp_path, sample


def test_stock_reader_roundtrip_and_no_extension_fields(episodes):
    root, sample = episodes
    calibrations = []
    for ep in range(2):
        path = root/f'{ep:02d}'
        with open(path/'data.pkl','rb') as f:
            d = pickle.load(f)
        calibrations.append(d['digit_info'])
        assert 'camera_poses' not in d['allegro']
        assert 'gel_depth' not in d['digit_info']
        assert not (path/'allegro/index/depth.npz').exists()
        dataset = consumer('TactileDataset')(str(path/'allegro/index'),gt_depth=True)
        image, encoded = dataset[0]
        assert image.shape == (12,16,3)
        transform = consumer('DepthTransform')(d['digit_info']['cam_dist'])
        state = SimpleNamespace(scene_dataset=dataset, gt_depth=True, rgb_transform=lambda x:x,
                                depth_transform=lambda x: transform(x/d['digit_info']['depth_scale']),
                                device='cpu',sensor_name='digit_index')
        loaded = consumer('DigitSensor','get_frame_data')(state,0,np.eye(4)).depth_batch_np[0]
        _, expected = rasterize(sample,d['digit_info']['intrinsics'])
        np.testing.assert_allclose(loaded,expected,atol=.51/d['digit_info']['depth_scale'],equal_nan=True)
        # Filenames stay compatible; OpenCV decodes the lossless PNG payload.
        assert Image.open(path/'allegro/index/depth/0.jpg').format == 'PNG'
    assert calibrations[0] == calibrations[1]
    with pytest.raises(FileExistsError):
        EpisodeWriter(root,0)


def test_adaptive_encoding_does_not_clip_large_depth_ranges():
    points = np.array([[0,0,-.031],[0,0,-.008]],np.float32)
    frames = [{'index':dict(points=points,contact=np.ones(2,bool))}]
    scale, offset = depth_encoding(frames,33333.34)
    assert scale < 33333.34
    z = np.array([[-.031,-.008,np.nan]],np.float32)
    encoded, mask = encode_depth(z,scale,offset)
    decoded = consumer('DepthTransform')(offset)(encoded/scale)
    np.testing.assert_allclose(decoded,z,atol=.51/scale,equal_nan=True)
    assert mask[0,2] == 0


def test_stock_scene_reader(episodes):
    root, _ = episodes
    ds = consumer('VisionDataset')(str(root/'00/realsense/front'),gt_seg=True,sim_noise_iters=0)
    image, depth = ds[0]
    assert image.shape[:2] == depth.shape == (12,16)
    assert np.any(depth == -1)
    assert np.all(depth <= 0)


def test_zbuffer_keeps_nearest_observed_surface():
    sample = dict(points=np.array([[0,0,-.03],[0,0,-.01]],np.float32),
                  rgb=np.array([[255,0,0],[0,255,0]],np.uint8),contact=np.array([True,True]))
    rgb, depth = rasterize(sample,dict(w=8,h=6,fx=10.,fy=10.,cx=4.,cy=3.))
    np.testing.assert_array_equal(rgb[3,4],[0,255,0])
    assert depth[3,4] == np.float32(-.01)
    assert np.isfinite(depth).sum() == 1


def test_camera_offset_matches_stock_neuralfeels():
    from scipy.spatial.transform import Rotation
    from dataset_collection.conventions import NEURALFEELS_TIP_TO_CAMERA, NEURALFEELS_CAMERA_QUAT_WXYZ
    stock = consumer('Allegro', '_hora_to_neural')(None, np.eye(4))
    np.testing.assert_allclose(NEURALFEELS_TIP_TO_CAMERA, stock, atol=1e-12)
    w, x, y, z = NEURALFEELS_CAMERA_QUAT_WXYZ
    np.testing.assert_allclose(Rotation.from_quat([x,y,z,w]).as_matrix(), stock[:3,:3], atol=1e-12)


def test_direct_images_keep_full_frame_and_intrinsics(tmp_path):
    k = dict(w=24,h=32,fx=33.,fy=33.,cx=11.75,cy=15.75)
    rgb = np.full((32,24,3),150,np.uint8)
    depth = np.full((32,24),-.0205,np.float32)
    info = dict(intrinsics=k.copy(),depth_scale=33333.34,cam_dist=-.022)
    writer = EpisodeWriter(tmp_path,0)
    writer.add_step(t=0,object_pose=np.eye(4),finger_poses=np.repeat(np.eye(4)[None],4,axis=0),
                    joint_state=np.zeros(16),base_pose=np.eye(4),
                    tactile={f:dict(rgb=rgb,depth=depth) for f in FINGER_NAMES},
                    scene_frames={'front':dict(rgb=rgb,depth=-np.ones((32,24)),
                                              seg=np.zeros((32,24),np.uint8),cam_pose=np.eye(4))})
    writer.finalize('fixture',None,info,{'front':k})
    d=pickle.load(open(tmp_path/'00/data.pkl','rb'))
    assert d['digit_info']['intrinsics'] == k == info['intrinsics']
    ds=consumer('TactileDataset')(str(tmp_path/'00/allegro/index'),gt_depth=True)
    image, encoded = ds[0]
    assert image.shape == rgb.shape and image.min() >= 149
    restored=consumer('DepthTransform')(d['digit_info']['cam_dist'])(encoded/d['digit_info']['depth_scale'])
    np.testing.assert_allclose(restored,depth,atol=.51/d['digit_info']['depth_scale'])
