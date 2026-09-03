# --------------------------------------------------------
# Ground-truth signed-distance-field export, matching NeuralFeels' own
# `gt_sdf_voxel=<voxel_size>.npz` files (see e.g.
# neuralfeels/data/feelsight_real/large_dice/02/gt_sdf_voxel=0.0005.npz`, loaded by
# `neuralfeels.modules.trainer.Trainer.load_gt_sdf`).
#
# `voxelize_subdivide`/`sdf_from_occupancy`/`sdf_from_mesh` are a direct port of
# `neuralfeels/datasets/sdf_util.py`'s functions of the same name (trimesh + scipy only,
# no neuralfeels import) -- keeping them bit-for-bit identical is what makes the `sdf`/`tf`
# arrays this writes out load-compatible with `Trainer.load_gt_sdf`, which expects exactly
# that occupancy -> EDT -> metric-scaled construction and the voxel-to-object `tf`.
#
# The manipulated objects here are native IsaacLab shape prims (Cuboid/Cylinder/Sphere,
# see `AllegroHandHoraEnv._build_object_cfg`), not a mesh file, so the ground-truth mesh is
# built straight from that spawn cfg instead of `sdf_util.load_gt_mesh`'s URDF path.
# --------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage


def trimesh_from_shape_cfg(shape_cfg) -> trimesh.Trimesh:
    """Ground-truth mesh for one `CuboidCfg`/`CylinderCfg`/`SphereCfg`, in the object's own
    local (root-prim) frame -- these primitives spawn centered on their prim's origin with
    no extra offset, matching the tracked `object.pose` (the RigidObject's root pose)."""
    import isaaclab.sim as sim_utils

    if isinstance(shape_cfg, sim_utils.CuboidCfg):
        return trimesh.creation.box(extents=shape_cfg.size)
    if isinstance(shape_cfg, sim_utils.CylinderCfg):
        mesh = trimesh.creation.cylinder(radius=shape_cfg.radius, height=shape_cfg.height)
        axis_rotation = {
            "Z": None,
            "X": trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]),
            "Y": trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]),
        }[shape_cfg.axis]
        if axis_rotation is not None:
            mesh.apply_transform(axis_rotation)
        return mesh
    if isinstance(shape_cfg, sim_utils.SphereCfg):
        return trimesh.creation.icosphere(radius=shape_cfg.radius)
    raise TypeError(f"no ground-truth mesh builder for object spawn cfg type {type(shape_cfg)}")


def voxelize_subdivide(mesh, pitch, origin_voxel=np.zeros(3), max_iter=10, edge_factor=2.0):
    """Voxelize a surface by subdividing a mesh until every edge is shorter than
    `pitch / edge_factor`. Adapted from trimesh's own `voxelize_subdivide` to allow shifting
    the voxel grid's origin (i.e. it doesn't need a voxel centered at [0, 0, 0])."""
    max_edge = pitch / edge_factor

    if max_iter is None:
        longest_edge = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=1).max()
        max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0)

    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=max_iter)

    hit = (v - origin_voxel) / pitch
    hit = np.round(hit).astype(int)

    unique, inverse = trimesh.grouping.unique_rows(hit)
    occupied_index = hit[unique]

    origin_index = occupied_index.min(axis=0)
    origin_position = origin_voxel + origin_index * pitch

    return trimesh.voxel.base.VoxelGrid(
        trimesh.voxel.encoding.SparseBinaryEncoding(occupied_index - origin_index),
        transform=trimesh.transformations.scale_and_translate(scale=pitch, translate=origin_position),
    )


def sdf_from_occupancy(occ_map, voxel_size):
    inv_occ_map = 1 - occ_map
    map_dist = ndimage.distance_transform_edt(inv_occ_map)
    inv_map_dist = ndimage.distance_transform_edt(occ_map)
    sdf = (map_dist - inv_map_dist).astype(float) * voxel_size
    return sdf


def sdf_from_mesh(mesh, voxel_size, extend_factor=0.15, origin_voxel=np.zeros(3)):
    voxels = voxelize_subdivide(mesh, voxel_size, origin_voxel=origin_voxel)
    voxels = voxels.fill()
    occ_map = voxels.matrix
    transform = voxels.transform

    extend = np.array(occ_map.shape) * extend_factor
    extend = np.repeat(extend, 2).reshape(3, 2)
    extend = np.round(extend).astype(int)
    occ_map = np.pad(occ_map, extend)
    transform[:3, 3] -= extend[:, 0] * voxel_size

    sdf = sdf_from_occupancy(occ_map, voxel_size)
    return sdf, np.array(transform)


def write_gt_sdf(episode_dir: Path, mesh: trimesh.Trimesh, voxel_size: float = 5e-4, extend_factor: float = 0.15) -> Path:
    """Write `<episode_dir>/gt_sdf_voxel=<voxel_size>.npz` (keys `sdf`, `tf`)."""
    sdf, tf = sdf_from_mesh(mesh, voxel_size, extend_factor=extend_factor, origin_voxel=np.zeros(3))
    out_path = Path(episode_dir) / f"gt_sdf_voxel={voxel_size}.npz"
    np.savez(out_path, sdf=sdf, tf=tf)
    return out_path
