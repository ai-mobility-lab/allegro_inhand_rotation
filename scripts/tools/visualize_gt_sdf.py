# --------------------------------------------------------
# Debug viewer for the `gt_sdf_voxel=<voxel_size>.npz` files written by
# `scripts/dataset_collection/gt_sdf.py`'s `write_gt_sdf` (one per episode dir under
# `--output_dir` from `collect_stage1_feelsight_dataset.py` / `collect_stage2_feelsight_dataset.py`,
# plus the per-asset ones cached under `assets/ycb/**`).
#
# Each file holds exactly two arrays:
#   sdf : (Nx, Ny, Nz) float, signed distance in meters (negative = inside the object) on a
#         regular voxel grid, built by `sdf_from_mesh`'s occupancy -> EDT construction.
#   tf  : (4, 4) affine mapping a voxel *index* (i, j, k) -> a point in the object's own local
#         (root-prim) frame, i.e. `point = tf @ [i, j, k, 1]` (matches the grid's `pitch` scale
#         and padded origin offset -- see `gt_sdf.py`'s module docstring/`sdf_from_mesh`).
#
# This is a plain numpy/matplotlib/scikit-image/trimesh script -- no isaaclab/hora import, no
# `AppLauncher` -- so it runs under any Python env with those installed, independent of the
# IsaacLab conda env the collection scripts need.
#
# Usage:
#   python scripts/tools/visualize_gt_sdf.py data/feelsight_sim/dextouch_pear/00/gt_sdf_voxel=0.0005.npz
#   python scripts/tools/visualize_gt_sdf.py <path>.npz --mode mesh --export /tmp/obj.glb
#   python scripts/tools/visualize_gt_sdf.py <path>.npz --mode slices --axis x
#   python scripts/tools/visualize_gt_sdf.py <path>.npz --no-show --save-fig /tmp/sdf.png
# --------------------------------------------------------

import argparse
import re
from pathlib import Path

import numpy as np


def load_gt_sdf(npz_path: Path):
    data = np.load(npz_path)
    sdf, tf = data["sdf"], data["tf"]
    match = re.search(r"voxel=([0-9eE.+-]+)\.npz$", npz_path.name)
    voxel_size = float(match.group(1)) if match else float(np.linalg.norm(tf[:3, 0]))
    return sdf, tf, voxel_size


def print_summary(sdf: np.ndarray, tf: np.ndarray, voxel_size: float, npz_path: Path):
    corners_idx = np.array([[0, 0, 0], [s - 1 for s in sdf.shape]], dtype=float)
    corners_world = corners_idx @ tf[:3, :3].T + tf[:3, 3]
    print(f"[gt_sdf] {npz_path}")
    print(f"  grid shape        : {sdf.shape}  (voxel_size={voxel_size:g} m)")
    print(f"  sdf min/max        : {sdf.min():.5f} / {sdf.max():.5f} m")
    print(f"  occupied voxels    : {int((sdf < 0).sum())} / {sdf.size} ({(sdf < 0).mean() * 100:.1f}%)")
    print(f"  local-frame bounds : min={corners_world[0]}, max={corners_world[1]}")
    print("  tf (voxel idx -> object local frame):")
    print(np.array2string(tf, precision=5, suppress_small=True, prefix="    "))


def plot_slices(sdf: np.ndarray, voxel_size: float, axis: str, cmap: str):
    """Interactive slider through slices of `sdf` along `axis`, sdf value as a diverging
    heatmap (centered at 0 = surface) with the zero-level contour drawn on top."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    n_slices = sdf.shape[axis_idx]
    vmax = float(np.abs(sdf).max()) or 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.15)

    def slice_at(idx):
        return np.take(sdf, idx, axis=axis_idx)

    start = n_slices // 2
    im = ax.imshow(slice_at(start).T, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
    contour = [ax.contour(slice_at(start).T, levels=[0.0], colors="k", linewidths=1.5)]
    ax.set_title(f"{axis}-slice {start}/{n_slices - 1}  ({start * voxel_size * 1000:.1f} mm along axis origin)")
    fig.colorbar(im, ax=ax, label="signed distance (m)")

    slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.03])
    slider = Slider(slider_ax, "slice", 0, n_slices - 1, valinit=start, valstep=1)

    def update(val):
        idx = int(slider.val)
        data = slice_at(idx)
        im.set_data(data.T)
        for artist in list(ax.collections):
            artist.remove()
        contour[0] = ax.contour(data.T, levels=[0.0], colors="k", linewidths=1.5)
        ax.set_title(f"{axis}-slice {idx}/{n_slices - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    return fig


def extract_mesh(sdf: np.ndarray, tf: np.ndarray, voxel_size: float, level: float = 0.0):
    """Zero-level-set surface of `sdf` as a `trimesh.Trimesh`, mapped into the object's local
    frame via `tf` -- the same convention `gt_sdf.py` used to build `tf` in the first place, so
    this should reproduce (a discretized version of) the original ground-truth mesh."""
    import trimesh
    from skimage import measure

    if not (sdf.min() < level < sdf.max()):
        raise ValueError(f"level={level} is outside the sdf's value range [{sdf.min():.5f}, {sdf.max():.5f}]")

    verts_idx, faces, normals, _ = measure.marching_cubes(sdf, level=level)
    verts_local = trimesh.transform_points(verts_idx, tf)
    return trimesh.Trimesh(vertices=verts_local, faces=faces, vertex_normals=normals, process=False)


def plot_mesh(mesh, title: str):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    tris = mesh.vertices[mesh.faces]
    coll = Poly3DCollection(tris, facecolor=(0.6, 0.7, 0.9), edgecolor="k", linewidths=0.1, alpha=0.9)
    ax.add_collection3d(coll)
    bounds_min, bounds_max = mesh.bounds
    center = (bounds_min + bounds_max) / 2
    radius = float(np.max(bounds_max - bounds_min)) / 2 or 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    return fig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_path", type=Path, help="a gt_sdf_voxel=<voxel_size>.npz file")
    parser.add_argument("--mode", choices=["slices", "mesh", "both"], default="both")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z", help="slice axis for --mode slices")
    parser.add_argument("--level", type=float, default=0.0, help="iso-surface level (m) for --mode mesh, 0 = the object surface")
    parser.add_argument("--cmap", default="RdBu", help="matplotlib colormap for the slice heatmap")
    parser.add_argument("--export", type=Path, default=None, help="also write the extracted mesh here (.obj/.glb/.ply/...)")
    parser.add_argument("--interactive-3d", action="store_true", help="open an interactive trimesh/pyglet viewer instead of a static matplotlib 3D plot")
    parser.add_argument("--save-fig", type=Path, default=None, help="save figure(s) here instead of / in addition to showing them (suffix _slices/_mesh is added when both are produced)")
    parser.add_argument("--no-show", action="store_true", help="don't call plt.show() (useful headless, combine with --save-fig)")
    return parser


def main():
    args = build_arg_parser().parse_args()

    if not args.npz_path.is_file():
        raise SystemExit(f"no such file: {args.npz_path}")

    sdf, tf, voxel_size = load_gt_sdf(args.npz_path)
    print_summary(sdf, tf, voxel_size, args.npz_path)

    mesh = None
    if args.mode in ("mesh", "both"):
        mesh = extract_mesh(sdf, tf, voxel_size, level=args.level)
        print(f"[gt_sdf] extracted mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
        if args.export:
            mesh.export(args.export)
            print(f"[gt_sdf] wrote mesh to {args.export}")
        if args.interactive_3d:
            mesh.show()

    figs = []
    if args.mode in ("slices", "both"):
        figs.append(("slices", plot_slices(sdf, voxel_size, args.axis, args.cmap)))
    if args.mode in ("mesh", "both") and not args.interactive_3d:
        figs.append(("mesh", plot_mesh(mesh, f"{args.npz_path.name} (level={args.level})")))

    if args.save_fig:
        for name, fig in figs:
            out = args.save_fig if len(figs) == 1 else args.save_fig.with_stem(f"{args.save_fig.stem}_{name}")
            fig.savefig(out, dpi=150)
            print(f"[gt_sdf] saved figure to {out}")

    if figs and not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
