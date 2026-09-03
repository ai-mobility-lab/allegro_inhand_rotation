#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.
# --------------------------------------------------------
# Ported from IsaacGym to IsaacLab (2026): this is a standalone visualizer, not
# an RL task, so it drives the two hand articulations directly via
# `write_joint_state_to_sim` every frame instead of going through
# `AllegroHandHoraEnv`. `AppLauncher` must run before any `isaaclab`-touching
# import -- see hora/tasks/isaaclab/allegro_hand_hora_env.py for the same
# constraint applied to the RL task, whose `REPO_ROOT`/`_stable_usd_cache_dir`
# helpers are reused below for URDF->USD conversion caching.
# --------------------------------------------------------

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare different URDFs (internal vs public)")
parser.add_argument("--speed_scale", type=float, default=1.0, help="Animation speed scale")
parser.add_argument("--show_axis", action="store_true", help="Visualize DOF axis")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- everything below may safely import isaaclab ----

import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply

from hora.tasks.isaaclab.allegro_hand_hora_env import REPO_ROOT, _stable_usd_cache_dir


# -----------------------------------------------------------------------------
# Use two URDF files: one for each agent.
# -----------------------------------------------------------------------------
class AssetDesc:
    def __init__(self, file_name):
        self.file_name = file_name


asset_descriptors = [
    AssetDesc("assets/allegro/allegro_right.urdf"),
    AssetDesc("assets/allegro/allegro_hora.urdf"),
]


# -----------------------------------------------------------------------------
# Parse each URDF's revolute joints for `--show_axis`: IsaacLab's `Articulation`
# doesn't expose a per-joint world-space frame the way `gym.get_dof_frame` did,
# so the (local axis, child link) pairs are read straight out of the URDF and
# the axis is rotated into world space using the child link's current pose.
# -----------------------------------------------------------------------------
def parse_joint_axes(urdf_path: str) -> dict:
    root = ET.parse(urdf_path).getroot()
    info = {}
    for joint in root.findall("joint"):
        if joint.get("type") not in ("revolute", "continuous"):
            continue
        axis_el = joint.find("axis")
        axis = tuple(float(v) for v in axis_el.get("xyz").split()) if axis_el is not None else (1.0, 0.0, 0.0)
        info[joint.get("name")] = {"axis": axis, "child": joint.find("child").get("link")}
    return info


# -----------------------------------------------------------------------------
# Spawn one hand articulation, tinted with a flat debug color so the two
# assets are visually distinguishable when overlapping.
# -----------------------------------------------------------------------------
def spawn_hand(prim_path: str, urdf_relpath: str, pos: tuple, color: tuple) -> Articulation:
    asset_path = os.path.join(REPO_ROOT, urdf_relpath)
    robot_cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=asset_path,
            usd_dir=_stable_usd_cache_dir(asset_path),
            fix_base=True,
            merge_fixed_joints=True,
            # `target_type="none"` documents that it forces stiffness/damping to 0.0 at
            # conversion time, but `gains.stiffness` still has no default and fails
            # `cfg.validate()` if left unset -- so set it explicitly too.
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            # DOF positions are teleported directly every frame, and env0/env1's two
            # hands sit on top of each other by design (overlap comparison) -- collision
            # would just add spurious contact forces/jitter with no visual benefit.
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        # `joint_12_0` (thumb base/CMC) is the only one of the 16 joints whose range,
        # [0.263, 1.396], excludes 0 -- same fix as `allegro_hand_hora_env.py`'s
        # `robot_cfg.init_state.joint_pos`.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos, joint_pos={"^(?!joint_12_0$).*$": 0.0, "joint_12_0": 1.104}
        ),
        actuators={"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
    robot = Articulation(robot_cfg)

    material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    material_path = f"{prim_path}/DebugColor"
    material_cfg.func(material_path, material_cfg)
    sim_utils.bind_visual_material(prim_path, material_path)

    return robot


# -----------------------------------------------------------------------------
# Extract and compute DOF sweep-animation info from an articulation.
# -----------------------------------------------------------------------------
def get_dof_info(robot: Articulation, speed_scale: float, asset_name: str) -> dict:
    num_dofs = robot.num_joints
    dof_names = robot.joint_names
    limits = robot.data.joint_pos_limits[0].cpu().numpy()
    stiffnesses = robot.data.joint_stiffness[0].cpu().numpy()
    dampings = robot.data.joint_damping[0].cpu().numpy()
    armatures = robot.data.joint_armature[0].cpu().numpy()

    # All of this repo's Allegro URDFs are pure revolute joints with explicit
    # <limit> tags -- unlike the original (a generic IsaacGym asset viewer), the
    # unlimited/prismatic-DOF branches never trigger here, so they're dropped.
    lower_limits = np.clip(limits[:, 0], -math.pi, math.pi)
    upper_limits = np.clip(limits[:, 1], -math.pi, math.pi)

    defaults = np.zeros(num_dofs)
    for i in range(num_dofs):
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]

    dof_positions = defaults.copy()
    speeds = speed_scale * np.clip(2 * (upper_limits - lower_limits), 0.25 * math.pi, 3.0 * math.pi)

    for i in range(num_dofs):
        print(f"Asset {asset_name} DOF {i}")
        print(f"  Name:     '{dof_names[i]}'")
        print(f"  Stiffness:  {stiffnesses[i]!r}")
        print(f"  Damping:    {dampings[i]!r}")
        print(f"  Armature:   {armatures[i]!r}")
        print(f"    Lower:   {lower_limits[i]:f}")
        print(f"    Upper:   {upper_limits[i]:f}")

    return {
        "num_dofs": num_dofs,
        "dof_names": dof_names,
        "dof_positions": dof_positions,
        "lower_limits": lower_limits,
        "upper_limits": upper_limits,
        "defaults": defaults,
        "speeds": speeds,
    }


def dof_axis_line(robot: Articulation, dof_name: str, joint_axes: dict):
    axis_info = joint_axes[dof_name]
    body_idx = robot.body_names.index(axis_info["child"])
    origin = robot.data.body_pos_w[0, body_idx]
    quat = robot.data.body_quat_w[0, body_idx]
    axis_local = torch.tensor(axis_info["axis"], device=origin.device, dtype=origin.dtype)
    axis_world = quat_apply(quat.unsqueeze(0), axis_local.unsqueeze(0)).squeeze(0)
    return origin.tolist(), (origin + axis_world * 0.7).tolist()


def step_animation(state: dict, dof_info: dict):
    """Advance one DOF through lower -> upper -> default, then move to the next DOF."""
    i = state["current_dof"]
    speed = dof_info["speeds"][i]
    if state["anim_state"] == ANIM_SEEK_LOWER:
        dof_info["dof_positions"][i] -= speed * dt
        if dof_info["dof_positions"][i] <= dof_info["lower_limits"][i]:
            dof_info["dof_positions"][i] = dof_info["lower_limits"][i]
            state["anim_state"] = ANIM_SEEK_UPPER
    elif state["anim_state"] == ANIM_SEEK_UPPER:
        dof_info["dof_positions"][i] += speed * dt
        if dof_info["dof_positions"][i] >= dof_info["upper_limits"][i]:
            dof_info["dof_positions"][i] = dof_info["upper_limits"][i]
            state["anim_state"] = ANIM_SEEK_DEFAULT
    elif state["anim_state"] == ANIM_SEEK_DEFAULT:
        dof_info["dof_positions"][i] -= speed * dt
        if dof_info["dof_positions"][i] <= dof_info["defaults"][i]:
            dof_info["dof_positions"][i] = dof_info["defaults"][i]
            state["anim_state"] = ANIM_FINISHED
    elif state["anim_state"] == ANIM_FINISHED:
        dof_info["dof_positions"][i] = dof_info["defaults"][i]
        state["current_dof"] = (i + 1) % dof_info["num_dofs"]
        state["anim_state"] = ANIM_SEEK_LOWER
        print(f"Animating DOF {state['current_dof']} ('{dof_info['dof_names'][state['current_dof']]}')")


# -----------------------------------------------------------------------------
# Initialize the simulation.
# -----------------------------------------------------------------------------
dt = 1.0 / 60.0
sim_cfg = sim_utils.SimulationCfg(dt=dt, device=args_cli.device)
sim = SimulationContext(sim_cfg)

sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
light_cfg.func("/World/Light", light_cfg)

# -----------------------------------------------------------------------------
# Create two groups:
# - Group 0: both hands at the same position (overlapping)
# - Group 1: hands at different positions for easier comparison
# -----------------------------------------------------------------------------
print("Creating 2 groups with 2 actors each")

right_env0 = spawn_hand("/World/Env0/Right", asset_descriptors[0].file_name, (0.0, 0.0, 0.3), (0.3, 0.5, 1.0))
hora_env0 = spawn_hand("/World/Env0/Hora", asset_descriptors[1].file_name, (0.0, 0.0, 0.3), (1.0, 0.5, 0.2))
right_env1 = spawn_hand("/World/Env1/Right", asset_descriptors[0].file_name, (1.5, 0.3, 0.3), (0.3, 0.5, 1.0))
hora_env1 = spawn_hand("/World/Env1/Hora", asset_descriptors[1].file_name, (1.5, -0.3, 0.3), (1.0, 0.5, 0.2))

sim.set_camera_view((1.8, -0.5, 0.5), (0.5, 0, 0.2))

sim.reset()

# -----------------------------------------------------------------------------
# Obtain DOF information for each asset (identical for env0/env1's copy of the
# same asset -- only queried once per asset, not once per actor instance).
# -----------------------------------------------------------------------------
dof_info0 = get_dof_info(right_env0, args_cli.speed_scale, asset_name=asset_descriptors[0].file_name)
dof_info1 = get_dof_info(hora_env0, args_cli.speed_scale, asset_name=asset_descriptors[1].file_name)

if dof_info0["num_dofs"] != dof_info1["num_dofs"]:
    print(f"Error: The two assets have different DOF counts: {dof_info0['num_dofs']} vs {dof_info1['num_dofs']}")
    sys.exit(1)

joint_axes0 = parse_joint_axes(os.path.join(REPO_ROOT, asset_descriptors[0].file_name))
joint_axes1 = parse_joint_axes(os.path.join(REPO_ROOT, asset_descriptors[1].file_name))

draw_interface = None
if args_cli.show_axis:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

    draw_interface = omni_debug_draw.acquire_debug_draw_interface()

# -----------------------------------------------------------------------------
# Joint animation state.
# -----------------------------------------------------------------------------
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

state0 = {"anim_state": ANIM_SEEK_LOWER, "current_dof": 0}
state1 = {"anim_state": ANIM_SEEK_LOWER, "current_dof": 0}
print(f"Animating asset0 DOF 0 ('{dof_info0['dof_names'][0]}')")
print(f"Animating asset1 DOF 0 ('{dof_info1['dof_names'][0]}')")

# -----------------------------------------------------------------------------
# Simulation loop.
# -----------------------------------------------------------------------------
sim_dt = sim.get_physics_dt()
all_robots = [right_env0, hora_env0, right_env1, hora_env1]

while simulation_app.is_running():
    step_animation(state0, dof_info0)
    step_animation(state1, dof_info1)

    pos0 = torch.tensor(dof_info0["dof_positions"], dtype=torch.float32, device=sim.device).unsqueeze(0)
    pos1 = torch.tensor(dof_info1["dof_positions"], dtype=torch.float32, device=sim.device).unsqueeze(0)
    vel0 = torch.zeros_like(pos0)
    vel1 = torch.zeros_like(pos1)
    for robot in (right_env0, right_env1):
        robot.write_joint_state_to_sim(pos0, vel0)
    for robot in (hora_env0, hora_env1):
        robot.write_joint_state_to_sim(pos1, vel1)

    if draw_interface is not None:
        draw_interface.clear_lines()
        points1, points2 = [], []
        for right_robot, hora_robot in ((right_env0, hora_env0), (right_env1, hora_env1)):
            p1, p2 = dof_axis_line(right_robot, dof_info0["dof_names"][state0["current_dof"]], joint_axes0)
            points1.append(p1)
            points2.append(p2)
            p1, p2 = dof_axis_line(hora_robot, dof_info1["dof_names"][state1["current_dof"]], joint_axes1)
            points1.append(p1)
            points2.append(p2)
        colors = [[1.0, 0.0, 0.0, 1.0]] * len(points1)
        sizes = [3.0] * len(points1)
        draw_interface.draw_lines(points1, points2, colors, sizes)

    for robot in all_robots:
        robot.write_data_to_sim()
    sim.step()
    for robot in all_robots:
        robot.update(sim_dt)

print("Done")
simulation_app.close()
