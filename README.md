# allegro_inhand_rotation

Reference Implementation of In-Hand Object Rotation for Allegro Hand Platforms

This repository provides an implementation example for in-hand object rotation using the **Allegro Hand Platforms**.
It combines a ROS2-based hardware controller with an AI-driven manipulation algorithm originally developed for in-hand rotation research, trained and simulated in **Isaac Lab**.

This implementation currently supports **Allegro Hand V4** — both the standard hand and a **left-hand variant fitted with DIGIT tactile fingertips** — but the software architecture is designed to be modular and extendable.
As additional robotic hand platforms are developed within our organization, this codebase may be expanded to include **plug-in modules and adapters** for new hardware versions, enabling broader compatibility across future Wonik Robotics hand systems.

This codebase utilizes:

- **Allegro Hand ROS2 Controller**
  https://github.com/Wonikrobotics-git/allegro_hand_ros2
- **AI Algorithm: In-Hand Object Rotation via Rapid Motor Adaptation (RMA)**
  Original research & implementation by Haozhi Qi
  https://haozhi.io/hora/
- **Simulation: Isaac Lab**
  https://isaac-sim.github.io/IsaacLab/

This project is forked from [Wonikrobotics-git/allegro_inhand_rotation](https://github.com/Wonikrobotics-git/allegro_inhand_rotation). On top of that base, the simulation backend was ported from IsaacGym to Isaac Lab, a DIGIT tactile-fingertip left-hand variant was added, and tooling was added to collect synthetic tactile datasets from trained policies. See the [License](#license) section for full third-party attribution.

## Test System Configuration

- Ubuntu 22.04
- ROS2 Humble
- Allegro Hand V4 (standard or DIGIT tactile fingertip variant)
- Isaac Lab (tested with 2.3.2)

## System Requirements

### 1. Isaac Lab Installation

Follow the official installation guide: [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

Isaac Lab bundles Isaac Sim and sets up its own Python (3.10+) environment for you, e.g.:

```bash
cd /path/to/IsaacLab
./isaaclab.sh --conda        # creates conda env `env_isaaclab` by default
conda activate env_isaaclab
```

### 2. Allegro Hand ROS2 Controller

This project operates based on the official controller:

**👉 [WonikRobotics-git/allegro_hand_ros2](https://github.com/WonikRobotics-git/allegro_hand_ros2)**

Refer to the official repository for detailed installation and setup instructions. You will need this for real-world deployment.

### 3. Python Environment Setup

We use two environments to keep the heavy Isaac Lab/Isaac Sim stack separate from the ROS 2 stack:

#### Environment 1: `env_isaaclab` (For Training in Isaac Lab)

```bash
# Created above via `./isaaclab.sh --conda`
conda activate env_isaaclab

# Install this repo's training dependencies on top of Isaac Lab
pip install -r hora_isaaclab_requirements.txt
```

**Use this environment for:**
- Training policies in simulation (`train.py`)
- Generating grasp poses (`gen_grasp.py`)
- Visualizing/comparing hand URDFs (`allegro_right_left.py`, `compare_hands.py`)
- Collecting synthetic tactile datasets (`scripts/collect_feelsight_dataset.sh`)

#### Environment 2: `allegro` (For Real-world Deployment)

```bash
# Create environment with Python 3.10
conda create -n allegro python=3.10

# Activate environment
conda activate allegro

# Install deployment dependencies
pip install -r allegro_requirements.txt
```

**Use this environment for:**
- Deploying trained policies to physical Allegro Hand hardware
- Running ROS 2 nodes and controllers
- Running `deploy_one_hand.sh`, `deploy_two_hands.sh`, etc.

### 4. Verify Installation

After setting up both environments, verify they work correctly:

#### Check 1: `env_isaaclab` Environment (Training)

Verify Isaac Lab and dependencies are installed correctly:

```bash
conda activate env_isaaclab
python compare_hands.py
```

**Expected output:** An Isaac Sim window will open showing both hand versions side-by-side, comparing the fingertip geometry of Allegro Hand V4 against the fingertip geometry used in the original HORA implementation.

<p align="center">
  <img src="./materials/compare.gif" width="60%" />
</p>

#### Check 2: `allegro` Environment (Deployment)

> **Note:** This check requires physical Allegro Hand hardware and ROS 2 setup. Skip if you only want to train in simulation.

**Step 1:** Launch the ROS 2 hand controller (in a separate terminal):

```bash
# Navigate to your allegro_hand_ros2 workspace
cd /path/to/allegro_hand_ros2_ws
source install/setup.bash

# Launch controller
ros2 launch allegro_hand_bringup allegro_hand.launch.py
```

**Step 2:** Test the deployment interface:

```bash
conda activate allegro
python hora/algo/deploy/robots/allegro_ros2.py
```


## Run

> **Note:** This repository focuses on **Allegro Hand V4 (Right and Left)**, including a **Left + DIGIT tactile fingertip** variant. The original HORA repository used different fingertip geometries, resulting in slightly different finger lengths. See the comparison images below for details.

### Verify Allegro Right/Left URDF

To verify the URDF configurations for both hands, you can visualize them in Isaac Lab:

```bash
python allegro_right_left.py
```

This script loads both the right and left hand models in a single environment, allowing you to compare their kinematics and collision geometries side by side. Pass `--show_axis` to visualize joint axes, or `--headless` to run without a viewer.

**URDF Files Location:**
- Right hand: `assets/allegro/allegro_right.urdf`
- Left hand: `assets/allegro/allegro_left.urdf`
- Left hand + DIGIT tactile fingertips: `assets/allegro/allegro_digit_left_elastomer.urdf`

**Visualization:**

<p align="center">
  <img src="./materials/allegro_right_left.gif" width="60%" alt="Allegro Right and Left Hands Comparison"/>
</p>

**Fingertip Geometry Comparison:**

<p align="center">
  <img src="./materials/hand_tips.png" width="60%" alt="Hand Fingertip Comparison"/>
</p>

The images above show the differences between the original HORA fingertips and the standard Allegro Hand V4 fingertips used in this repository.

### Configuration Structure

This repository uses [Hydra](https://hydra.cc/) for hierarchical configuration management. Configs are organized in `configs/` directory:

- **`config.yaml`** - Main entry point (sets device, physics engine, defaults)
- **`task/*.yaml`** - Environment settings (rewards, randomization, URDF paths)
- **`train/*.yaml`** - Training parameters (PPO hyperparameters, network architecture)

Task environments are implemented in `hora/tasks/isaaclab/` as Isaac Lab `DirectRLEnv`s and registered by name in `hora/tasks/__init__.py`'s `isaaclab_task_map`.

#### Configuration Inheritance Diagram

```mermaid
graph TD
    A[config.yaml] -->|loads| B[task/AllegroHandHora.yaml]
    A -->|loads| C[train/AllegroHandHora.yaml]

    B -->|inherits| D[task/AllegroHandGrasp.yaml]
    B -->|inherits| E[task/RightAllegroHandHora.yaml]
    B -->|inherits| F[task/LeftAllegroHandHora.yaml]
    F -->|inherits| N[task/LeftAllegroHandDigitHora.yaml]
    E -->|inherits| G[task/RightAllegroHandGrasp.yaml]
    F -->|inherits| H[task/LeftAllegroHandGrasp.yaml]
    N -->|inherits| O[task/LeftAllegroHandDigitGrasp.yaml]

    C -->|mirrors| I[train/AllegroHandGrasp.yaml]
    C -->|mirrors| J[train/RightAllegroHandHora.yaml]
    C -->|mirrors| K[train/LeftAllegroHandHora.yaml]
    K -->|mirrors| P[train/LeftAllegroHandDigitHora.yaml]
    J -->|mirrors| L[train/RightAllegroHandGrasp.yaml]
    K -->|mirrors| M[train/LeftAllegroHandGrasp.yaml]
    P -->|mirrors| Q[train/LeftAllegroHandDigitGrasp.yaml]

    style B fill:#e1f5ff
    style C fill:#ffe1f5
    style E fill:#e1ffe1
    style F fill:#e1ffe1
    style N fill:#fff3cd
```

**Config Types:**
- **Hora** = In-hand rotation (training/testing)
- **Grasp** = Grasp pose generation only
- **Right/Left** = Hand-specific URDF and grasp caches
- **Digit** = Left hand fitted with DIGIT tactile fingertips (`link_*_tip_elastomer` contact links instead of `link_*_tip`)

**Usage:**
```bash
# Default (AllegroHandHora)
python train.py

# Specific task with overrides
python train.py task=LeftAllegroHandDigitHora train.ppo.learning_rate=1e-4
```

### Generate Grasping Poses

To achieve a stable initial grasp, you must prepare reliable grasp poses for the target objects.

**Download pre-generated grasp poses:**

1. Download the grasp pose files from [HuggingFace](https://huggingface.co/datasets/Wonik-Robotics/allegro_inhand_rotation)
2. Extract and place the `cache/` folder in the project root directory

Your directory structure should look like:
```
allegro_inhand_rotation/
├── cache/              # Downloaded grasp poses
├── configs/
├── hora/
└── ...
```

Alternatively, you can generate grasp poses **from scratch** using the scripts included in this repository. By default these target the DIGIT-fitted left hand (`task=LeftAllegroHandDigitGrasp`), sweeping a range of object scales:

```bash
scripts/gen_grasp.sh 0 # GPU ID
```

This script will run the full grasp-pose generation pipeline and produce the necessary `.npy` files (named after each task's `grasp_cache_name`, e.g. `allegro_digit_left`, `allegro_left`, `allegro_right`) for training or evaluation.

If you have multiple gpus, you can parallelize the process by running multiple instances with different GPU IDs:

```bash
scripts/gen_grasp_multigpus.sh 0 1 2
```


### Train

The training pipeline follows a two-stage approach using **Rapid Motor Adaptation (RMA)** with support for various object shapes (Ball, Cylinder, Cube, etc.).

**Training Stages:**
- **Stage 1**: Teacher policy with privileged observations (object dynamics, external forces)
- **Stage 2**: Student policy using only proprioceptive observations (joint positions, velocities, history)

<p align="center">
  <img src="./materials/training_stages.png" width="80%" alt="Training Process"/>
</p>

> **Note**: `scripts/train_s1.sh`/`train_s2.sh` default to **LeftAllegroHandDigitHora**. To train a different hand, override the `task` parameter, e.g. `task=RightAllegroHandHora` or `task=LeftAllegroHandHora`.

#### Stage 1: Teacher Policy Training

Train the teacher policy with privileged information:

```bash
./scripts/train_s1.sh 0 42 my_experiment
# Arguments: GPU_ID SEED RUN_NAME
```

#### Stage 2: Student Policy Training (Adaptation)

Train the student policy using proprioceptive adaptation:

```bash
./scripts/train_s2.sh 0 42 my_experiment
# Arguments: GPU_ID SEED RUN_NAME
```


### Test in Simulation

After training, you can test your policy using two methods: **evaluation** (quantitative metrics) and **visualization** (qualitative inspection).

#### Evaluation (Headless)

Runs 10,240 parallel environments in headless mode to measure success rates and performance metrics. All domain randomizations are enabled for robust testing.

```bash
# Stage 1 (Teacher policy)
./scripts/eval_s1.sh 0 my_experiment  # GPU_ID RUN_NAME

# Stage 2 (Student policy)
./scripts/eval_s2.sh 0 my_experiment  # GPU_ID RUN_NAME
```


#### Visualization (Visual Inspection)

Renders 64 environments with GUI to visually inspect policy behavior. Most randomizations are disabled for clearer observation.

```bash
# Stage 1 (Teacher policy)
./scripts/vis_s1.sh my_experiment  # RUN_NAME

# Stage 2 (Student policy)
./scripts/vis_s2.sh my_experiment  # RUN_NAME
```

<p align="center">
  <img src="./materials/vis.gif" width="60%"/>
</p>

#### Debugging

Enable debugging tools to visualize policy behavior and save action data for analysis.

**Enable in `configs/task/AllegroHandHora.yaml`:**

```yaml
env:
  enableDebugPlots: True        # Visualize DOF trajectories (PNG plots)
  enableActionRecording: True   # Save action history (NPZ file)
```

**Output** (saved to `debug/` directory):
- `obs_debug_*.png`, `allegro_debug_*.png` - Joint trajectories and commands
- `actions_500.npz` - First 500 actions from environment 0

### Collect a Synthetic Tactile Dataset (feelsight)

For the DIGIT tactile fingertip hand, a trained Stage 2 policy can be rolled out in Isaac Lab to record a synthetic tactile dataset (camera + tactile + ground-truth SDF) in the feelsight format, for downstream tactile-perception work:

```bash
scripts/collect_feelsight_dataset.sh
```

This runs `scripts/collect_stage2_feelsight_dataset.py` through Isaac Lab's own launcher (so that `--enable_cameras` is available) against a `LeftAllegroHandDigitHora` Stage 2 checkpoint, and writes episodes to `data/feelsight_sim/`. Edit the script to point `--checkpoint` at your own trained model, and set `OBJECT_TYPE` to the object to manipulate (`sphere`, `cylinder`, `cuboid`, ...). The recording/writing logic lives in `scripts/dataset_collection/` (`sensors.py`, `gt_sdf.py`, `feelsight_writer.py`).

### Test in Real-world

Deploy your trained policy to physical Allegro Hand hardware. This requires switching to the `allegro` conda environment (Python 3.10+) for ROS 2 compatibility.

#### Prerequisites

Before starting, ensure:
- Allegro Hand(s) connected via USB and powered on
- CAN interface hardware properly installed
- [allegro_hand_ros2](https://github.com/WonikRobotics-git/allegro_hand_ros2) package installed and built
- ROS 2 workspace sourced (`source install/setup.bash`)
- `allegro` conda environment activated (`conda activate allegro`)

#### Step 1: CAN Network Setup

Configure CAN bus interface for hand communication. The bitrate must be set to 1,000,000 for Allegro Hand V4.

**Single Hand (can0 only):**

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

**Dual Hand (can0 + can1):**

```bash
# Right hand on can0
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# Left hand on can1
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up
```

**Verify CAN connection:**
```bash
candump can0  # Should show periodic CAN messages if hand is connected
```

#### Step 2: Launch ROS 2 Hand Controller

Start the ROS 2 controller node that manages hand hardware communication.

**Configure PD Gains:**

Before launching the controller, configure the PD gains for optimal performance with this deployment. Edit the PD gains configuration file in your `allegro_hand_ros2` workspace:

```yaml
# File: allegro_hand_ros2/allegro_hand_hardwares/v4/description/config/pd_gains.yaml
p_gains:
  joint00: 3.6
  ...
  joint33: 3.6
d_gain: 
  joint00: 0.124
  ...
  joint33: 0.124
```

Reference configuration: [pd_gains.yaml](https://github.com/Wonikrobotics-git/allegro_hand_ros2/blob/main/allegro_hand_hardwares/v4/description/config/pd_gains.yaml)

**Single Hand:**

```bash
ros2 launch allegro_hand_bringup allegro_hand.launch.py
```


**Dual Hand:**

```bash
ros2 launch allegro_hand_bringup allegro_hand_duo.launch.py
```


> [!IMPORTANT]
> Controller command topics differ based on setup:
> - **Single hand:** `allegro_hand_position_controller/commands`
> - **Dual hands:** `allegro_hand_position_controller_r/commands` and `allegro_hand_position_controller_l/commands`

> **Note:** Keep this terminal running. Open a new terminal for the next step.

#### Step 3: Deploy HORA Algorithm

> [!IMPORTANT]
> Make sure you have activated the `allegro` conda environment before running deployment scripts:
> ```bash
> conda activate allegro
> ```

Run the trained policy on the physical hardware. The deployment script loads Stage 2 (student) checkpoints and executes the policy in real-time.



**Single Hand:**

Since previous training examples used `RightAllegroHandHora`, the deploy script defaults to loading from that directory:

```bash
scripts/deploy_one_hand.sh my_experiment
# Loads: outputs/RightAllegroHandHora/my_experiment/stage2_nn/best.pth
```

<p align="center">
  <img src="./materials/hw_deploy.gif" width="60%" />
</p>

**Dual Hand:**

For dual hand deployment, specify checkpoint names for both hands. Each hand loads from its respective training directory:

```bash
# Different experiments for each hand
scripts/deploy_two_hands.sh exp_right exp_left
# Right: outputs/RightAllegroHandHora/exp_right/stage2_nn/best.pth
# Left:  outputs/LeftAllegroHandHora/exp_left/stage2_nn/best.pth

# Same experiment name, different hand directories
scripts/deploy_two_hands.sh my_experiment
# Right: outputs/RightAllegroHandHora/my_experiment/stage2_nn/best.pth
# Left:  outputs/LeftAllegroHandHora/my_experiment/stage2_nn/best.pth
```

---

## License

This repository is licensed under the MIT License.

- Original work by [Haozhi Qi](https://haozhi.io/hora/) (HORA, © 2022)
- Modifications and integration by [**Wonik Robotics**](https://github.com/Wonikrobotics-git) (© 2025)
- Additional contributions (Isaac Lab port, DIGIT tactile hand, dataset collection tooling) (© 2025)

The full license text, including third-party software/asset notices (IsaacGymEnvs, rl_games, YCB object set, Isaac Gym, allegro_hand_ros2), is available in the [LICENSE](./LICENSE) file.
