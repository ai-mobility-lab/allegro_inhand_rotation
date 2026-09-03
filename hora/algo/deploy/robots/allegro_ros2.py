#!/usr/bin/env python3
"""
Allegro Hand I/O for ROS2

Provides simple I/O interface for Allegro robotic hands (left/right).
Features:
- Command publishing to position controllers
- Joint state subscription and reordering
- Position gap monitoring (difference between commanded and actual positions)
- Support for both single-hand and two-hands operation
- Dynamic motion demos (open/close, pinch, wave, etc.)

Joint name conventions:
- Single hand: ah_joint00, ah_joint01, ... (no side prefix)
- Two hands:   ahr_joint00 (right), ahl_joint00 (left), ... (with side prefix)
"""

import threading
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


DEFAULT_ORDER = {
    # For single hand (no side prefix)
    "single": [
        "ah_joint00", "ah_joint01", "ah_joint02", "ah_joint03",
        "ah_joint10", "ah_joint11", "ah_joint12", "ah_joint13",
        "ah_joint20", "ah_joint21", "ah_joint22", "ah_joint23",
        "ah_joint30", "ah_joint31", "ah_joint32", "ah_joint33",
    ],
    # For two hands (with side prefix)
    "right": [
        "ahr_joint00", "ahr_joint01", "ahr_joint02", "ahr_joint03",
        "ahr_joint10", "ahr_joint11", "ahr_joint12", "ahr_joint13",
        "ahr_joint20", "ahr_joint21", "ahr_joint22", "ahr_joint23",
        "ahr_joint30", "ahr_joint31", "ahr_joint32", "ahr_joint33",
    ],
    "left": [
        "ahl_joint00", "ahl_joint01", "ahl_joint02", "ahl_joint03",
        "ahl_joint10", "ahl_joint11", "ahl_joint12", "ahl_joint13",
        "ahl_joint20", "ahl_joint21", "ahl_joint22", "ahl_joint23",
        "ahl_joint30", "ahl_joint31", "ahl_joint32", "ahl_joint33",
    ],
}

###################################################################################################

class AllegroHandIO(Node):
    """
    ROS 2 I/O node for Allegro hand.
    - Publishes commands to /<controller_name>/commands
    - Subscribes to /joint_states and returns a 16-D vector in Allegro order
    - Publishes position gap to /position_gap (difference between commanded and actual positions)
    """

    def __init__(
        self,
        side: str = "right",
        controller_name: Optional[str] = None,
        joint_states_topic: str = "/joint_states",
        command_topic: Optional[str] = None,
        use_side_prefix: bool = False,  # True for two-hands setup (ahr_/ahl_), False for single hand (ah_)
    ):
        super().__init__("allegro_hand_io")

        side = (side or "right").lower()
        if side not in ("right", "left"):
            self.get_logger().warn(f"Unknown side '{side}', defaulting to 'right'.")
            side = "right"
        self.side = side
        self.use_side_prefix = use_side_prefix

        if controller_name is None:
            if use_side_prefix:
                # Two-hands setup: use side-specific controller names
                controller_name = (
                    "allegro_hand_position_controller_r"
                    if self.side == "right"
                    else "allegro_hand_position_controller_l"
                )
            else:
                # Single hand setup: use generic controller name
                controller_name = "allegro_hand_position_controller"

        if command_topic is None:
            command_topic = f"/{controller_name}/commands"

        self._cmd_pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self._gap_pub = self.create_publisher(JointState, "/position_gap", 10)
        self._last_js: Optional[JointState] = None
        self._index_map: Optional[List[int]] = None
        self._index_map_attempted: bool = False  # Track if we already tried to build index map
        self._last_cmd: Optional[np.ndarray] = None

        # Select appropriate joint name order based on setup
        if use_side_prefix:
            self._desired_names = DEFAULT_ORDER[self.side]  # "right" or "left"
        else:
            self._desired_names = DEFAULT_ORDER["single"]  # "ah_joint00", etc.

        self.create_subscription(JointState, joint_states_topic, self._on_js, 10)

        self.safe_pose = np.array([
            0.5, 0.2, 0.0, 0.0,   # Thumb
            0.0, 0.0, 0.0, 0.0,   # Index
            0.0, 0.0, 0.0, 0.0,   # Middle
            0.0, 0.0, 0.0, 0.0,   # Ring
        ], dtype=float)

        self.get_logger().info(f"[AllegroHandIO] side={self.side}")
        self.get_logger().info(f"[AllegroHandIO] cmd topic={command_topic}")
        self.get_logger().info(f"[AllegroHandIO] joint_states topic={joint_states_topic}")
        self.get_logger().info(f"[AllegroHandIO] gap topic=/position_gap")
        if use_side_prefix:
            self.get_logger().info(f"[AllegroHandIO] joint names: {'ahr_' if self.side=='right' else 'ahl_'}joint##")
        else:
            self.get_logger().info(f"[AllegroHandIO] joint names: ah_joint## (no side prefix)")

    def command_joint_position(self, positions: List[float]) -> bool:
        """Publish 16D target pose + publish gap once immediately after"""
        try:
            data = [float(x) for x in list(positions)]
        except Exception:
            self.get_logger().warn("command_joint_position: positions must be a sequence of numbers.")
            return False

        if len(data) != 16:
            self.get_logger().warn(f"command_joint_position: expected 16 elements, got {len(data)}.")
            return False

        msg = Float64MultiArray()
        msg.data = data
        self._cmd_pub.publish(msg)
        self._last_cmd = np.asarray(data, dtype=float)

        # Publish gap once with the latest state (non-blocking)
        self._publish_position_gap()
        return True


    def poll_joint_position(
        self, wait: bool = False, timeout: float = 3.0
    ) -> Optional[np.ndarray]:
        """Returns the current joint positions in Allegro order (16-D).

        Args:
            wait (bool): Whether to wait for data (spins until timeout if True)
            timeout (float): Maximum wait time in seconds

        Returns:
            np.ndarray | None: 16-D joint vector or None
        """

        # Wait for JointState
        if self._last_js is None and wait:
            start_time = time.time()
            while self._last_js is None and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.05)

        js = self._last_js
        if js is None or not js.position:
            return None

        # Initialize name-based index mapping (only on first receipt)
        if self._index_map is None and not self._index_map_attempted and js.name:
            self._index_map = self._build_index_map(js.name)
            self._index_map_attempted = True  # Mark that we tried (success or fail)

        # If index mapping is successful, sort and return
        if self._index_map:
            try:
                vec = np.array([js.position[i] for i in self._index_map], dtype=float)
                if vec.size == 16:
                    return vec
            except Exception:
                self.get_logger().warn("poll_joint_position: index mapping failed, fallback to raw order.")

        # Fallback on mapping failure — use the first 16
        if len(js.position) >= 16:
            return np.array(js.position[:16], dtype=float)

        return None


    def _build_index_map(self, joint_names: List[str]) -> Optional[List[int]]:
        """Create Allegro 16D index mapping from the list of joint_states names."""
        name_to_index = {n.lower(): i for i, n in enumerate(joint_names)}
        index_map = []

        for desired in self._desired_names:
            idx = name_to_index.get(desired.lower())
            if idx is None:
                # If even one match fails, invalidate the entire mapping
                self.get_logger().info(
                    f"Joint name matching failed for '{desired}'. "
                    f"Available names: {joint_names[:20] if len(joint_names) <= 20 else joint_names[:20] + ['...']}\n"
                    f"Using fallback mode (raw order from /joint_states)."
                )
                return None
            index_map.append(idx)

        if len(index_map) == 16:
            self.get_logger().info(f"Successfully mapped joint names: {self._desired_names[0]}, {self._desired_names[1]}, ...")
            return index_map
        return None


    def go_safe(self):
        # fix: use correct method name
        self.command_joint_position(self.safe_pose)

    def _on_js(self, msg: JointState):
        self._last_js = msg

    def _publish_position_gap(self):
        """Publish the gap between commanded and current joint positions"""
        if self._last_cmd is None or self._last_js is None:
            return
        cur = self.poll_joint_position(wait=False)
        if cur is None or cur.size != 16:
            return
        gap = (self._last_cmd - cur).astype(float)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = ""
        js.name = list(self._desired_names)
        js.position = gap.tolist()
        self._gap_pub.publish(js)


################### one hand io runner ####################

class _Runner:
    def __init__(self, node: AllegroHandIO):
        self.node = node
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(node)
        self.thread = threading.Thread(target=self.exec.spin, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        try:
            self.exec.shutdown()
        finally:
            self.node.destroy_node()


def start_allegro_io(side: str = "right") -> AllegroHandIO:
    """
    Start a single Allegro hand I/O node.
    Uses generic joint names without side prefix (ah_joint00, ah_joint01, ...).
    """
    if not rclpy.ok():
        rclpy.init()
    io = AllegroHandIO(side=side, use_side_prefix=False)  # Single hand: no side prefix
    io._runner = _Runner(io)   # keep reference
    io._runner.start()
    return io


def stop_allegro_io(io: AllegroHandIO):
    if hasattr(io, "_runner") and io._runner:
        io._runner.stop()
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass

################ two hands io demo ####################

# --- Add: Run multiple nodes in one executor ---
class _RunnerMany:
    def __init__(self, nodes):
        from rclpy.executors import SingleThreadedExecutor
        self.nodes = nodes
        self.exec = SingleThreadedExecutor()
        for n in nodes:
            self.exec.add_node(n)
        import threading
        self.thread = threading.Thread(target=self.exec.spin, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        try:
            self.exec.shutdown()
        finally:
            for n in self.nodes:
                n.destroy_node()

def start_allegro_ios(sides=("right", "left")):
    """
    Start multiple Allegro hand I/O nodes (for two-hands setup).
    Uses side-specific joint names with prefix (ahr_joint00, ahl_joint00, ...).
    """
    if not rclpy.ok():
        rclpy.init()
    nodes = [AllegroHandIO(side=s, use_side_prefix=True) for s in sides]  # Two hands: use side prefix
    runner = _RunnerMany(nodes)
    runner.start()
    # Keep a reference to the runner (needed for cleanup)
    for n in nodes:
        n._runner_many = runner
    return {n.side: n for n in nodes}

def stop_allegro_ios(nodes_dict):
    # Since they share the same runner, just turn off one
    any_node = next(iter(nodes_dict.values()))
    if hasattr(any_node, "_runner_many"):
        any_node._runner_many.stop()
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass


def demo_one_hand(side: str = "right", use_side_prefix: bool = False):
    """
    Dynamic motion demo for one hand.
    Demonstrates: open, close, pinch, and sequential finger wave.

    Args:
        side: "right" or "left"
        use_side_prefix: True for dual setup (ahr_/ahl_), False for single setup (ah_)
    """
    print(f"\n{'='*70}")
    print(f"  ONE HAND DEMO ({side.upper()})")
    print(f"{'='*70}\n")

    # Start the appropriate I/O based on setup
    if use_side_prefix:
        # Dual setup: need to start single node with side prefix
        if not rclpy.ok():
            rclpy.init()
        io = AllegroHandIO(side=side, use_side_prefix=True)
        io._runner = _Runner(io)
        io._runner.start()
    else:
        # Single setup: use standard start function
        io = start_allegro_io(side=side)
    try:
        print(f"[{side}] Waiting for /joint_states ...")
        cur = io.poll_joint_position(wait=True, timeout=5.0)
        print(f"[{side}] Current 16-D:", None if cur is None else np.round(cur, 3))

        if cur is None:
            print(f"[{side}] ❌ Failed to read joint states. Exiting.")
            return

        # Define some dynamic poses (Thumb, Index, Middle, Ring order)
        open_pose = np.array([
            0.5, 0.0, 0.0, 0.0,   # Thumb
            0.0, 0.0, 0.0, 0.0,   # Index
            0.0, 0.0, 0.0, 0.0,   # Middle
            0.0, 0.0, 0.0, 0.0,   # Ring
        ])

        closed_pose = np.array([
            0.5, 0.8, 0.8, 0.8,   # Thumb
            0.0, 1.0, 1.0, 1.0,   # Index
            0.0, 1.0, 1.0, 1.0,   # Middle
            0.0, 1.0, 1.0, 1.0,   # Ring
        ])

        pinch_pose = np.array([
            0.5, 1.0, 0.5, 0.5,   # Thumb - pinch
            0.0, 1.2, 1.0, 1.0,   # Index - pinch
            0.0, 0.2, 0.2, 0.2,   # Middle - open
            0.0, 0.2, 0.2, 0.2,   # Ring - open
        ])

        print(f"\n[{side}] Starting dynamic motion sequence...")
        print(f"[{side}] 1/5: Open hand")
        io.command_joint_position(open_pose)
        time.sleep(1.5)

        print(f"[{side}] 2/5: Close hand")
        io.command_joint_position(closed_pose)
        time.sleep(1.5)

        print(f"[{side}] 3/5: Open again")
        io.command_joint_position(open_pose)
        time.sleep(1.5)

        print(f"[{side}] 4/5: Pinch gesture (thumb + index)")
        io.command_joint_position(pinch_pose)
        time.sleep(1.5)

        print(f"[{side}] 5/5: Sequential finger wave")
        # Wave motion - close fingers one by one
        for i in range(4):
            wave_pose = open_pose.copy()
            for finger_idx in range(i + 1):
                wave_pose[finger_idx*4 + 1] = 1.0
                wave_pose[finger_idx*4 + 2] = 1.0
                wave_pose[finger_idx*4 + 3] = 1.0
            io.command_joint_position(wave_pose)
            print(f"[{side}]   - Finger {i+1}/4")
            time.sleep(0.5)

        time.sleep(1.0)
        print(f"\n[{side}] ✓ Motion sequence complete!")

    finally:
        print(f"[{side}] Going to safe pose and stopping...")
        io.go_safe()
        time.sleep(1.0)

        # Stop based on setup type
        if use_side_prefix:
            # Dual setup: manually stop
            if hasattr(io, "_runner") and io._runner:
                io._runner.stop()
        else:
            # Single setup: use standard stop function
            stop_allegro_io(io)

        print(f"[{side}] ✓ Demo finished.\n")


def demo_two_hands():
    """
    Dynamic motion demo for two hands (left + right).
    Demonstrates: synchronized open/close and alternating motions.
    """
    print(f"\n{'='*70}")
    print(f"  TWO HANDS DEMO (LEFT + RIGHT)")
    print(f"{'='*70}\n")

    ios = start_allegro_ios(("right", "left"))
    io_r = ios["right"]
    io_l = ios["left"]

    try:
        print("[two-hands] Waiting for /joint_states ...")
        cur_r = io_r.poll_joint_position(wait=True, timeout=5.0)
        cur_l = io_l.poll_joint_position(wait=True, timeout=5.0)
        print(f"[two-hands] Right: {None if cur_r is None else np.round(cur_r, 3)}")
        print(f"[two-hands] Left:  {None if cur_l is None else np.round(cur_l, 3)}")

        if cur_r is None or cur_l is None:
            print("[two-hands] ❌ Failed to read joint states. Exiting.")
            return

        open_pose = np.array([
            0.5, 0.0, 0.0, 0.0,   # Thumb
            0.0, 0.0, 0.0, 0.0,   # Index
            0.0, 0.0, 0.0, 0.0,   # Middle
            0.0, 0.0, 0.0, 0.0,   # Ring
        ])

        closed_pose = np.array([
            0.5, 0.8, 0.8, 0.8,   # Thumb
            0.0, 1.0, 1.0, 1.0,   # Index
            0.0, 1.0, 1.0, 1.0,   # Middle
            0.0, 1.0, 1.0, 1.0,   # Ring
        ])

        print("\n[two-hands] Starting synchronized motion sequence...")

        print("[two-hands] 1/4: Both hands open")
        io_r.command_joint_position(open_pose)
        io_l.command_joint_position(open_pose)
        time.sleep(2.0)

        print("[two-hands] 2/4: Both hands close")
        io_r.command_joint_position(closed_pose)
        io_l.command_joint_position(closed_pose)
        time.sleep(2.0)

        print("[two-hands] 3/4: Alternating - Right open, Left close")
        io_r.command_joint_position(open_pose)
        io_l.command_joint_position(closed_pose)
        time.sleep(2.0)

        print("[two-hands] 4/4: Alternating - Right close, Left open")
        io_r.command_joint_position(closed_pose)
        io_l.command_joint_position(open_pose)
        time.sleep(2.0)

        print("\n[two-hands] ✓ Motion sequence complete!")

    finally:
        print("[two-hands] Going to safe pose and stopping...")
        io_r.go_safe()
        io_l.go_safe()
        time.sleep(1.0)
        stop_allegro_ios(ios)
        print("[two-hands] ✓ Demo finished.\n")


def detect_connected_hands():
    """
    Detect how many Allegro hands are connected by checking ROS2 topics.

    Returns:
        str: "single" if one hand detected, "dual" if two hands detected, "none" if no hands
    """
    if not rclpy.ok():
        rclpy.init()

    # Create a temporary node to get topic list
    temp_node = Node("temp_topic_checker")
    topic_names_and_types = temp_node.get_topic_names_and_types()
    temp_node.destroy_node()

    # Extract just the topic names
    topics = [name for name, _ in topic_names_and_types]

    # Check for single hand controller
    has_single = "/allegro_hand_position_controller/commands" in topics

    # Check for dual hand controllers
    has_right = "/allegro_hand_position_controller_r/commands" in topics
    has_left = "/allegro_hand_position_controller_l/commands" in topics

    print("\n[Detection] ROS2 Topics found:")
    if has_single:
        print("  ✓ /allegro_hand_position_controller/commands (single hand)")
    if has_right:
        print("  ✓ /allegro_hand_position_controller_r/commands (right hand)")
    if has_left:
        print("  ✓ /allegro_hand_position_controller_l/commands (left hand)")

    if has_right and has_left:
        print("\n[Detection] 🖐️🖐️ Two hands detected (dual setup)")
        return "dual"
    elif has_single:
        print("\n[Detection] 🖐️ One hand detected (single setup)")
        return "single"
    else:
        print("\n[Detection] ❌ No Allegro hand controllers detected")
        return "none"


if __name__ == "__main__":
    import sys

    print("\n" + "="*70)
    print("  ALLEGRO HAND DEMO SELECTOR")
    print("="*70)

    # Detect connected hands
    hand_setup = detect_connected_hands()

    if hand_setup == "none":
        print("\n❌ No Allegro hands detected. Please check:")
        print("  - ROS2 nodes are running")
        print("  - Allegro hand controllers are launched")
        print("  - Try: ros2 topic list")
        sys.exit(1)

    # Build menu based on detected setup
    menu_options = {}
    menu_idx = 1

    print("\n" + "="*70)
    print("Available demos:")

    if hand_setup == "single":
        # Single hand: only one hand demo (no left/right distinction)
        print(f"  {menu_idx}. One hand demo")
        menu_options[menu_idx] = ("single", None)
        menu_idx += 1

    elif hand_setup == "dual":
        # Dual hands: both single hand demos and two-hands demo
        print(f"  {menu_idx}. One hand (right only)")
        menu_options[menu_idx] = ("right", None)
        menu_idx += 1

        print(f"  {menu_idx}. One hand (left only)")
        menu_options[menu_idx] = ("left", None)
        menu_idx += 1

        print(f"  {menu_idx}. Two hands (synchronized)")
        menu_options[menu_idx] = ("dual", None)
        menu_idx += 1

    print("\nUsage:")
    print("  python allegro_ros2.py [demo_number]")
    print("  Example: python allegro_ros2.py 1")
    print("="*70 + "\n")

    # Parse command line argument
    demo_choice = None
    if len(sys.argv) > 1:
        try:
            demo_choice = int(sys.argv[1])
        except ValueError:
            print(f"❌ Invalid demo number. Please use 1-{len(menu_options)}.")
            sys.exit(1)
    else:
        # Interactive mode
        try:
            demo_choice = int(input(f"Select demo number (1-{len(menu_options)}): "))
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input or cancelled.")
            sys.exit(1)

    # Validate choice
    if demo_choice not in menu_options:
        print(f"❌ Invalid demo number: {demo_choice}. Please use 1-{len(menu_options)}.")
        sys.exit(1)

    # Run selected demo
    demo_type, _ = menu_options[demo_choice]

    if demo_type == "single":
        # Single hand setup: use generic side (will use ah_joint## names)
        demo_one_hand(side="right", use_side_prefix=False)  # side doesn't matter for single hand
    elif demo_type == "right":
        # Dual hand setup: right only
        demo_one_hand(side="right", use_side_prefix=True)
    elif demo_type == "left":
        # Dual hand setup: left only
        demo_one_hand(side="left", use_side_prefix=True)
    elif demo_type == "dual":
        # Dual hand setup: both hands
        demo_two_hands()
    else:
        print(f"❌ Unknown demo type: {demo_type}")
        sys.exit(1)
