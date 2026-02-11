#!/usr/bin/env python3
"""
PyBullet simulation for the OWL 68 robot arm with SpaceMouse control.
Uses Cartesian move_to_pose control similar to python_robgpt_ws.
Gets initial pose from real robot via gRPC (robgpt_control_service).
"""

import pybullet as p
import pybullet_data
import time
import os
import re
import sys
import threading
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import pyspacemouse
    SPACEMOUSE_AVAILABLE = True
except ImportError:
    SPACEMOUSE_AVAILABLE = False
    print("Warning: pyspacemouse not installed.")

# gRPC imports for direct robot communication
GRPC_AVAILABLE = False
try:
    import grpc
    import robgpt_control_service_pb2
    import robgpt_control_service_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    print("gRPC proto files not found. Generate them with:")
    print("  python generate_grpc_protos.py")

GRPC_ADDRESS = "localhost:50052"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "owl_68_robot_description", "urdf", "owl_68_gripper_drill.urdf")
PACKAGE_DIR = os.path.join(SCRIPT_DIR, "owl_68_robot_description")

# SpaceMouse settings
SENSITIVITY = 0.002      # meters per unit (translation mode)
Z_SENSITIVITY = 0.15     # Z is slower (multiplier)
ROTATION_SENSITIVITY = 0.002  # radians per unit (rotation mode)
DEADZONE = 0.25          # Ignore small inputs (high to prevent drift)
INPUT_SMOOTHING = 0.15   # EMA alpha for SpaceMouse input (lower=smoother, more lag)

# Real robot control settings
REAL_ROBOT_JOINT_SPEED = 20.0     # deg/s for MoveToJoint (slower, smoother)
REAL_ROBOT_SEND_INTERVAL = 0.025  # seconds between gRPC commands (~40Hz)

def get_robot_state() -> dict:
    """
    Get the current robot state via gRPC (joint positions + TCP pose).
    Uses GetRobotState for joint-level data, which is more accurate than
    Cartesian-only GetPose when initializing a simulation.
    """
    if not GRPC_AVAILABLE:
        print("gRPC proto files not available, using default pose")
        return None

    try:
        channel = grpc.insecure_channel(GRPC_ADDRESS)
        grpc.channel_ready_future(channel).result(timeout=3.0)
        stub = robgpt_control_service_pb2_grpc.RobGPTControlServiceStub(channel)

        request = robgpt_control_service_pb2.GetRobotStateRequest()
        response = stub.GetRobotState(request, timeout=5.0)

        if not response.success:
            channel.close()
            print(f"GetRobotState failed: {response.error_message}")
            return None

        joint_positions = list(response.joint_positions)
        joint_names = list(response.joint_names)

        result = {
            'joint_positions': joint_positions,
            'joint_names': joint_names,
        }

        # Include TCP pose if the server populated it
        if response.HasField('tcp_position') and response.HasField('tcp_orientation'):
            result['tcp_position'] = np.array([
                response.tcp_position.x, response.tcp_position.y, response.tcp_position.z
            ])
            result['tcp_orientation'] = np.array([
                response.tcp_orientation.x, response.tcp_orientation.y,
                response.tcp_orientation.z, response.tcp_orientation.w
            ])

        # Also get real robot's TCP via GetPose (real FK, not sim FK)
        try:
            pose_request = robgpt_control_service_pb2.GetPoseRequest()
            pose_response = stub.GetPose(pose_request, timeout=5.0)
            if pose_response.success:
                result['real_tcp_position'] = np.array([
                    pose_response.position.x,
                    pose_response.position.y,
                    pose_response.position.z,
                ])
                result['real_tcp_orientation'] = np.array([
                    pose_response.orientation.x,
                    pose_response.orientation.y,
                    pose_response.orientation.z,
                    pose_response.orientation.w,
                ])
        except Exception as e:
            print(f"GetPose warning: {e}")

        channel.close()

        print(f"Got robot state: {len(joint_positions)} joints")
        for name, pos in zip(joint_names, joint_positions):
            print(f"  {name}: {pos:.4f} rad ({np.degrees(pos):.1f} deg)")
        if 'real_tcp_position' in result:
            tcp = result['real_tcp_position']
            print(f"  Real TCP: [{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}] m")

        return result

    except grpc.FutureTimeoutError:
        print("Robot gRPC server not reachable (timeout)")
        return None
    except Exception as e:
        print(f"GetRobotState error: {e}")
        return None


class RealRobotController:
    """
    Sends joint position commands to the real robot via gRPC MoveToJoint
    from a background thread so the main sim loop never blocks on network I/O.
    """

    def __init__(self, address: str = GRPC_ADDRESS):
        self.address = address
        self.channel = None
        self.stub = None
        self.enabled = False
        self.connected = False
        self.last_error = ""
        # Shared state between main thread and sender thread
        self._lock = threading.Lock()
        self._joint_positions = None  # Latest joint targets to send
        self._has_new_target = False
        self._thread = None
        self._stop_event = threading.Event()

    def connect(self) -> bool:
        """Establish gRPC connection to robot."""
        if not GRPC_AVAILABLE:
            self.last_error = "gRPC not available"
            return False
        try:
            self.channel = grpc.insecure_channel(self.address)
            grpc.channel_ready_future(self.channel).result(timeout=3.0)
            self.stub = robgpt_control_service_pb2_grpc.RobGPTControlServiceStub(self.channel)
            self.connected = True
            self.last_error = ""
            return True
        except grpc.FutureTimeoutError:
            self.last_error = "Connection timeout"
            self.connected = False
            return False
        except Exception as e:
            self.last_error = str(e)
            self.connected = False
            return False

    def disconnect(self):
        """Stop sender thread and close gRPC connection."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None
        self.connected = False

    def _sender_loop(self):
        """Background thread that sends joint positions to the real robot at a fixed rate."""
        while not self._stop_event.is_set():
            if not self.enabled or not self.connected or not self.stub:
                self._stop_event.wait(timeout=0.05)
                continue

            # Grab the latest joint targets (never hold lock while waiting)
            joints = None
            with self._lock:
                if self._has_new_target:
                    joints = self._joint_positions[:]
                    self._has_new_target = False

            if joints is None:
                self._stop_event.wait(timeout=REAL_ROBOT_SEND_INTERVAL)
                continue

            try:
                request = robgpt_control_service_pb2.MoveToJointRequest(
                    joint_positions=joints,
                    speed=REAL_ROBOT_JOINT_SPEED,
                    wait=False,
                )
                response = self.stub.MoveToJoint(request, timeout=2.0)
                if not response.success:
                    self.last_error = response.error_message
                else:
                    self.last_error = ""
            except Exception as e:
                self.last_error = str(e)

            # Sleep for the send interval
            self._stop_event.wait(timeout=REAL_ROBOT_SEND_INTERVAL)

    def _ensure_thread(self):
        """Start the sender thread if not already running."""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._thread.start()

    def toggle(self) -> bool:
        """Toggle real robot control on/off. Returns new state."""
        if not self.enabled:
            if not self.connected:
                if not self.connect():
                    print(f"Cannot enable real robot: {self.last_error}")
                    return False
            self._ensure_thread()
            self.enabled = True
            print("REAL ROBOT CONTROL: ENABLED")
        else:
            self.enabled = False
            print("REAL ROBOT CONTROL: DISABLED")
        return self.enabled

    def update_target(self, joint_positions: list):
        """
        Update the target joint positions (non-blocking). The background
        thread picks up the latest target and sends it to the real robot.

        Args:
            joint_positions: List of 6 joint angles in radians
        """
        if not self.enabled or joint_positions is None:
            return
        with self._lock:
            self._joint_positions = joint_positions[:]
            self._has_new_target = True


def create_pybullet_urdf(original_path: str, output_path: str) -> str:
    """Create a URDF with absolute mesh paths for PyBullet."""
    with open(original_path, 'r') as f:
        urdf_content = f.read()
    
    def replace_package_url(match):
        relative_path = match.group(1)
        abs_path = os.path.join(PACKAGE_DIR, relative_path)
        return f'filename="{abs_path}"'
    
    modified_content = re.sub(
        r'filename="package://owl_68_robot_description/([^"]+)"',
        replace_package_url,
        urdf_content
    )
    
    modified_content = re.sub(r'<gazebo[^>]*>.*?</gazebo>', '', modified_content, flags=re.DOTALL)
    modified_content = re.sub(r'<safety_controller[^/]*/>', '', modified_content)
    
    with open(output_path, 'w') as f:
        f.write(modified_content)
    
    return output_path


class CartesianRobotController:
    """
    Cartesian robot controller using PyBullet IK.
    Similar interface to python_robgpt_ws move_to_pose.
    """
    
    def __init__(self, robot_id: int, end_effector_index: int, joint_indices: list,
                 initial_joint_positions: list = None):
        self.robot_id = robot_id
        self.end_effector_index = end_effector_index
        self.joint_indices = joint_indices

        # Apply real robot joint positions directly if available
        if initial_joint_positions is not None:
            for i, joint_idx in enumerate(self.joint_indices):
                if i < len(initial_joint_positions):
                    p.resetJointState(self.robot_id, joint_idx, initial_joint_positions[i])
            print("Set simulation joints to match real robot")

        # Read the resulting TCP pose from simulation as starting target
        self._update_current_pose()
        self.initial_position = self.current_position.copy()
        self.initial_orientation = self.current_orientation.copy()
        self.target_position = self.current_position.copy()
        self.target_orientation = self.current_orientation.copy()
        self.last_joint_positions = list(initial_joint_positions[:6]) if initial_joint_positions else None
    
    def _update_current_pose(self):
        """Update current end effector pose from simulation."""
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        self.current_position = np.array(ee_state[4])
        self.current_orientation = np.array(ee_state[5])  # Quaternion [x,y,z,w]
    
    def get_tcp_pose(self) -> dict:
        """Get current TCP pose (similar to python_robgpt_ws)."""
        self._update_current_pose()
        return {
            'x': self.current_position[0],
            'y': self.current_position[1],
            'z': self.current_position[2],
            'qx': self.current_orientation[0],
            'qy': self.current_orientation[1],
            'qz': self.current_orientation[2],
            'qw': self.current_orientation[3],
        }
    
    def move_to_pose(self, position: np.ndarray, orientation: np.ndarray = None) -> bool:
        """
        Move to Cartesian pose using IK.
        Similar to move_to_pose from python_robgpt_ws.
        
        Args:
            position: Target [x, y, z] in meters
            orientation: Target quaternion [x, y, z, w] (optional, keeps current if None)
        
        Returns:
            True if IK solution found
        """
        self.target_position = np.array(position)
        
        if orientation is not None:
            self.target_orientation = np.array(orientation)
        
        # Calculate IK
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            self.target_position.tolist(),
            self.target_orientation.tolist(),
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        if joint_positions is None:
            return False
        
        # Store the first 6 joint positions (robot arm joints only, no gripper)
        self.last_joint_positions = list(joint_positions[:6])

        # Apply joint positions with position control
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=1000,
                    maxVelocity=10.0
                )
        
        return True
    
    def move_delta(self, dx: float = 0, dy: float = 0, dz: float = 0) -> bool:
        """
        Move relative to current position in Cartesian space.
        
        Args:
            dx, dy, dz: Delta movement in meters
        
        Returns:
            True if successful
        """
        new_position = self.target_position + np.array([dx, dy, dz])
        
        # Clamp to workspace limits
        new_position[0] = np.clip(new_position[0], -0.8, 0.8)
        new_position[1] = np.clip(new_position[1], -0.8, 0.8)
        new_position[2] = np.clip(new_position[2], 0.05, 1.2)
        
        return self.move_to_pose(new_position)

    def rotate_delta(self, droll: float = 0, dpitch: float = 0, dyaw: float = 0) -> bool:
        """
        Rotate the TCP orientation by small euler angle deltas.

        Args:
            droll, dpitch, dyaw: Delta rotation in radians (applied in robot base frame)

        Returns:
            True if successful
        """
        # Convert current quaternion [x,y,z,w] to scipy format
        current_rot = Rotation.from_quat(self.target_orientation)
        delta_rot = Rotation.from_euler('xyz', [droll, dpitch, dyaw])
        new_rot = delta_rot * current_rot
        new_orientation = new_rot.as_quat()  # Returns [x,y,z,w]

        return self.move_to_pose(self.target_position, orientation=new_orientation)


def main():
    # Get initial joint state from real robot via gRPC
    print("=" * 60)
    print("Getting joint state from real robot...")
    print("=" * 60)
    robot_state = get_robot_state()

    if robot_state:
        print("Using real robot joint positions")
    else:
        print("Could not get robot state, using default position")

    # Initialize real robot controller
    real_robot = RealRobotController()
    real_robot.connect()  # Pre-connect but don't enable yet

    # Initialize SpaceMouse
    spacemouse_device = None
    if SPACEMOUSE_AVAILABLE:
        try:
            spacemouse_device = pyspacemouse.open()
            if spacemouse_device:
                print(f"SpaceMouse connected: {spacemouse_device.product_name}")
            else:
                print("SpaceMouse not found.")
        except Exception as e:
            print(f"SpaceMouse error: {e}")
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    
    # Load ground plane
    p.loadURDF("plane.urdf")
    
    # Load robot
    modified_urdf_path = os.path.join(SCRIPT_DIR, "robot_pybullet.urdf")
    create_pybullet_urdf(URDF_PATH, modified_urdf_path)
    
    print("Loading robot...")
    robot_id = p.loadURDF(modified_urdf_path, [0, 0, 0], useFixedBase=True)
    print("Robot loaded!")
    
    # Find joints and end effector
    num_joints = p.getNumJoints(robot_id)
    joint_indices = []
    end_effector_index = num_joints - 1
    
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        joint_type = info[2]
        
        if joint_type != p.JOINT_FIXED:
            joint_indices.append(i)
        
        # Look for gripper TCP (target_frame) or robot TCP
        if 'target_frame' in joint_name.lower():
            end_effector_index = i
        elif 'tool_mount' in joint_name.lower() or 'tcp' in joint_name.lower():
            if end_effector_index == num_joints - 1:  # Only if not already set
                end_effector_index = i
    
    # Print which end effector link we're using
    ee_info = p.getJointInfo(robot_id, end_effector_index)
    print(f"End effector: link {end_effector_index} ({ee_info[12].decode('utf-8')})")
    
    # Apply custom colors
    link_colors = {
        -1: [0.15, 0.15, 0.15, 1.0],
        0: [0.2, 0.2, 0.25, 1.0],
        1: [0.3, 0.5, 0.7, 1.0],
        2: [0.4, 0.6, 0.8, 1.0],
        3: [0.5, 0.7, 0.9, 1.0],
        4: [0.6, 0.75, 0.85, 1.0],
        5: [0.9, 0.5, 0.2, 1.0],
        6: [1.0, 0.3, 0.3, 1.0],
        7: [1.0, 0.2, 0.2, 1.0],
    }
    p.changeVisualShape(robot_id, -1, rgbaColor=link_colors.get(-1, [0.5, 0.5, 0.5, 1]))
    for i in range(num_joints):
        p.changeVisualShape(robot_id, i, rgbaColor=link_colors.get(i, [0.6, 0.6, 0.7, 1.0]))
    
    # Create Cartesian controller (pass joint positions from real robot if available)
    initial_joints = robot_state['joint_positions'] if robot_state else None
    controller = CartesianRobotController(robot_id, end_effector_index, joint_indices,
                                          initial_joint_positions=initial_joints)
    
    # Draw world coordinate axes
    axis_length = 0.3
    p.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], [1, 0, 0], lineWidth=3)
    p.addUserDebugLine([0, 0, 0], [0, axis_length, 0], [0, 1, 0], lineWidth=3)
    p.addUserDebugLine([0, 0, 0], [0, 0, axis_length], [0, 0, 1], lineWidth=3)
    p.addUserDebugText("X", [axis_length + 0.05, 0, 0], [1, 0, 0], textSize=1.5)
    p.addUserDebugText("Y", [0, axis_length + 0.05, 0], [0, 1, 0], textSize=1.5)
    p.addUserDebugText("Z", [0, 0, axis_length + 0.05], [0, 0, 1], textSize=1.5)
    
    # Create target marker
    target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.7])
    target_marker = p.createMultiBody(baseVisualShapeIndex=target_visual, 
                                      basePosition=controller.target_position.tolist())
    
    # GUI slider
    sens_slider = p.addUserDebugParameter("Sensitivity", 0.002, 0.02, SENSITIVITY)
    
    # Debug text
    debug_text_id = p.addUserDebugText("SpaceMouse: waiting...", [0.5, 0, 0.8], 
                                       textColorRGB=[1, 1, 0], textSize=1.2)
    pose_text_id = p.addUserDebugText("TCP: ...", [0.5, 0, 0.7], 
                                      textColorRGB=[0, 1, 1], textSize=1.0)
    robot_status_id = p.addUserDebugText("Real Robot: OFF", [0.5, 0, 0.6],
                                         textColorRGB=[1, 0.3, 0.3], textSize=1.0)
    mode_text_id = p.addUserDebugText("Mode: TRANSLATION (XYZ)", [0.5, 0, 0.5],
                                       textColorRGB=[0.8, 0.8, 1.0], textSize=1.0)
    
    # Camera
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=-90, cameraPitch=-15,
                                 cameraTargetPosition=[0, 0, 0.3])
    
    print("\n" + "="*60)
    print("OWL 68 Robot - Cartesian Move To Pose Control")
    print("="*60)
    print("\nTranslation mode (default):")
    print("  Push FORWARD/BACK  -> Move X (red axis)")
    print("  Push RIGHT/LEFT    -> Move Y (green axis)")
    print("  Push UP/DOWN       -> Move Z (blue axis)")
    print("\nRotation mode:")
    print("  Twist/tilt knob    -> Roll / Pitch / Yaw")
    print("\n  Axis Lock: Only ONE axis at a time")
    print(f"\n  Real robot: {'connected' if real_robot.connected else 'not connected'}")
    print("  LEFT BUTTON:  toggle real robot control")
    print("  RIGHT BUTTON: toggle translation/rotation mode")
    print("="*60 + "\n")
    sys.stdout.flush()
    
    # Main loop
    frame_count = 0
    prev_button_left = False
    prev_button_right = False
    orientation_mode = False  # Right button toggles rotation control
    # EMA-smoothed SpaceMouse values (translation)
    smooth_x = 0.0
    smooth_y = 0.0
    smooth_z = 0.0
    # EMA-smoothed SpaceMouse values (rotation)
    smooth_roll = 0.0
    smooth_pitch = 0.0
    smooth_yaw = 0.0
    alpha = INPUT_SMOOTHING
    try:
        while p.isConnected():
            frame_count += 1
            
            try:
                sens = p.readUserDebugParameter(sens_slider)
            except:
                break
            
            # Read SpaceMouse
            dx = dy = dz = 0
            active_axis = "-"
            has_input = False
            
            if spacemouse_device:
                # Read multiple times to flush buffer and get latest state
                state = spacemouse_device.read()
                for _ in range(5):
                    new_state = spacemouse_device.read()
                    if new_state is not None:
                        state = new_state

                # Toggle real robot on left button press (rising edge)
                button_left = len(state.buttons) > 0 and state.buttons[0]
                if button_left and not prev_button_left:
                    real_robot.toggle()
                prev_button_left = button_left

                # Toggle orientation mode on right button press (rising edge)
                button_right = len(state.buttons) > 1 and state.buttons[1]
                if button_right and not prev_button_right:
                    orientation_mode = not orientation_mode
                    mode_name = "ROTATION (RPY)" if orientation_mode else "TRANSLATION (XYZ)"
                    print(f"Mode: {mode_name}")
                prev_button_right = button_right

                drift_threshold = 0.02

                if not orientation_mode:
                    # --- Translation mode ---
                    raw_x = state.x if abs(state.x) > DEADZONE else 0.0
                    raw_y = state.y if abs(state.y) > DEADZONE else 0.0
                    raw_z = state.z if abs(state.z) > DEADZONE else 0.0

                    smooth_x = alpha * raw_x + (1 - alpha) * smooth_x
                    smooth_y = alpha * raw_y + (1 - alpha) * smooth_y
                    smooth_z = alpha * raw_z + (1 - alpha) * smooth_z

                    if abs(smooth_x) < drift_threshold:
                        smooth_x = 0.0
                    if abs(smooth_y) < drift_threshold:
                        smooth_y = 0.0
                    if abs(smooth_z) < drift_threshold:
                        smooth_z = 0.0

                    # Axis lock - only move in strongest axis
                    abs_x, abs_y, abs_z = abs(smooth_x), abs(smooth_y), abs(smooth_z)
                    max_axis = max(abs_x, abs_y, abs_z)

                    if max_axis > drift_threshold:
                        has_input = True
                        if abs_x == max_axis:
                            dy = -smooth_x * sens
                            active_axis = "Y"
                        elif abs_y == max_axis:
                            dx = smooth_y * sens
                            active_axis = "X"
                        else:
                            dz = smooth_z * sens * Z_SENSITIVITY
                            active_axis = "Z"

                        controller.move_delta(dx, dy, dz)
                        real_robot.update_target(controller.last_joint_positions)

                else:
                    # --- Rotation mode ---
                    raw_roll = state.roll if abs(state.roll) > DEADZONE else 0.0
                    raw_pitch = state.pitch if abs(state.pitch) > DEADZONE else 0.0
                    raw_yaw = state.yaw if abs(state.yaw) > DEADZONE else 0.0

                    smooth_roll = alpha * raw_roll + (1 - alpha) * smooth_roll
                    smooth_pitch = alpha * raw_pitch + (1 - alpha) * smooth_pitch
                    smooth_yaw = alpha * raw_yaw + (1 - alpha) * smooth_yaw

                    if abs(smooth_roll) < drift_threshold:
                        smooth_roll = 0.0
                    if abs(smooth_pitch) < drift_threshold:
                        smooth_pitch = 0.0
                    if abs(smooth_yaw) < drift_threshold:
                        smooth_yaw = 0.0

                    # Axis lock - only rotate around strongest axis
                    abs_r, abs_p, abs_yw = abs(smooth_roll), abs(smooth_pitch), abs(smooth_yaw)
                    max_rot = max(abs_r, abs_p, abs_yw)

                    if max_rot > drift_threshold:
                        has_input = True
                        dr = dp = dyw = 0.0
                        rot_sens = ROTATION_SENSITIVITY
                        if abs_r == max_rot:
                            dr = smooth_roll * rot_sens
                            active_axis = "Roll"
                        elif abs_p == max_rot:
                            dp = smooth_pitch * rot_sens
                            active_axis = "Pitch"
                        else:
                            dyw = smooth_yaw * rot_sens
                            active_axis = "Yaw"

                        controller.rotate_delta(dr, dp, dyw)
                        real_robot.update_target(controller.last_joint_positions)
            
            # Update target marker
            p.resetBasePositionAndOrientation(target_marker, 
                                              controller.target_position.tolist(), [0, 0, 0, 1])
            
            # Update debug text
            if frame_count % 10 == 0:
                if not orientation_mode:
                    sm_text = f"XYZ: dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} [Axis: {active_axis}]"
                else:
                    sm_text = f"RPY: [Axis: {active_axis}]"
                p.addUserDebugText(sm_text, [0.5, 0, 0.8], textColorRGB=[1, 1, 0], 
                                  textSize=1.2, replaceItemUniqueId=debug_text_id)
                
                tcp = controller.get_tcp_pose()
                rpy = Rotation.from_quat([tcp['qx'], tcp['qy'], tcp['qz'], tcp['qw']]).as_euler('xyz', degrees=True)
                pose_text = f"TCP: x={tcp['x']:.3f} y={tcp['y']:.3f} z={tcp['z']:.3f}"
                rpy_text = f"  R={rpy[0]:.1f} P={rpy[1]:.1f} Y={rpy[2]:.1f} deg"
                p.addUserDebugText(pose_text + rpy_text, [0.5, 0, 0.7], textColorRGB=[0, 1, 1], 
                                  textSize=1.0, replaceItemUniqueId=pose_text_id)

                # Update mode text
                mode_name = "ROTATION (RPY)" if orientation_mode else "TRANSLATION (XYZ)"
                mode_color = [1.0, 0.6, 0.2] if orientation_mode else [0.8, 0.8, 1.0]
                p.addUserDebugText(f"Mode: {mode_name}", [0.5, 0, 0.5],
                                   textColorRGB=mode_color, textSize=1.0,
                                   replaceItemUniqueId=mode_text_id)

                # Update real robot status text
                if real_robot.enabled:
                    status = "Real Robot: ON"
                    if real_robot.last_error:
                        status += f" (err: {real_robot.last_error})"
                    color = [0.3, 1.0, 0.3]
                else:
                    status = "Real Robot: OFF"
                    color = [1.0, 0.3, 0.3]
                p.addUserDebugText(status, [0.5, 0, 0.6], textColorRGB=color,
                                   textSize=1.0, replaceItemUniqueId=robot_status_id)
            
            p.stepSimulation()
            time.sleep(1./500.)
            
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        real_robot.disconnect()
        if spacemouse_device:
            spacemouse_device.close()
        p.disconnect()
        if os.path.exists(modified_urdf_path):
            os.remove(modified_urdf_path)


if __name__ == "__main__":
    main()
