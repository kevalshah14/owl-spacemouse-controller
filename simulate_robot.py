#!/usr/bin/env python3
"""
PyBullet simulation for the OWL 68 robot arm with SpaceMouse control.
Uses Cartesian move_to_pose control similar to python_robgpt_ws.
Gets initial pose from real robot via gRPC.
"""

import pybullet as p
import pybullet_data
import time
import os
import re
import sys
import numpy as np

try:
    import pyspacemouse
    SPACEMOUSE_AVAILABLE = True
except ImportError:
    SPACEMOUSE_AVAILABLE = False
    print("Warning: pyspacemouse not installed.")

# Add python_robgpt_ws to path for gRPC client
ROBGPT_WS_PATH = "/Users/keval/Documents/VSCode/python_robgpt_ws"
if ROBGPT_WS_PATH not in sys.path:
    sys.path.insert(0, ROBGPT_WS_PATH)

GRPC_CLIENT_AVAILABLE = False
try:
    from src.grpc_service.robot_grpc_client import RobotGRPCClient
    GRPC_CLIENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: gRPC client not available: {e}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "owl_68_robot_description", "urdf", "owl_68_gripper_drill.urdf")
PACKAGE_DIR = os.path.join(SCRIPT_DIR, "owl_68_robot_description")

# SpaceMouse settings (same as ball control)
SENSITIVITY = 0.008      # meters per unit
Z_SENSITIVITY = 0.15     # Z is slower (multiplier)
DEADZONE = 0.12          # Ignore small inputs (increased to prevent drift)

# gRPC settings
GRPC_SERVER = "localhost:50051"


def get_robot_pose_from_grpc() -> dict:
    """
    Get the current robot pose from the real robot via gRPC.
    Returns position [x,y,z] and orientation quaternion [x,y,z,w].
    """
    if not GRPC_CLIENT_AVAILABLE:
        print("gRPC client not available, using default pose")
        return None
    
    try:
        print(f"Connecting to robot gRPC server at {GRPC_SERVER}...")
        client = RobotGRPCClient()
        
        if client.connect(GRPC_SERVER):
            print("Connected to gRPC server!")
            pose = client.get_pose()
            client.disconnect()
            
            if pose.get("success"):
                print(f"Got robot pose: position={pose['position']}, orientation={pose['orientation']}")
                return {
                    'position': np.array(pose['position']),
                    'orientation': np.array(pose['orientation'])  # [x,y,z,w] quaternion
                }
            else:
                print(f"Failed to get pose: {pose.get('error')}")
                return None
        else:
            print("Failed to connect to gRPC server")
            return None
            
    except Exception as e:
        print(f"gRPC error: {e}")
        return None


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
                 initial_pose: dict = None):
        self.robot_id = robot_id
        self.end_effector_index = end_effector_index
        self.joint_indices = joint_indices
        
        # Current target pose
        self.target_position = None
        self.target_orientation = None
        
        # Get initial pose from simulation first
        self._update_current_pose()
        
        # Use provided initial pose if available (from real robot)
        if initial_pose is not None:
            self.target_position = initial_pose['position'].copy()
            self.target_orientation = initial_pose['orientation'].copy()
            print(f"Initialized controller with real robot pose")
        else:
            self.target_position = self.current_position.copy()
            self.target_orientation = self.current_orientation.copy()
    
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


def main():
    # Get initial pose from real robot via gRPC
    print("=" * 60)
    print("Getting initial pose from real robot...")
    print("=" * 60)
    initial_pose = get_robot_pose_from_grpc()
    
    if initial_pose:
        print(f"Using real robot pose:")
        print(f"  Position: {initial_pose['position']}")
        print(f"  Orientation (quat): {initial_pose['orientation']}")
    else:
        print("Could not get robot pose, using default position")
    
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
    
    # Create Cartesian controller (pass initial pose from real robot if available)
    controller = CartesianRobotController(robot_id, end_effector_index, joint_indices,
                                          initial_pose=initial_pose)
    
    # If we got a pose from the real robot, move simulation robot to match
    if initial_pose is not None:
        print("Moving simulation robot to match real robot pose...")
        controller.move_to_pose(initial_pose['position'], initial_pose['orientation'])
        # Step simulation a few times to let robot settle
        for _ in range(100):
            p.stepSimulation()
        print("Simulation robot aligned with real robot!")
    
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
    
    # Camera
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-35,
                                 cameraTargetPosition=[0, 0, 0.3])
    
    print("\n" + "="*60)
    print("OWL 68 Robot - Cartesian Move To Pose Control")
    print("="*60)
    print("\nSpaceMouse controls (same as ball):")
    print("  Push FORWARD/BACK  → Move X (red axis)")
    print("  Push RIGHT/LEFT    → Move Y (green axis)")
    print("  Push UP/DOWN       → Move Z (blue axis) - slower")
    print("\n  Axis Lock: Only ONE axis moves at a time")
    print("="*60 + "\n")
    sys.stdout.flush()
    
    # Main loop
    frame_count = 0
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
                # Read fresh state (discard any buffered data)
                state = spacemouse_device.read()
                
                # Apply deadzone - set to exactly 0 if below threshold
                raw_x = 0.0
                raw_y = 0.0
                raw_z = 0.0
                
                if abs(state.x) > DEADZONE:
                    raw_x = state.x
                if abs(state.y) > DEADZONE:
                    raw_y = state.y
                if abs(state.z) > DEADZONE:
                    raw_z = state.z
                
                # Axis lock - only move in strongest axis
                abs_x, abs_y, abs_z = abs(raw_x), abs(raw_y), abs(raw_z)
                max_axis = max(abs_x, abs_y, abs_z)
                
                if max_axis > 0:
                    has_input = True
                    if abs_x == max_axis:
                        dy = raw_x * sens  # SpaceMouse X → Robot Y
                        active_axis = "Y"
                    elif abs_y == max_axis:
                        dx = raw_y * sens  # SpaceMouse Y → Robot X
                        active_axis = "X"
                    else:
                        dz = raw_z * sens * Z_SENSITIVITY
                        active_axis = "Z"
                    
                    # Only move when there's active input
                    controller.move_delta(dx, dy, dz)
            
            # Update target marker
            p.resetBasePositionAndOrientation(target_marker, 
                                              controller.target_position.tolist(), [0, 0, 0, 1])
            
            # Update debug text
            if frame_count % 10 == 0:
                sm_text = f"SpaceMouse: dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} [Axis: {active_axis}]"
                p.addUserDebugText(sm_text, [0.5, 0, 0.8], textColorRGB=[1, 1, 0], 
                                  textSize=1.2, replaceItemUniqueId=debug_text_id)
                
                tcp = controller.get_tcp_pose()
                pose_text = f"TCP: x={tcp['x']:.3f} y={tcp['y']:.3f} z={tcp['z']:.3f}"
                p.addUserDebugText(pose_text, [0.5, 0, 0.7], textColorRGB=[0, 1, 1], 
                                  textSize=1.0, replaceItemUniqueId=pose_text_id)
            
            p.stepSimulation()
            time.sleep(1./500.)
            
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if spacemouse_device:
            spacemouse_device.close()
        p.disconnect()
        if os.path.exists(modified_urdf_path):
            os.remove(modified_urdf_path)


if __name__ == "__main__":
    main()
