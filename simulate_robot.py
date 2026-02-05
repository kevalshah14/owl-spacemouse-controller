#!/usr/bin/env python3
"""
PyBullet simulation for the OWL 68 robot arm with SpaceMouse control.
Controls the end effector position/orientation using inverse kinematics.
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
    print("Warning: pyspacemouse not installed. Using sliders only.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "owl_68_robot_description", "urdf", "owl_68.urdf")
PACKAGE_DIR = os.path.join(SCRIPT_DIR, "owl_68_robot_description")

# SpaceMouse sensitivity - matching ball control
SENSITIVITY = 0.012   # meters per unit (same feel as ball)
Z_SENSITIVITY = 0.15  # Z is slower (multiplier)
DEADZONE = 0.08       # Ignore inputs below this threshold


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


def main():
    # Initialize SpaceMouse
    spacemouse_device = None
    if SPACEMOUSE_AVAILABLE:
        try:
            spacemouse_device = pyspacemouse.open()
            if spacemouse_device:
                print(f"SpaceMouse connected: {spacemouse_device.product_name}")
            else:
                print("SpaceMouse not found. Using sliders only.")
        except Exception as e:
            print(f"SpaceMouse error: {e}")
    
    # Connect to PyBullet with GUI
    physics_client = p.connect(p.GUI)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    
    plane_id = p.loadURDF("plane.urdf")
    
    modified_urdf_path = os.path.join(SCRIPT_DIR, "robot_pybullet.urdf")
    create_pybullet_urdf(URDF_PATH, modified_urdf_path)
    
    print(f"Loading robot...")
    sys.stdout.flush()
    
    start_pos = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
    try:
        robot_id = p.loadURDF(
            modified_urdf_path,
            start_pos,
            start_orientation,
            useFixedBase=True,
        )
        print("Robot loaded!")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error loading URDF: {e}")
        p.disconnect()
        return
    
    # Apply custom colors to make robot visually distinct
    # Color scheme: gradient from base to end effector
    link_colors = {
        -1: [0.15, 0.15, 0.15, 1.0],      # Base - dark charcoal
        0: [0.2, 0.2, 0.25, 1.0],          # Link 1 - dark blue-gray
        1: [0.3, 0.5, 0.7, 1.0],           # Link 2 - steel blue
        2: [0.4, 0.6, 0.8, 1.0],           # Link 3 - light steel blue
        3: [0.5, 0.7, 0.9, 1.0],           # Link 4 - sky blue
        4: [0.6, 0.75, 0.85, 1.0],         # Link 5 - light blue
        5: [0.9, 0.5, 0.2, 1.0],           # Link 6 - orange (end effector)
        6: [1.0, 0.3, 0.3, 1.0],           # Tool mount - red
        7: [1.0, 0.2, 0.2, 1.0],           # TCP - bright red
    }
    
    # Apply colors to base link
    p.changeVisualShape(robot_id, -1, rgbaColor=link_colors.get(-1, [0.5, 0.5, 0.5, 1]))
    
    # Apply colors to all other links
    for i in range(p.getNumJoints(robot_id)):
        color = link_colors.get(i, [0.6, 0.6, 0.7, 1.0])
        p.changeVisualShape(robot_id, i, rgbaColor=color)
    
    print("Applied custom color scheme!")
    
    # Find joints and end effector
    num_joints = p.getNumJoints(robot_id)
    joint_indices = []
    joint_info_list = []
    end_effector_index = -1
    
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        joint_type = info[2]
        
        if joint_type != p.JOINT_FIXED:
            joint_indices.append(i)
            joint_info_list.append({
                'index': i,
                'name': joint_name,
                'lower': info[8],
                'upper': info[9]
            })
        
        # Find the end effector link (tool_mount or last link)
        if 'tool_mount' in joint_name.lower() or 'tcp' in joint_name.lower():
            end_effector_index = i
    
    # Use last joint if no tool_mount found
    if end_effector_index == -1:
        end_effector_index = num_joints - 1
    
    print(f"End effector link index: {end_effector_index}")
    print(f"Controllable joints: {[j['name'] for j in joint_info_list]}")
    
    # Get initial end effector pose
    ee_state = p.getLinkState(robot_id, end_effector_index)
    target_pos = list(ee_state[4])  # World position
    target_orn = list(p.getEulerFromQuaternion(ee_state[5]))  # Euler angles
    
    print(f"Initial EE position: {target_pos}")
    print(f"Initial EE orientation (euler): {target_orn}")
    
    # GUI slider for sensitivity
    sens_slider = p.addUserDebugParameter("Sensitivity", 0.005, 0.03, SENSITIVITY)
    
    # Debug text for SpaceMouse values
    debug_text_id = p.addUserDebugText("SpaceMouse: waiting...", [0.5, 0, 0.8], textColorRGB=[1, 1, 0], textSize=1.2)
    ee_text_id = p.addUserDebugText("EE Pos: ...", [0.5, 0, 0.7], textColorRGB=[0, 1, 1], textSize=1.0)
    
    # Set up camera - aligned with world frame for intuitive Cartesian view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=90,      # Looking along -Y axis
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 0.3]
    )
    
    # Draw world coordinate axes at origin for reference
    axis_length = 0.3
    p.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], [1, 0, 0], lineWidth=3)  # X = Red
    p.addUserDebugLine([0, 0, 0], [0, axis_length, 0], [0, 1, 0], lineWidth=3)  # Y = Green
    p.addUserDebugLine([0, 0, 0], [0, 0, axis_length], [0, 0, 1], lineWidth=3)  # Z = Blue
    p.addUserDebugText("X", [axis_length + 0.05, 0, 0], [1, 0, 0], textSize=1.5)
    p.addUserDebugText("Y", [0, axis_length + 0.05, 0], [0, 1, 0], textSize=1.5)
    p.addUserDebugText("Z", [0, 0, axis_length + 0.05], [0, 0, 1], textSize=1.5)
    
    # Create target position marker (small sphere)
    target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.7])
    target_marker = p.createMultiBody(baseVisualShapeIndex=target_visual, basePosition=target_pos)
    
    print("\n" + "="*60)
    print("OWL 68 Robot - Cartesian Control (Same as Ball)")
    print("="*60)
    if spacemouse_device:
        print("\nSpaceMouse Control (Translation Only, Axis Locked):")
        print("  Push RIGHT/LEFT    → Move X (red axis)")
        print("  Push FORWARD/BACK  → Move Y (green axis)")
        print("  Push UP/DOWN       → Move Z (blue axis) - slower")
        print("\n  Only ONE axis moves at a time!")
        print("\n*** If no response, quit Logi Options+ from menu bar ***")
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
            sm_x = sm_y = sm_z = sm_roll = sm_pitch = sm_yaw = 0
            
            if spacemouse_device:
                state = spacemouse_device.read()
                
                # Apply deadzone (translation only - no rotation)
                raw_x = state.x if abs(state.x) > DEADZONE else 0
                raw_y = state.y if abs(state.y) > DEADZONE else 0
                raw_z = state.z if abs(state.z) > DEADZONE else 0
                
                # Axis lock - only move in the strongest axis
                abs_x, abs_y, abs_z = abs(raw_x), abs(raw_y), abs(raw_z)
                max_axis = max(abs_x, abs_y, abs_z)
                
                if max_axis > 0:
                    sm_x = raw_x if abs_x == max_axis else 0
                    sm_y = raw_y if abs_y == max_axis else 0
                    sm_z = raw_z if abs_z == max_axis else 0
                else:
                    sm_x = sm_y = sm_z = 0
                
                # Update target position (same as ball control)
                target_pos[0] += sm_x * sens
                target_pos[1] += sm_y * sens
                target_pos[2] += sm_z * sens * Z_SENSITIVITY  # Z is slower
                
                # Clamp position to reasonable workspace
                target_pos[0] = np.clip(target_pos[0], -0.8, 0.8)
                target_pos[1] = np.clip(target_pos[1], -0.8, 0.8)
                target_pos[2] = np.clip(target_pos[2], 0.05, 1.2)
            
            # Update target marker position
            p.resetBasePositionAndOrientation(target_marker, target_pos, [0, 0, 0, 1])
            
            # Update debug text every 10 frames
            if frame_count % 10 == 0:
                # Show active axis
                active = "X" if sm_x != 0 else ("Y" if sm_y != 0 else ("Z" if sm_z != 0 else "-"))
                sm_text = f"SpaceMouse: x={sm_x:.2f} y={sm_y:.2f} z={sm_z:.2f} [Axis: {active}]"
                p.addUserDebugText(sm_text, [0.5, 0, 0.8], textColorRGB=[1, 1, 0], textSize=1.2, 
                                  replaceItemUniqueId=debug_text_id)
                
                ee_text = f"EE Target: x={target_pos[0]:.3f} y={target_pos[1]:.3f} z={target_pos[2]:.3f}"
                p.addUserDebugText(ee_text, [0.5, 0, 0.7], textColorRGB=[0, 1, 1], textSize=1.0,
                                  replaceItemUniqueId=ee_text_id)
                
                # Print to terminal if there's significant input
                if abs(sm_x) > 0.05 or abs(sm_y) > 0.05 or abs(sm_z) > 0.05:
                    print(f"SM input: x={sm_x:.2f} y={sm_y:.2f} z={sm_z:.2f}")
                    sys.stdout.flush()
            
            # Calculate inverse kinematics
            target_orn_quat = p.getQuaternionFromEuler(target_orn)
            
            joint_positions = p.calculateInverseKinematics(
                robot_id,
                end_effector_index,
                target_pos,
                target_orn_quat,
                maxNumIterations=50,
                residualThreshold=1e-4
            )
            
            # Apply joint positions
            for i, joint_idx in enumerate(joint_indices):
                if i < len(joint_positions):
                    # Clamp to joint limits
                    pos = joint_positions[i]
                    pos = max(joint_info_list[i]['lower'], min(joint_info_list[i]['upper'], pos))
                    
                    p.setJointMotorControl2(
                        robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=pos,
                        force=1000,
                        maxVelocity=10.0
                    )
            
            p.stepSimulation()
            time.sleep(1./500.)  # Faster update rate
            
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
