# OWL 68 SpaceMouse Controller

A PyBullet simulation for the OWL 68 robot arm with 3Dconnexion SpaceMouse control. Control the robot's end-effector in Cartesian space using intuitive SpaceMouse inputs.

## Features

- **Cartesian End-Effector Control**: Move the robot TCP (Tool Center Point) in X, Y, Z space using inverse kinematics
- **SpaceMouse Integration**: Direct control via 3Dconnexion SpaceMouse (wireless or wired)
- **Axis Lock**: Only one axis moves at a time for precise positioning
- **Deadzone Filtering**: Prevents drift from small unintentional inputs
- **Real Robot Sync**: Optional gRPC integration to initialize simulation from real robot pose
- **Visual Feedback**: Real-time display of TCP position and active control axis

## Requirements

- Python 3.10+
- macOS (tested), Linux should work
- 3Dconnexion SpaceMouse (optional, for control)
- `hidapi` system library

## Installation

1. Install system dependencies (macOS):
   ```bash
   brew install hidapi
   ```

2. Create environment and install Python dependencies:
   ```bash
   uv sync
   ```

3. (Optional) Quit Logi Options+ or 3Dconnexion driver to allow direct HID access to SpaceMouse.

## Usage

Run the simulation:
```bash
uv run python simulate_robot.py
```

### SpaceMouse Controls

| Input | Action |
|-------|--------|
| Push Forward/Back | Move X axis (red) |
| Push Right/Left | Move Y axis (green) |
| Push Up/Down | Move Z axis (blue, slower) |

**Axis Lock**: Only the dominant axis registers movement, making precise single-axis positioning easier.

### Ball Demo

A simple 2D ball control demo for testing SpaceMouse input:
```bash
uv run python spacemouse_ball.py
```

## Configuration

Key parameters in `simulate_robot.py`:

```python
SENSITIVITY = 0.008      # meters per unit of SpaceMouse input
Z_SENSITIVITY = 0.15     # Z axis is slower (multiplier)
DEADZONE = 0.12          # Ignore inputs below this threshold
```

## gRPC Integration (Optional)

The simulation can fetch the initial end-effector pose from a real robot via gRPC:

1. Install grpcio: `uv add grpcio`
2. Ensure `python_robgpt_ws` is available at the configured path
3. Start the robot gRPC server on `localhost:50051`

The simulation will automatically sync to the real robot's pose on startup.

## Project Structure

```
owl-spacemouse-controller/
├── simulate_robot.py          # Main robot simulation
├── spacemouse_ball.py         # SpaceMouse test utility
├── owl_68_robot_description/  # Robot URDF and meshes
│   ├── urdf/
│   │   └── owl_68.urdf
│   └── meshes/
└── pyproject.toml
```

## Troubleshooting

**SpaceMouse not detected**: Quit Logi Options+ or any 3Dconnexion software that may be capturing the device.

**HID API error**: Ensure `hidapi` is installed (`brew install hidapi` on macOS).

**Robot moves after releasing SpaceMouse**: The deadzone may need adjustment. Increase `DEADZONE` value in the script.
