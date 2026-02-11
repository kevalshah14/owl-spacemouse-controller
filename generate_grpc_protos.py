#!/usr/bin/env python3
"""Generate Python gRPC stubs from the proto file."""

import subprocess
import sys
import os


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proto_file = os.path.join(script_dir, "robgpt_control_service.proto")

    if not os.path.exists(proto_file):
        print(f"Proto file not found: {proto_file}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={script_dir}",
        f"--python_out={script_dir}",
        f"--grpc_python_out={script_dir}",
        proto_file,
    ]

    print(f"Generating gRPC stubs from {os.path.basename(proto_file)}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"protoc failed:\n{result.stderr}")
        sys.exit(1)

    print("Generated:")
    print(f"  robgpt_control_service_pb2.py")
    print(f"  robgpt_control_service_pb2_grpc.py")


if __name__ == "__main__":
    main()
