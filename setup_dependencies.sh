#!/bin/bash
# This script installs system dependencies required for OpenCV

echo "Installing system dependencies for OpenCV..."

# Update package lists
apt-get update

# Install graphics libraries required by OpenCV
apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev

# Install additional dependencies that may be needed
apt-get install -y libglib2.0-0 libxkbcommon0

echo "System dependencies installed successfully!"

# Install or upgrade pip and Python packages needed by the project
if command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --upgrade pip setuptools wheel || true
    python3 -m pip install opencv-python || true
fi
