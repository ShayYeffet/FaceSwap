#!/bin/bash
# FaceSwap Application Installation Script for Linux/macOS
# This script automates the installation process on Unix-like systems

set -e  # Exit on any error

echo "========================================"
echo "FaceSwap Application Installer"
echo "========================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    echo "Please install Python 3.8+ using your system package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  macOS: brew install python"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" || {
    print_error "Python 3.8 or higher is required"
    echo "Current version: $(python3 --version)"
    exit 1
}

print_success "Python version check passed!"
echo

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed"
    echo "Please install pip3 using your system package manager"
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip --user

# Install system dependencies (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Checking for system dependencies..."
    
    # Check if we're on Ubuntu/Debian
    if command -v apt &> /dev/null; then
        echo "Detected Debian/Ubuntu system"
        echo "You may need to install system packages:"
        echo "  sudo apt update"
        echo "  sudo apt install python3-dev libgl1-mesa-glx libglib2.0-0"
        echo "  sudo apt install ffmpeg  # For video processing"
        echo
    # Check if we're on CentOS/RHEL
    elif command -v yum &> /dev/null; then
        echo "Detected CentOS/RHEL system"
        echo "You may need to install system packages:"
        echo "  sudo yum install python3-devel mesa-libGL glib2"
        echo "  sudo yum install ffmpeg  # May require EPEL repository"
        echo
    fi
fi

# Install Python requirements
echo "Installing Python dependencies..."
pip3 install -r requirements.txt --user

if [ $? -ne 0 ]; then
    print_error "Failed to install dependencies"
    echo "Please check your internet connection and try again"
    exit 1
fi

# Check CUDA availability (optional)
echo
echo "Checking for CUDA support..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || {
    print_warning "Could not check CUDA availability"
    echo "GPU acceleration may not be available"
}

# Make the main script executable
chmod +x faceswap.py 2>/dev/null || true

echo
echo "========================================"
print_success "Installation completed successfully!"
echo "========================================"
echo
echo "To test the installation, run:"
echo "  python3 faceswap.py --help"
echo
echo "For GPU support, make sure you have:"
echo "  - NVIDIA GPU with CUDA support"
echo "  - NVIDIA CUDA Toolkit 11.8 or higher"
echo "  - Appropriate PyTorch CUDA version"
echo
echo "Platform-specific notes:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  macOS: GPU acceleration uses Metal Performance Shaders (MPS)"
    echo "  Install via: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "  Linux: For CUDA support, install PyTorch with CUDA:"
    echo "  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
fi
echo