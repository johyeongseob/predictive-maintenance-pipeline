#!/bin/bash
# PACE Complete Setup Script
# Sets up the PACE system with a single unified conda environment

set -e  # Exit on any error

echo "============================================================"
echo "PACE - Complete System Setup (Unified Environment)"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# ============================================================
# Step 1: Check Prerequisites
# ============================================================
echo "Step 1: Checking prerequisites..."
echo ""

# Check conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ Conda not found${NC}"
    echo "  Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}✓ Conda found${NC}"

echo ""

# ============================================================
# Step 2: Create/Activate Conda Environment
# ============================================================
echo "Step 2: Setting up unified 'pace' conda environment..."
echo ""

# Activate conda for this script
eval "$(conda shell.bash hook)"

# --- Setup pace environment ---
echo "Setting up 'pace' environment (inference, agents, and conversion)..."
if conda env list | grep -q "^pace "; then
    echo -e "${YELLOW}⚠ Environment 'pace' already exists${NC}"
    read -p "  Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Removing existing environment..."
        conda env remove -n pace -y
        echo "  Creating 'pace' environment..."
        conda create -n pace python=3.10 -y
        conda activate pace
        pip install --upgrade pip
        pip install -r setup/requirements.txt
    else
        echo "  Using existing environment..."
    fi
else
    echo "Creating 'pace' environment..."
    conda create -n pace python=3.10 -y
    conda activate pace
    pip install --upgrade pip
    pip install -r setup/requirements.txt
fi
echo -e "${GREEN}✓ 'pace' environment ready${NC}"
echo ""

# ============================================================
# Step 3: Verify OpenVINO Devices
# ============================================================
echo "Step 3: Verifying OpenVINO devices..."
echo ""

# Activate runtime environment
conda activate pace

python -c "
import openvino as ov
core = ov.Core()
devices = core.available_devices
print('Available OpenVINO devices:', devices)
if 'GPU' in devices:
    print('✓ Intel GPU (iGPU) available')
if 'CPU' in devices:
    print('✓ CPU available')
if 'NPU' in devices:
    print('✓ Intel NPU available')
"

echo ""

# ============================================================
# Step 4: Create Data and Database Directories
# ============================================================
echo "Step 4: Creating data directories..."
echo ""

mkdir -p out
echo -e "${GREEN}✓ SQL data directory created (out/)${NC}"
echo ""

# ============================================================
# Step 5: Create Output Directories
# ============================================================
echo "Step 5: Creating output directories..."
echo ""

mkdir -p out

echo -e "${GREEN}✓ Output directories created (per-use-case dirs created at runtime)${NC}"
echo ""

# ============================================================
# Setup Complete
# ============================================================
echo "============================================================"
echo -e "${GREEN}✓ PACE Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "A unified conda environment has been created:"
echo "  - pace: For inference, agents, model conversion, and runtime"
echo ""
echo "Next steps:"
echo ""
echo "  See docs/user-guide/QUICKSTART.md for:"
echo "    - Downloading and preparing datasets"
echo "    - Training a YOLO model"
echo "    - Converting the model to OpenVINO"
echo "    - Downloading and exporting LLMs"
echo "    - Running the pipeline"
echo ""
echo "For more details, see README.md and docs/user-guide/QUICKSTART.md"
echo ""
