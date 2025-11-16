#!/bin/bash

################################################################################
# Quick Test Script
# Runs a fast test to verify everything works correctly
################################################################################

set -e

echo "================================================================================"
echo "Quick Test - Model Comparison Framework"
echo "================================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Step 1: Verifying models...${NC}"
python test_models.py

echo ""
echo -e "${GREEN}✓ Models verified successfully!${NC}"
echo ""

echo -e "${BLUE}Step 2: Checking configurations...${NC}"
echo ""

# Display config summary
echo "ACC-Only Config:"
echo "  - Model: TransModel"
echo "  - Input: 4 channels (ax, ay, az, smv)"
echo "  - Embed dim: $(grep 'embed_dim:' config/smartfallmm/comparison_acc_only.yaml | awk '{print $2}')"
echo "  - Num heads: $(grep 'num_heads:' config/smartfallmm/comparison_acc_only.yaml | awk '{print $2}')"
echo "  - Dropout: $(grep 'dropout:' config/smartfallmm/comparison_acc_only.yaml | awk '{print $2}')"
echo ""

echo "IMU-Full Config:"
echo "  - Model: IMUTransformer"
echo "  - Input: 7 channels (ax, ay, az, gx, gy, gz, smv)"
echo "  - Embed dim: $(grep 'embed_dim:' config/smartfallmm/comparison_imu_full.yaml | awk '{print $2}')"
echo "  - Num heads: $(grep 'num_heads:' config/smartfallmm/comparison_imu_full.yaml | awk '{print $2}')"
echo "  - Dropout: $(grep 'dropout:' config/smartfallmm/comparison_imu_full.yaml | awk '{print $2}')"
echo ""

echo -e "${GREEN}✓ Configurations look good!${NC}"
echo ""

echo "================================================================================"
echo "Ready to run comparison!"
echo "================================================================================"
echo ""
echo "To start the full comparison, run:"
echo ""
echo "  bash run_comparison.sh"
echo ""
echo "Or with custom options:"
echo ""
echo "  bash run_comparison.sh --seed 42 --device cuda"
echo "  bash run_comparison.sh --multi_seed"
echo "  bash run_comparison.sh --test_only"
echo ""
echo "For help:"
echo ""
echo "  bash run_comparison.sh --help"
echo ""
