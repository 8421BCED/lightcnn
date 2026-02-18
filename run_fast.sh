#!/bin/bash
# Ultra-fast detection script

echo "🚀 ULTRA-FAST DETECTION - 30 FPS TARGET"
echo "========================================"

# Set optimizations
export QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Activate venv
source /home/sweet/Desktop/thedronecnn/.venv/bin/activate

# Run with optimizations
python -O industrial_detector_fast.py

# Keep open if error
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error. Press Enter to exit."
    read
fi
