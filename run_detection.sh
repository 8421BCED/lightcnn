#!/bin/bash
# Production launcher for industrial detection

echo "🚀 Starting Industrial LightGlue Detection"
echo "=========================================="

# Activate virtual environment
source /home/sweet/Desktop/thedronecnn/.venv/bin/activate

# Run detection
python industrial_detector.py

# Keep terminal open if there's an error
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error occurred. Press Enter to exit."
    read
fi
