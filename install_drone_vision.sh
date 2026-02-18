#!/bin/bash
# /home/sweet/Desktop/thedronecnn/install_drone_vision.sh

echo "🚁 Installing Drone Vision System for Pi 5"

# Activate virtual environment
source /home/sweet/Desktop/thedronecnn/.venv/bin/activate

# Install basic dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy

# Create the vision module
cat > ultra_light_matcher.py << 'EOF'
# PASTE THE ENTIRE CODE FROM ABOVE HERE
EOF

echo "✅ Installation complete!"
echo ""
echo "To test:"
echo "  source .venv/bin/activate"
echo "  python ultra_light_matcher.py"  