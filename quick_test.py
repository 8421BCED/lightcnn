#!/usr/bin/env python3
"""
Quick test for your images
"""

import cv2
import os
from pathlib import Path

print("="*60)
print("QUICK IMAGE TEST")
print("="*60)

# Check pics folder
pics_dir = Path("pics")
if not pics_dir.exists():
    print("❌ 'pics' folder not found!")
    exit(1)

# List all images
images = list(pics_dir.glob("*.png")) + list(pics_dir.glob("*.jpg")) + list(pics_dir.glob("*.jpeg"))

print(f"\n📸 Found {len(images)} images:")
for i, img_path in enumerate(images):
    print(f"  {i+1}. {img_path.name}")

print("\n🔍 Testing image loading...")
for img_path in images[:3]:  # Test first 3
    img = cv2.imread(str(img_path))
    if img is not None:
        h, w = img.shape[:2]
        print(f"  ✓ {img_path.name}: {w}x{h}")
    else:
        print(f"  ❌ {img_path.name}: Failed to load")

print("\n✅ Done!")
