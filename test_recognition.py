#!/usr/bin/env python3
"""
Test object recognition with static images
"""

import cv2
import sys
from pathlib import Path
from object_detector import ObjectDetector

def test_image(detector, image_path):
    """Test recognition on a single image"""
    print(f"\n🔍 Testing: {image_path}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ✗ Failed to load image")
        return
    
    # Extract features
    keypoints, descriptors, scores, gray = detector.extract_features(img)
    
    # Match against references
    matches = detector.match_objects(keypoints, descriptors, scores)
    
    if matches:
        print(f"  ✓ Found {len(matches)} potential matches")
        for i, match in enumerate(matches[:3]):
            print(f"    {i+1}. {match['object_name']} - Confidence: {match['confidence']:.2f}")
    else:
        print(f"  ✗ No matches found")
    
    # Show image with keypoints
    vis = img.copy()
    for kp in keypoints[:50]:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imshow(f"Test: {image_path.name}", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetector()
    
    if not detector.load_reference_images("pics"):
        print("Please add images to 'pics' folder first")
        sys.exit(1)
    
    # Test all images in current directory
    for img_path in Path(".").glob("*.jpg"):
        test_image(detector, img_path)
    for img_path in Path(".").glob("*.png"):
        test_image(detector, img_path)
