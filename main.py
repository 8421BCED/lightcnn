#!/usr/bin/env python3
"""
Drone Object Recognition System - Main Launcher
"""

import os
import sys
from object_detector import ObjectDetector

def main():
    print("="*70)
    print("DRONE OBJECT RECOGNITION SYSTEM")
    print("="*70)
    print("\n1. Load reference images from 'pics' folder")
    print("2. Start live object detection")
    print("3. Test with single image")
    print("4. Exit")
    print("="*70)
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        print("\n📸 Loading reference images...")
        detector = ObjectDetector()
        if detector.load_reference_images("pics"):
            print(f"\n✓ Successfully loaded {len(detector.reference_objects)} objects")
        else:
            print("\n✗ No images found in 'pics' folder")
        input("\nPress Enter to continue...")
    
    elif choice == '2':
        # Run live detection
        detector = ObjectDetector()
        if detector.load_reference_images("pics"):
            detector.run_detection()
        else:
            print("\n✗ Please add images to 'pics' folder first")
            input("\nPress Enter to continue...")
    
    elif choice == '3':
        # Test with image
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            detector = ObjectDetector()
            detector.load_reference_images("pics")
            
            # Load and test image
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                keypoints, descriptors, scores, gray = detector.extract_features(img)
                matches = detector.match_objects(keypoints, descriptors, scores)
                
                print(f"\n📊 Results for {os.path.basename(image_path)}:")
                if matches:
                    for i, match in enumerate(matches[:3]):
                        print(f"  {i+1}. {match['object_name']} - Confidence: {match['confidence']:.2f}")
                else:
                    print("  No matches found")
                
                # Show image
                vis = img.copy()
                for kp in keypoints[:50]:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                
                cv2.imshow('Test Image', vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("✗ Cannot load image")
        else:
            print("✗ File not found!")
        input("\nPress Enter to continue...")
    
    elif choice == '4':
        print("\n👋 Goodbye!")
        return
    
    else:
        print("\n❌ Invalid choice!")
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        main()