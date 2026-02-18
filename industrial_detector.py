#!/usr/bin/env python3
"""
INDUSTRIAL-GRADE OBJECT DETECTION SYSTEM
Using LightGlue - Production Version
No menus, no demos - Direct detection
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import threading
from collections import deque
import signal

# Add LightGlue to path
LIGHTGLUE_PATH = os.path.join(os.path.dirname(__file__), 'LightGlue')
sys.path.append(LIGHTGLUE_PATH)

# Import LightGlue
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import rbd

print("="*80)
print("🏭 INDUSTRIAL OBJECT DETECTION - LIGHTGLUE POWERED")
print("="*80)

class IndustrialDetector:
    """
    Production-ready object detection with LightGlue
    """
    
    def __init__(self, 
                 reference_dir="references",  # Directory with reference images
                 width=640,
                 height=480,
                 confidence_threshold=0.2,     # LightGlue is very accurate
                 min_matches=15,                # Minimum matches for detection
                 camera_id=0):
        
        self.reference_dir = Path(reference_dir)
        self.width = width
        self.height = height
        self.confidence_threshold = confidence_threshold
        self.min_matches = min_matches
        self.camera_id = camera_id
        self.running = True
        
        # Create directories
        self.reference_dir.mkdir(exist_ok=True)
        self.detections_dir = Path("detections")
        self.detections_dir.mkdir(exist_ok=True)
        
        # Initialize LightGlue
        print("\n🔧 Initializing LightGlue...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use SuperPoint feature extractor
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        
        # Use LightGlue matcher
        self.matcher = LightGlue(
            features='superpoint',
            depth_confidence=0.95,
            width_confidence=0.99,
            flash=False  # Disable flash for CPU
        ).eval().to(self.device)
        
        print(f"  ✓ Device: {self.device}")
        print(f"  ✓ Extractor: SuperPoint")
        print(f"  ✓ Matcher: LightGlue")
        
        # Load reference objects
        self.reference_objects = self.load_references()
        print(f"\n📸 Loaded {len(self.reference_objects)} reference objects")
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.detection_history = []
        
        # Detection lock
        self.lock = threading.Lock()
        
    def load_references(self):
        """Load all reference images and extract features"""
        references = {}
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        reference_images = []
        for ext in image_extensions:
            reference_images.extend(self.reference_dir.glob(f"*{ext}"))
            reference_images.extend(self.reference_dir.glob(f"*{ext.upper()}"))
        
        if not reference_images:
            print(f"⚠ No reference images found in {self.reference_dir}")
            print(f"  Please add images to: {self.reference_dir.absolute()}")
            return references
        
        print(f"\n📸 Processing {len(reference_images)} reference images...")
        
        for img_path in reference_images:
            try:
                # Load and preprocess image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"  ✗ Cannot load: {img_path.name}")
                    continue
                
                # Convert to RGB and tensor
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = torch.from_numpy(image_rgb / 255.).float()[None].permute(0, 3, 1, 2)
                image_tensor = image_tensor.to(self.device)
                
                # Extract features
                with torch.no_grad():
                    feats = self.extractor.extract(image_tensor)
                
                # Store reference
                references[img_path.stem] = {
                    'path': str(img_path),
                    'features': feats,
                    'image_size': (image.shape[1], image.shape[0])
                }
                
                print(f"  ✓ {img_path.name}: {feats['keypoints'].shape[1]} keypoints")
                
            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")
        
        return references
    
    def preprocess_frame(self, frame):
        """Preprocess camera frame for LightGlue"""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_resized / 255.).float()[None].permute(0, 3, 1, 2)
        frame_tensor = frame_tensor.to(self.device)
        
        return frame_tensor, frame_resized
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        start_time = time.time()
        detections = []
        
        # Preprocess frame
        frame_tensor, frame_resized = self.preprocess_frame(frame)
        
        # Extract features from frame
        with torch.no_grad():
            frame_feats = self.extractor.extract(frame_tensor)
        
        # Match against each reference
        for obj_name, ref_data in self.reference_objects.items():
            try:
                # Match features
                with torch.no_grad():
                    matches_dict = self.matcher({
                        'image0': ref_data['features'],
                        'image1': frame_feats
                    })
                
                # Remove batch dimension
                matches = rbd(matches_dict)
                
                # Get matches
                matches0 = matches['matches0'][0].cpu().numpy()
                matches1 = matches['matches1'][0].cpu().numpy()
                match_scores = matches['scores'][0].cpu().numpy()
                
                # Count valid matches
                valid_matches = matches0 > -1
                num_matches = valid_matches.sum()
                
                if num_matches >= self.min_matches:
                    # Calculate confidence
                    avg_score = match_scores[valid_matches].mean() if num_matches > 0 else 0
                    confidence = avg_score * (num_matches / self.max_keypoints)
                    
                    if confidence > self.confidence_threshold:
                        # Get matched keypoints
                        kpts_ref = ref_data['features']['keypoints'][0].cpu().numpy()
                        kpts_frame = frame_feats['keypoints'][0].cpu().numpy()
                        
                        matched_ref_pts = kpts_ref[matches0[valid_matches]]
                        matched_frame_pts = kpts_frame[matches1[matches0[valid_matches]]]
                        
                        # Calculate bounding box
                        if len(matched_frame_pts) > 3:
                            x_min = int(matched_frame_pts[:, 0].min())
                            x_max = int(matched_frame_pts[:, 0].max())
                            y_min = int(matched_frame_pts[:, 1].min())
                            y_max = int(matched_frame_pts[:, 1].max())
                            
                            # Add padding
                            pad = 20
                            x_min = max(0, x_min - pad)
                            x_max = min(self.width, x_max + pad)
                            y_min = max(0, y_min - pad)
                            y_max = min(self.height, y_max + pad)
                        else:
                            x_min, y_min, x_max, y_max = 0, 0, self.width, self.height
                        
                        detections.append({
                            'object_name': obj_name,
                            'confidence': float(confidence),
                            'num_matches': int(num_matches),
                            'bbox': [x_min, y_min, x_max, y_max],
                            'matches': matched_frame_pts,
                            'match_scores': match_scores[valid_matches].tolist()
                        })
                        
            except Exception as e:
                print(f"⚠ Match error for {obj_name}: {e}")
                continue
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        self.fps_history.append(fps)
        
        return detections, frame_resized, frame_feats, fps
    
    def draw_detections(self, frame, detections, frame_feats, fps):
        """Draw detection results on frame"""
        vis = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw all keypoints (light gray)
        if frame_feats is not None:
            kpts = frame_feats['keypoints'][0].cpu().numpy()
            for kp in kpts:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(vis, (x, y), 2, (200, 200, 200), -1)
        
        # Draw detections
        for detection in detections:
            # Color based on confidence
            conf = detection['confidence']
            if conf > 0.8:
                color = (0, 255, 0)      # Green - excellent
            elif conf > 0.5:
                color = (0, 255, 255)    # Yellow - good
            elif conf > 0.3:
                color = (0, 165, 255)    # Orange - fair
            else:
                color = (0, 0, 255)       # Red - poor
            
            # Draw bounding box
            x_min, y_min, x_max, y_max = detection['bbox']
            cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color, 3)
            
            # Draw matches
            for match_pt in detection['matches']:
                x, y = int(match_pt[0]), int(match_pt[1])
                cv2.circle(vis, (x, y), 4, color, -1)
            
            # Add label
            label = f"{detection['object_name']} ({detection['confidence']:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis, 
                         (x_min, y_min - label_size[1] - 10),
                         (x_min + label_size[0] + 10, y_min),
                         color, -1)
            
            # Draw label text
            cv2.putText(vis, label, (x_min + 5, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detection count
        cv2.putText(vis, f"Detections: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
    
    def save_detection(self, frame, detections):
        """Save detection results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save image
        img_path = self.detections_dir / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(img_path), frame)
        
        # Save metadata
        meta_path = self.detections_dir / f"detection_{timestamp}.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'detections': detections,
                'fps': np.mean(list(self.fps_history)) if self.fps_history else 0
            }, f, indent=2)
        
        print(f"  💾 Saved detection: {img_path.name}")
    
    def run(self):
        """Main detection loop"""
        if not self.reference_objects:
            print("\n❌ No reference objects loaded. Exiting.")
            return
        
        print("\n🎥 Initializing camera...")
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print(f"✓ Camera initialized ({self.width}x{self.height})")
        print("\n🚀 Starting detection - Press 'q' to quit, 's' to save")
        print("="*80)
        
        frame_count = 0
        detection_count = 0
        
        # Signal handler for clean exit
        def signal_handler(sig, frame):
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            while self.running:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("⚠ Lost camera connection")
                    break
                
                # Detect objects
                detections, frame_resized, frame_feats, fps = self.detect_objects(frame)
                
                # Update counters
                frame_count += 1
                if detections:
                    detection_count += 1
                
                # Draw results
                vis = self.draw_detections(frame_resized, detections, frame_feats, fps)
                
                # Show frame
                cv2.imshow('Industrial Object Detection - LightGlue', vis)
                
                # Print status every 30 frames
                if frame_count % 30 == 0:
                    print(f"  Frame {frame_count}: {fps:.1f} FPS | Detections: {len(detections)}")
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Stopping detection")
                    break
                elif key == ord('s'):
                    if detections:
                        self.save_detection(vis, detections)
                    else:
                        print("  ⚠ No detections to save")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Print summary
            print("\n" + "="*80)
            print("📊 DETECTION SUMMARY")
            print("="*80)
            print(f"  Total frames: {frame_count}")
            print(f"  Frames with detections: {detection_count}")
            print(f"  Detection rate: {(detection_count/frame_count)*100:.1f}%")
            print(f"  Average FPS: {np.mean(list(self.fps_history)):.1f}")
            print("="*80)

def main():
    """Main entry point"""
    print("\n🏭 INDUSTRIAL OBJECT DETECTION SYSTEM")
    print("Using LightGlue for maximum accuracy")
    
    # Configuration
    config = {
        'reference_dir': 'references',    # Put your images here
        'width': 640,                      # Resolution
        'height': 480,
        'confidence_threshold': 0.2,       # LightGlue threshold
        'min_matches': 15,                  # Minimum matches required
        'camera_id': 0                       # Camera ID
    }
    
    # Create detector
    detector = IndustrialDetector(**config)
    
    # Check if we have references
    if not detector.reference_objects:
        print(f"\n📸 Please add reference images to: {detector.reference_dir.absolute()}")
        print("   Supported formats: jpg, jpeg, png, bmp")
        return
    
    # Start detection
    detector.run()

if __name__ == "__main__":
    main()
