#!/usr/bin/env python3
"""
ULTRA-FAST INDUSTRIAL OBJECT DETECTION - 30 FPS on Pi 5
Optimized for maximum performance
"""

import torch
import numpy as np
import cv2
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Suppress Qt warnings
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Add LightGlue to path
LIGHTGLUE_PATH = os.path.join(os.path.dirname(__file__), 'LightGlue')
sys.path.append(LIGHTGLUE_PATH)

# Import LightGlue
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

print("="*80)
print("🚀 ULTRA-FAST INDUSTRIAL DETECTION - 30 FPS OPTIMIZED")
print("="*80)

class UltraFastDetector:
    """
    Optimized for 30 FPS on Pi 5
    """
    
    def __init__(self, 
                 reference_dir="references",
                 width=640,
                 height=480,
                 confidence_threshold=0.2,
                 min_matches=10,  # Reduced for speed
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
        
        # Initialize LightGlue with OPTIMIZED settings
        print("\n🔧 Initializing LightGlue (Optimized)...")
        self.device = torch.device('cpu')
        
        # Use SuperPoint with reduced keypoints for speed
        self.extractor = SuperPoint(
            max_num_keypoints=512,  # Reduced for speed
            detection_threshold=0.0005
        ).eval().to(self.device)
        
        # Use LightGlue with OPTIMIZED settings
        self.matcher = LightGlue(
            features='superpoint',
            depth_confidence=0.9,  # Slightly reduced for speed
            width_confidence=0.95,  # Slightly reduced for speed
            flash=False
        ).eval().to(self.device)
        
        print(f"  ✓ Device: {self.device}")
        print(f"  ✓ Max keypoints: 512")
        print(f"  ✓ Target FPS: 30")
        
        # Load reference objects
        self.reference_objects = self.load_references()
        print(f"\n📸 Loaded {len(self.reference_objects)} reference objects")
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.frame_times = deque(maxlen=30)
        
    def load_references(self):
        """Load reference images"""
        references = {}
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        reference_images = []
        for ext in image_extensions:
            reference_images.extend(self.reference_dir.glob(f"*{ext}"))
            reference_images.extend(self.reference_dir.glob(f"*{ext.upper()}"))
        
        if not reference_images:
            print(f"\n⚠ No images in {self.reference_dir}")
            return references
        
        print(f"\n📸 Loading {len(reference_images)} references...")
        
        for img_path in reference_images:
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Convert to RGB and tensor
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = torch.from_numpy(image_rgb / 255.).float()[None].permute(0, 3, 1, 2)
                
                # Extract features
                with torch.no_grad():
                    feats = self.extractor.extract(image_tensor)
                
                references[img_path.stem] = {
                    'features': feats,
                    'name': img_path.stem
                }
                
                kp_count = feats['keypoints'].shape[1]
                print(f"  ✓ {img_path.name}: {kp_count} keypoints")
                
            except Exception as e:
                print(f"  ✗ Error: {img_path.name}")
        
        return references
    
    def process_frame_fast(self, frame):
        """Ultra-fast frame processing"""
        # Convert to RGB and resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_resized / 255.).float()[None].permute(0, 3, 1, 2)
        
        return frame_tensor, frame_resized
    
    def detect_fast(self, frame):
        """Fast detection - optimized for 30 FPS"""
        start_time = time.time()
        detections = []
        
        # Process frame
        frame_tensor, frame_resized = self.process_frame_fast(frame)
        
        # Extract features
        with torch.no_grad():
            frame_feats = self.extractor.extract(frame_tensor)
        
        # Quick check - if not enough features, skip matching
        if frame_feats['keypoints'].shape[1] < self.min_matches:
            fps = 1.0 / (time.time() - start_time)
            return [], frame_resized, None, fps
        
        # Match against references
        for obj_name, ref_data in self.reference_objects.items():
            try:
                # Fast matching
                with torch.no_grad():
                    matches_dict = self.matcher({
                        'image0': ref_data['features'],
                        'image1': frame_feats
                    })
                
                matches = rbd(matches_dict)
                matches0 = matches['matches0'][0].cpu().numpy()
                
                # Count matches
                valid = matches0 > -1
                num_matches = valid.sum()
                
                if num_matches >= self.min_matches:
                    # Get match scores
                    scores = matches['scores'][0].cpu().numpy()
                    avg_score = scores[valid].mean() if num_matches > 0 else 0
                    confidence = avg_score * (num_matches / 100)
                    
                    if confidence > self.confidence_threshold:
                        # Get keypoints for bounding box
                        kpts = frame_feats['keypoints'][0].cpu().numpy()
                        matched_kpts = kpts[matches0[valid]]
                        
                        if len(matched_kpts) > 3:
                            x_min = int(matched_kpts[:, 0].min())
                            x_max = int(matched_kpts[:, 0].max())
                            y_min = int(matched_kpts[:, 1].min())
                            y_max = int(matched_kpts[:, 1].max())
                            
                            detections.append({
                                'name': obj_name,
                                'confidence': float(confidence),
                                'matches': num_matches,
                                'bbox': [x_min, y_min, x_max, y_max],
                                'keypoints': matched_kpts
                            })
                            
            except:
                continue
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        self.fps_history.append(fps)
        self.frame_times.append(time.time() - start_time)
        
        return detections, frame_resized, frame_feats, fps
    
    def draw_results(self, frame, detections, frame_feats, fps):
        """Draw results without Qt dependencies"""
        vis = frame.copy()
        
        # Draw keypoints (optional - disable for more FPS)
        if frame_feats is not None and False:  # Disabled for speed
            kpts = frame_feats['keypoints'][0].cpu().numpy()
            for kp in kpts[:50]:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(vis, (x, y), 2, (128, 128, 128), -1)
        
        # Draw detections
        for det in detections:
            # Color by confidence
            conf = det['confidence']
            if conf > 0.8:
                color = (0, 255, 0)
            elif conf > 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
            
            # Draw bounding box
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw matches
            for kp in det['keypoints']:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(vis, (x, y), 3, color, -1)
            
            # Add label
            label = f"{det['name']} {conf:.2f}"
            cv2.putText(vis, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw FPS (simple, no fancy fonts)
        cv2.putText(vis, f"{fps:.1f} FPS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detection count
        cv2.putText(vis, f"{len(detections)} detections", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
    
    def run(self):
        """Main loop - OPTIMIZED for 30 FPS"""
        if not self.reference_objects:
            print("\n❌ No references loaded!")
            return
        
        print("\n🎥 Opening camera...")
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print(f"✓ Camera OK ({self.width}x{self.height})")
        print("\n🚀 RUNNING - Press 'q' to quit, 's' to save")
        print("="*80)
        
        frame_count = 0
        last_fps_print = time.time()
        
        try:
            while self.running:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect (optimized)
                detections, frame_resized, frame_feats, fps = self.detect_fast(frame)
                
                # Draw
                vis = self.draw_results(frame_resized, detections, frame_feats, fps)
                
                # Show
                cv2.imshow('Industrial Detection - 30 FPS', vis)
                
                # Update counter
                frame_count += 1
                
                # Print FPS every second
                current_time = time.time()
                if current_time - last_fps_print >= 1.0:
                    avg_fps = np.mean(list(self.fps_history)) if self.fps_history else 0
                    print(f"  FPS: {avg_fps:.1f} | Detections: {len(detections)} | Frames: {frame_count}")
                    last_fps_print = current_time
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and detections:
                    # Save detection
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"detections/detection_{timestamp}.jpg", vis)
                    print(f"  💾 Saved detection")
                    
        except Exception as e:
            print(f"Error: {e}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Summary
            avg_fps = np.mean(list(self.fps_history)) if self.fps_history else 0
            print("\n" + "="*80)
            print(f"✅ DETECTION COMPLETE")
            print(f"   Total frames: {frame_count}")
            print(f"   Average FPS: {avg_fps:.1f}")
            print("="*80)

def main():
    """Main"""
    # Create detector
    detector = UltraFastDetector(
        reference_dir="references",
        width=640,
        height=480,
        confidence_threshold=0.2,
        min_matches=10
    )
    
    # Run
    detector.run()

if __name__ == "__main__":
    main()
