#!/usr/bin/env python3
"""
Complete Object Recognition System for Drone
- Loads reference images from /pics folder
- Real-time object detection from camera
- Confidence scoring
- Visual feedback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time
from collections import deque
from pathlib import Path
import pickle
import threading

print("="*70)
print("DRONE OBJECT RECOGNITION SYSTEM - Pi5 OPTIMIZED")
print("="*70)

class FastFeatureExtractor(nn.Module):
    """Ultra-fast feature extractor for Pi 5"""
    
    def __init__(self):
        super().__init__()
        # Super lightweight CNN
        self.backbone = nn.Sequential(
            # Layer 1: 1x240x320 -> 16x120x160
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 2: 16x120x160 -> 32x60x80
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 3: 32x60x80 -> 64x30x40
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Keypoint detector
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Descriptor head (32-dim descriptors)
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        print(f"  Model parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Detect keypoints
        keypoint_scores = self.keypoint_head(features)
        
        # Compute descriptors
        descriptors = self.descriptor_head(features)
        descriptors = F.normalize(descriptors, p=2, dim=1)
        
        return keypoint_scores, descriptors

class ReferenceObject:
    """Store reference object information"""
    
    def __init__(self, name, image_path, keypoints, descriptors, scores):
        self.name = name
        self.image_path = image_path
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.scores = scores
        self.timestamp = time.time()
        
    def get_features(self):
        """Get features for matching"""
        return self.keypoints, self.descriptors, self.scores

class ObjectDetector:
    """Main object detection class"""
    
    def __init__(self, width=320, height=240, max_keypoints=200, confidence_threshold=0.6):
        self.width = width
        self.height = height
        self.max_keypoints = max_keypoints
        self.confidence_threshold = confidence_threshold
        
        # Initialize model
        self.device = torch.device('cpu')
        self.model = FastFeatureExtractor().to(self.device)
        self.model.eval()
        
        # Storage for reference objects
        self.reference_objects = {}
        self.object_features = {}  # Pre-computed features
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.detection_history = deque(maxlen=100)
        self.running = False
        
        # Camera
        self.cap = None
        
        print("✓ System initialized")
        print(f"  Resolution: {width}x{height}")
        print(f"  Max keypoints: {max_keypoints}")
        print(f"  Confidence threshold: {confidence_threshold}")
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
        else:
            gray = image
        
        # Resize
        gray = cv2.resize(gray, (self.width, self.height))
        
        # Normalize and convert to tensor
        tensor = torch.from_numpy(gray).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return tensor, gray
    
    def extract_features(self, image):
        """Extract features from an image"""
        # Preprocess
        tensor, gray = self.preprocess_image(image)
        
        # Run model
        with torch.no_grad():
            scores, descriptors = self.model(tensor)
        
        # Extract keypoints
        keypoints, confidences = self.extract_keypoints(scores)
        
        # Get descriptors for each keypoint
        desc_np = descriptors.squeeze().cpu().numpy()  # [32, H/4, W/4]
        h, w = desc_np.shape[1:]
        
        # Scale keypoints to descriptor map
        scale_y = h / self.height
        scale_x = w / self.width
        
        # Sample descriptors at keypoint locations
        sampled_descriptors = []
        for kp in keypoints:
            x_desc = int(kp[0] * scale_x)
            y_desc = int(kp[1] * scale_y)
            x_desc = min(max(x_desc, 0), w-1)
            y_desc = min(max(y_desc, 0), h-1)
            sampled_descriptors.append(desc_np[:, y_desc, x_desc])
        
        sampled_descriptors = np.array(sampled_descriptors) if sampled_descriptors else np.array([])
        
        return keypoints, sampled_descriptors, confidences, gray
    
    def extract_keypoints(self, score_map):
        """Extract keypoints from score map"""
        scores = score_map.squeeze().cpu().detach().numpy()
        
        # Find top k keypoints
        h, w = scores.shape
        flat_scores = scores.flatten()
        
        if len(flat_scores) > self.max_keypoints:
            indices = np.argpartition(flat_scores, -self.max_keypoints)[-self.max_keypoints:]
            indices = indices[np.argsort(-flat_scores[indices])]
        else:
            indices = np.argsort(-flat_scores)
        
        # Convert to coordinates
        y_coords, x_coords = np.unravel_index(indices, (h, w))
        scores_values = flat_scores[indices]
        
        # Scale coordinates to original image size
        scale_x = self.width / w
        scale_y = self.height / h
        
        keypoints = np.stack([x_coords * scale_x, y_coords * scale_y], axis=1)
        
        return keypoints, scores_values
    
    def load_reference_images(self, folder_path="pics"):
        """Load all reference images from folder"""
        folder = Path(folder_path)
        if not folder.exists():
            print(f"✗ Folder {folder_path} not found!")
            return False
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"✗ No images found in {folder_path}")
            return False
        
        print(f"\n📸 Loading {len(image_files)} reference images...")
        
        for img_path in image_files:
            name = img_path.stem
            print(f"  Processing: {name}")
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    ✗ Failed to load {img_path.name}")
                continue
            
            # Extract features
            keypoints, descriptors, scores, gray = self.extract_features(img)
            
            if len(keypoints) > 10:
                # Store reference object
                obj = ReferenceObject(
                    name=name,
                    image_path=str(img_path),
                    keypoints=keypoints,
                    descriptors=descriptors,
                    scores=scores
                )
                
                self.reference_objects[name] = obj
                self.object_features[name] = (keypoints, descriptors, scores)
                
                print(f"    ✓ Loaded {len(keypoints)} keypoints")
            else:
                print(f"    ✗ Not enough features ({len(keypoints)} keypoints)")
        
        print(f"\n✓ Loaded {len(self.reference_objects)} reference objects")
        return len(self.reference_objects) > 0
    
    def match_objects(self, query_keypoints, query_descriptors, query_scores):
        """Match query features against reference objects"""
        if len(query_keypoints) == 0:
            return []
        
        matches = []
        
        for obj_name, (ref_keypoints, ref_descriptors, ref_scores) in self.object_features.items():
            if len(ref_descriptors) == 0:
                continue
            
            # Compute similarity matrix
            # Normalize descriptors
            query_desc_norm = query_descriptors / (np.linalg.norm(query_descriptors, axis=1, keepdims=True) + 1e-8)
            ref_desc_norm = ref_descriptors / (np.linalg.norm(ref_descriptors, axis=1, keepdims=True) + 1e-8)
            
            # Compute cosine similarity
            sim_matrix = np.dot(query_desc_norm, ref_desc_norm.T)
            
            # Find best matches
            best_matches = []
            for i in range(len(query_keypoints)):
                best_idx = np.argmax(sim_matrix[i])
                best_score = sim_matrix[i][best_idx]
                if best_score > self.confidence_threshold:
                    best_matches.append({
                        'query_idx': i,
                        'ref_idx': best_idx,
                        'score': best_score,
                        'query_kp': query_keypoints[i],
                        'ref_kp': ref_keypoints[best_idx]
                    })
            
            # Calculate overall confidence
            if best_matches:
                avg_score = np.mean([m['score'] for m in best_matches])
                match_ratio = len(best_matches) / max(len(query_keypoints), 1)
                confidence = avg_score * match_ratio
                
                matches.append({
                    'object_name': obj_name,
                    'matches': best_matches,
                    'num_matches': len(best_matches),
                    'avg_score': avg_score,
                    'match_ratio': match_ratio,
                    'confidence': confidence
                })
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches
    
    def init_camera(self, camera_id=0):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if self.cap.isOpened():
            print(f"\n✓ Camera initialized ({self.width}x{self.height})")
            return True
        else:
            print("✗ Camera failed to initialize")
            return False
    
    def run_detection(self):
        """Run real-time object detection"""
        if not self.reference_objects:
            print("\n✗ No reference objects loaded!")
            print("  Please add images to the 'pics' folder first.")
            return
        
        if not self.init_camera():
            return
        
        print("\n" + "="*70)
        print("LIVE OBJECT DETECTION - Press 'q' to quit")
        print("="*70)
        print("\nDetecting objects:")
        for name in self.reference_objects.keys():
            print(f"  • {name}")
        print("")
        
        self.running = True
        frame_count = 0
        last_detection = None
        
        while self.running:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Extract features from current frame
            query_keypoints, query_descriptors, query_scores, gray = self.extract_features(frame)
            
            # Match against reference objects
            if len(query_keypoints) > 10:
                matches = self.match_objects(query_keypoints, query_descriptors, query_scores)
            else:
                matches = []
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            self.fps_history.append(fps)
            
            # Draw results
            vis_frame = frame.copy()
            
            # Draw keypoints
            for kp in query_keypoints[:100]:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(vis_frame, (x, y), 2, (0, 255, 0), -1)
            
            # Draw detection results
            if matches:
                best_match = matches[0]
                confidence = best_match['confidence']
                
                # Color based on confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green - high confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 165, 255)  # Orange - low confidence
                
                # Draw bounding box
                h, w = frame.shape[:2]
                cv2.rectangle(vis_frame, (10, 40), (w-10, h-10), color, 2)
                
                # Draw matches for best object
                for match in best_match['matches'][:20]:  # Show first 20 matches
                    x_q, y_q = int(match['query_kp'][0]), int(match['query_kp'][1])
                    cv2.circle(vis_frame, (x_q, y_q), 3, color, -1)
                
                last_detection = best_match
                
                # Show detection info
                info_text = f"Detected: {best_match['object_name']}"
                conf_text = f"Confidence: {confidence:.2f}"
                matches_text = f"Matches: {best_match['num_matches']}"
                
                cv2.putText(vis_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(vis_frame, conf_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(vis_frame, matches_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif last_detection:
                # Show last detection
                cv2.putText(vis_frame, f"Last: {last_detection['object_name']}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Show FPS and keypoints
            avg_fps = np.mean(self.fps_history)
            cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, self.height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Keypoints: {len(query_keypoints)}", (10, self.height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Drone Object Detection', vis_frame)
            
            # Update counter
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  FPS: {avg_fps:.1f} | Detected: {matches[0]['object_name'] if matches else 'None'}")
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                cv2.imwrite(f"detected/detection_{timestamp}.jpg", vis_frame)
                print(f"  📸 Saved detection")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Detection ended")

def main():
    """Main function"""
    print("\n🚀 Starting Object Recognition System...")
    
    # Create detector
    detector = ObjectDetector(
        width=320,
        height=240,
        max_keypoints=200,
        confidence_threshold=0.6
    )
    
    # Load reference images
    if not detector.load_reference_images("pics"):
        print("\n📝 Please add reference images to the 'pics' folder:")
        print("  cd pics")
        print("  # Add your images here (jpg, png)")
        print("\nExample structure:")
        print("  pics/")
        print("  ├── drone.jpg")
        print("  ├── landing_pad.jpg")
        print("  ├── obstacle.jpg")
        print("  └── target.jpg")
        return
    
    # Ask user to continue
    print("\n" + "="*70)
    response = input("Start live object detection? (y/n): ")
    
    if response.lower() == 'y':
        detector.run_detection()
    else:
        print("\nExiting. Add more images to 'pics' and try again.")
    
    print("\n✨ System ready!")

if __name__ == "__main__":
    main()
