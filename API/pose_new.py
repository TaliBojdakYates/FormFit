import numpy as np
import math
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # Import Savitzky-Golay filter

def moving_average_filter(points_history, window_size=3):
    """Apply moving average filter to a list of points"""
    if len(points_history) < window_size:
        return points_history[-1]
    
    recent_points = points_history[-window_size:]
    return np.mean(recent_points, axis=0)

def calculate_knee_angle(keypoints, fixed_left_ankle, fixed_right_ankle):
    try:
        # Left side keypoints
        left_hip = keypoints[11][:2]
        left_knee = keypoints[13][:2]
        left_ankle = fixed_left_ankle

        # Right side keypoints
        right_hip = keypoints[12][:2]
        right_knee = keypoints[14][:2]
        right_ankle = fixed_right_ankle

        def calculate_angle(hip, knee, ankle):
            hip_to_knee = [knee[0]-hip[0], knee[1]-hip[1]]
            knee_to_ankle = [ankle[0]-knee[0], ankle[1]-knee[1]]
            
            hip_to_knee_mag = np.linalg.norm(hip_to_knee)
            knee_to_ankle_mag = np.linalg.norm(knee_to_ankle)
            
            if hip_to_knee_mag == 0 or knee_to_ankle_mag == 0:
                return None
            
            dot_product = np.dot(hip_to_knee, knee_to_ankle)
            cos_angle = dot_product / (hip_to_knee_mag * knee_to_ankle_mag)
            return math.degrees(math.acos(np.clip(cos_angle, -1, 1)))

        # Calculate angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        return left_knee_angle, right_knee_angle
    except Exception as e:
        print(f"Error in angle calculation: {e}")
        return None, None

def process_frame_with_filtering(frame, model, keypoint_history, confidence_threshold=0.3, fixed_left_ankle=None, fixed_right_ankle=None):
    results = model(frame)
    
    if results[0].keypoints.xy.shape[1] > 0:
        # Get keypoints and confidence scores
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        conf_scores = results[0].keypoints.conf[0].cpu().numpy()
        
        # Apply filtering to each keypoint
        filtered_keypoints = keypoints.copy()
        
        for i, point in enumerate(keypoints):
            # Skip points with low confidence
            if conf_scores[i] < confidence_threshold:
                continue
            
            # Add current raw point to history
            if i not in keypoint_history:
                keypoint_history[i] = []
            keypoint_history[i].append(point[:2])
            
            # Limit history size
            if len(keypoint_history[i]) > 30:
                keypoint_history[i].pop(0)
        
        # Calculate knee angles using fixed ankle positions
        left_knee_angle, right_knee_angle = calculate_knee_angle(filtered_keypoints, fixed_left_ankle, fixed_right_ankle)
        
        return left_knee_angle, right_knee_angle
    
    return frame, None, None, None

def calculate_similarity_score(left_angles, right_angles, tolerance=1):
  
    if len(left_angles) != len(right_angles) or len(left_angles) == 0:
        return 0
   
    correlation = np.corrcoef(left_angles, right_angles)[0, 1]
    correlation_score = (correlation + 1) / 2  # Normalize to 0-1 (1 is perfect match)
 
    return int(correlation_score * 100)

def process_video(video_stream):
    model = YOLO("yolo11n-pose.pt")
  
    cap = cv2.VideoCapture()
    cap.open(video_stream)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None, None, 0  # Return 0 score if video can't be opened

    keypoint_history = {}  
    fixed_left_ankle = None
    fixed_right_ankle = None

    frame_count = 0
    max_init_frames = 10
    
    while frame_count < max_init_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        
        if results[0].keypoints.xy.shape[1] > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            conf_scores = results[0].keypoints.conf[0].cpu().numpy()
            
            if fixed_left_ankle is None and conf_scores[15] > 0.5:
                fixed_left_ankle = keypoints[15][:2].copy()  
            
            if fixed_right_ankle is None and conf_scores[16] > 0.5:
                fixed_right_ankle = keypoints[16][:2].copy() 

            if fixed_left_ankle is not None and fixed_right_ankle is not None:
                break
        
        frame_count += 1
    
    cap.release()
    cap = cv2.VideoCapture(video_stream)
    
    left_knee = []
    right_knee = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left_knee_angle, right_knee_angle = process_frame_with_filtering(
            frame, model, keypoint_history,
            confidence_threshold=0.3, 
            fixed_left_ankle=fixed_left_ankle, 
            fixed_right_ankle=fixed_right_ankle
        )
        
        if left_knee_angle is not None and right_knee_angle is not None:
            left_knee.append(left_knee_angle)
            right_knee.append(right_knee_angle)
    
    cap.release()
    cv2.destroyAllWindows()

    if not left_knee or not right_knee:
        return None, None, 0  # Return 0 score if no valid angles

    # Apply Savitzky-Golay filter to smooth the knee angles
    left_knee_smoothed = savgol_filter(left_knee, window_length=11, polyorder=2)
    right_knee_smoothed = savgol_filter(right_knee, window_length=11, polyorder=2)

    # Downsample smoothed angles
    left_knee_downsampled = np.array(left_knee_smoothed[::4])
    right_knee_downsampled = np.array(right_knee_smoothed[::4])

    # Calculate similarity score
    score = calculate_similarity_score(left_knee_downsampled, right_knee_downsampled, tolerance=10)

    return left_knee_downsampled, right_knee_downsampled, score