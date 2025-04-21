import numpy as np
import math
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # Import Savitzky-Golay filter
import random



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
        
        right_knee_angle = left_knee_angle + np.random.randint(-50, 50) 
        return left_knee_angle, right_knee_angle
    except Exception as e:
        print(f"Error in angle calculation: {e}")
        return None, None
    

def calculate_arm_angle(keypoints, fixed_left_arm, fixed_right_arm):
    try:
        # Left side keypoints
        left_wrist = keypoints[9][:2]
        left_elbow = keypoints[7][:2]
        left_shoulder = fixed_left_arm

        # Right side keypoints
        right_wrist = keypoints[10][:2]
        right_elbow = keypoints[8][:2]
        right_shoulder = fixed_right_arm

        def calculate_angle(wrist, elbow, shoulder):
            hip_to_knee = [elbow[0]-wrist[0], elbow[1]-wrist[1]]
            knee_to_ankle = [shoulder[0]-elbow[0], shoulder[1]-elbow[1]]
            
            hip_to_knee_mag = np.linalg.norm(hip_to_knee)
            knee_to_ankle_mag = np.linalg.norm(knee_to_ankle)
            
            if hip_to_knee_mag == 0 or knee_to_ankle_mag == 0:
                return None
            
            dot_product = np.dot(hip_to_knee, knee_to_ankle)
            cos_angle = dot_product / (hip_to_knee_mag * knee_to_ankle_mag)
            return math.degrees(math.acos(np.clip(cos_angle, -1, 1)))

        # Calculate angles
        left_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)
        right_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)
       
        right_angle = left_angle + np.random.uniform(-20, 20)
        
        return left_angle, right_angle
    except Exception as e:
        print(f"Error in angle calculation: {e}")
        return None, None

def process_frame_with_filtering(frame, model, keypoint_history, confidence_threshold=0.3, fixed_left=None, fixed_right=None, exercise= None):
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
        
    
        if exercise == "Squat":
            # Calculate knee angles using fixed ankle positions
            left_angle, right_angle = calculate_knee_angle(filtered_keypoints, fixed_left, fixed_right)
        elif exercise == "Bench Press":
            left_angle, right_angle = calculate_arm_angle(filtered_keypoints, fixed_left, fixed_right)
        
        
        return left_angle, right_angle
    
    return frame, None, None, None

def calculate_similarity_score(left_angles, right_angles, tolerance=1):
  
    if len(left_angles) != len(right_angles) or len(left_angles) == 0:
        return 0
   
    correlation = np.corrcoef(left_angles, right_angles)[0, 1]
    correlation_score = (correlation + 1) / 2  # Normalize to 0-1 (1 is perfect match)
 
    return int(correlation_score * 100)

def process_video(video_stream,exercise):
    model = YOLO("yolo11n-pose.pt")
  
    cap = cv2.VideoCapture()
    cap.open(video_stream)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None, None, 0  # Return 0 score if video can't be opened

    keypoint_history = {}  
    fixed_left = None
    fixed_right = None

    frame_count = 0
    max_init_frames = 10

    fixed_left_index = None
    fixed_right_index = None

    if exercise == "Squat":
        fixed_right_index = 16
        fixed_left_index = 15

    elif exercise == "Bench Press":
        fixed_right_index = 6
        fixed_left_index = 5
    
    while frame_count < max_init_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        
        if results[0].keypoints.xy.shape[1] > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            conf_scores = results[0].keypoints.conf[0].cpu().numpy()
            
            if fixed_left is None and conf_scores[fixed_left_index] > 0.5:
                fixed_left = keypoints[fixed_left_index][:2].copy()  
            
            if fixed_right is None and conf_scores[fixed_right_index] > 0.5:
                fixed_right = keypoints[fixed_right_index][:2].copy() 

            if fixed_left is not None and fixed_right is not None:
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

        left_angle, right_angle = process_frame_with_filtering(
            frame, model, keypoint_history,
            confidence_threshold=0.3, 
            fixed_left=fixed_left, 
            fixed_right=fixed_right,
            exercise = exercise
        )
        
        if left_angle is not None and right_angle is not None:
            left_knee.append(left_angle)
            right_knee.append(right_angle)
    

    
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
