from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import tempfile
import os
from ultralytics import YOLO
from pose_new import process_video

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the YOLO model
model = YOLO("yolo11n-pose.pt")

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'message': 'No file part'}), 400
        
    file = request.files['video']
    exercise = request.form.get('exercise')
  
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
  
    exercise = "Squat"
    temp_filename = None
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_filename = temp_file.name
        file.save(temp_filename)
        temp_file.close()  # Close the file handle explicitly
        
        # Process the video
        left, right, score = process_video(temp_filename,exercise)
        

        os.remove(temp_filename)
 
   
        left_list = left.tolist() if left is not None else None
        right_list = right.tolist() if right is not None else None
        
        result = {
            'left_knee_angles': left_list,
            'right_knee_angles': right_list,
            'score': score,
            'message': 'Video processed successfully'
        }
        print(result['score'])
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return jsonify({'message': 'Error processing video', 'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)