from flask import Flask, request, jsonify, Response, render_template
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from anomalib.deploy import TorchInferencer
from anomalib.post_processing import Visualizer

app = Flask(__name__)

# Initialize the YOLO model
model = None

# Function to process webcam frames
def process_frame(frame):
    global model
    if model is not None:
        results = model(frame)
        annotated_frame = results[0].render()
        return annotated_frame
    else:
        return frame



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_yolo', methods=['POST'])
def upload_yolo():
    
    model_file = request.files.get('model_file')

    if model_file and model_file.filename.endswith('.pt'):
        # Save the uploaded model file
        current_directory = os.getcwd()

        # Save the uploaded model file with the absolute path
        model_filename = model_file.filename
        model_path = os.path.join(current_directory, 'yolo.pt')
        model_file.save(model_path)

        print(f'Model file received and saved at: {model_path}')

        # Load the model
        

        return 'Model uploaded successfully'

    return 'Invalid model file'

@app.route('/upload_anom', methods=['POST'])
def upload_anom():
    
    model_file = request.files.get('model_file')

    if model_file and model_file.filename.endswith('.pt'):
        # Save the uploaded model file
        current_directory = os.getcwd()

        # Save the uploaded model file with the absolute path
        model_filename = model_file.filename
        model_path = os.path.join(current_directory, 'anom.pt')
        model_file.save(model_path)

        print(f'Model file received and saved at: {model_path}')

        # Load the model
        

        return 'Model uploaded successfully'

    return 'Invalid model file'

def webcam_yolo():
    camera = cv2.VideoCapture(0) 
    model = YOLO('yolo.pt')
     # 0 represents the default webcam
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        results = model(frame)    
        # Process the frame with YOLO
        annotated_frame = results[0].plot()

        # Encode the annotated frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)

        # Send the frame as a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    camera.release()

def webcam_anom():
    camera = cv2.VideoCapture(0) 
    inferencer = TorchInferencer(path='anom.pt', device='auto')
    visualizer = Visualizer(mode='simple', task='segmentation')
     # 0 represents the default webcam
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        predictions = inferencer.predict(frame)
        annotated_frame = visualizer.visualize_image(predictions)

        # Encode the annotated frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)

        # Send the frame as a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    camera.release()

@app.route('/stream_yolo')
def stream_yolo():
    return Response(webcam_yolo(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_anom')
def stream_anom():
    return Response(webcam_anom(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3333)
