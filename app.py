from flask import Flask, request, jsonify, Response, render_template
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from anomalib.deploy import TorchInferencer
from anomalib.post_processing import Visualizer

from pyueye import ueye

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
    
    model = YOLO('yolo.pt')
    hCam = ueye.HIDS(0)             #0: first available camera;  1-254: The camera with the specified camera ID
    sInfo = ueye.SENSORINFO()
    cInfo = ueye.CAMINFO()
    pcImageMemory = ueye.c_mem_p()
    MemID = ueye.int()
    rectAOI = ueye.IS_RECT()
    pitch = ueye.INT()
    nBitsPerPixel = ueye.INT(24)    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
    channels = 3                    #3: channels for color mode(RGB); take 1 channel for monochrome
    m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
    bytes_per_pixel = int(nBitsPerPixel / 8)

    nRet = ueye.is_InitCamera(hCam, None)
    if nRet != ueye.IS_SUCCESS:
        print("is_InitCamera ERROR")

    nRet = ueye.is_GetCameraInfo(hCam, cInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetCameraInfo ERROR")

    nRet = ueye.is_GetSensorInfo(hCam, sInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetSensorInfo ERROR")

    nRet = ueye.is_ResetToDefault( hCam)
    if nRet != ueye.IS_SUCCESS:
        print("is_ResetToDefault ERROR")

    nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)

    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
    # setup the color depth to the current windows setting
        ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_BAYER: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()
    elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
        # for color camera models use RGB32 mode
        m_nColorMode = ueye.IS_CM_BGRA8_PACKED
        nBitsPerPixel = ueye.INT(32)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_CBYCRY: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
        # for color camera models use RGB32 mode
        m_nColorMode = ueye.IS_CM_MONO8
        nBitsPerPixel = ueye.INT(8)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_MONOCHROME: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    
    else:
    # for monochrome camera models use Y8 mode
        m_nColorMode = ueye.IS_CM_MONO8
        nBitsPerPixel = ueye.INT(8)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("else")
    
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    if nRet != ueye.IS_SUCCESS:
        print("is_AOI ERROR")

    width = rectAOI.s32Width
    height = rectAOI.s32Height

    print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
    print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
    print("Maximum image width:\t", width)
    print("Maximum image height:\t", height)
    print()

    nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_AllocImageMem ERROR")
    else:
        # Makes the specified image memory the active memory
        nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetImageMem ERROR")
        else:
            # Set the desired color mode
            nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

    # Activates the camera's live video mode (free run mode)
    nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")

    # Enables the queue mode for existing image memory sequences
    nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
    if nRet != ueye.IS_SUCCESS:
        print("is_InquireImageMem ERROR")
    else:
        print("Press q to leave the programm")    
     # 0 represents the default webcam


    while(nRet == ueye.IS_SUCCESS):
        
        array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
        frame = np.reshape(array,(height.value, width.value, bytes_per_pixel))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
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
    hCam = ueye.HIDS(0)             #0: first available camera;  1-254: The camera with the specified camera ID
    sInfo = ueye.SENSORINFO()
    cInfo = ueye.CAMINFO()
    pcImageMemory = ueye.c_mem_p()
    MemID = ueye.int()
    rectAOI = ueye.IS_RECT()
    pitch = ueye.INT()
    nBitsPerPixel = ueye.INT(24)    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
    channels = 3                    #3: channels for color mode(RGB); take 1 channel for monochrome
    m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
    bytes_per_pixel = int(nBitsPerPixel / 8)

    nRet = ueye.is_InitCamera(hCam, None)
    if nRet != ueye.IS_SUCCESS:
        print("is_InitCamera ERROR")

    nRet = ueye.is_GetCameraInfo(hCam, cInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetCameraInfo ERROR")

    nRet = ueye.is_GetSensorInfo(hCam, sInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetSensorInfo ERROR")

    nRet = ueye.is_ResetToDefault( hCam)
    if nRet != ueye.IS_SUCCESS:
        print("is_ResetToDefault ERROR")

    nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)

    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
    # setup the color depth to the current windows setting
        ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_BAYER: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
        # for color camera models use RGB32 mode
        m_nColorMode = ueye.IS_CM_BGRA8_PACKED
        nBitsPerPixel = ueye.INT(32)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_CBYCRY: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
        # for color camera models use RGB32 mode
        m_nColorMode = ueye.IS_CM_MONO8
        nBitsPerPixel = ueye.INT(8)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_MONOCHROME: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    
    else:
    # for monochrome camera models use Y8 mode
        m_nColorMode = ueye.IS_CM_MONO8
        nBitsPerPixel = ueye.INT(8)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("else")
    
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    if nRet != ueye.IS_SUCCESS:
        print("is_AOI ERROR")

    width = rectAOI.s32Width
    height = rectAOI.s32Height

    print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
    print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
    print("Maximum image width:\t", width)
    print("Maximum image height:\t", height)
    print()

    nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_AllocImageMem ERROR")
    else:
        # Makes the specified image memory the active memory
        nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetImageMem ERROR")
        else:
            # Set the desired color mode
            nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

    # Activates the camera's live video mode (free run mode)
    nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")

    # Enables the queue mode for existing image memory sequences
    nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
    if nRet != ueye.IS_SUCCESS:
        print("is_InquireImageMem ERROR")
    else:
        print("Press q to leave the programm")





    inferencer = TorchInferencer(path='anom.pt', device='auto')
    visualizer = Visualizer(mode='simple', task='segmentation')



     # 0 represents the default webcam
    while(nRet == ueye.IS_SUCCESS):

        array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
        frame = np.reshape(array,(height.value, width.value, bytes_per_pixel))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
        predictions = inferencer.predict(frame)
        annotated_frame = visualizer.visualize_image(predictions)

        # Encode the annotated frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)

        # Send the frame as a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def webcam():
    hCam = ueye.HIDS(0)             #0: first available camera;  1-254: The camera with the specified camera ID
    sInfo = ueye.SENSORINFO()
    cInfo = ueye.CAMINFO()
    pcImageMemory = ueye.c_mem_p()
    MemID = ueye.int()
    rectAOI = ueye.IS_RECT()
    pitch = ueye.INT()
    nBitsPerPixel = ueye.INT(24)    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
    channels = 3                    #3: channels for color mode(RGB); take 1 channel for monochrome
    m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
    bytes_per_pixel = int(nBitsPerPixel / 8)

    nRet = ueye.is_InitCamera(hCam, None)
    if nRet != ueye.IS_SUCCESS:
        print("is_InitCamera ERROR")

    nRet = ueye.is_GetCameraInfo(hCam, cInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetCameraInfo ERROR")

    nRet = ueye.is_GetSensorInfo(hCam, sInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetSensorInfo ERROR")

    nRet = ueye.is_ResetToDefault( hCam)
    if nRet != ueye.IS_SUCCESS:
        print("is_ResetToDefault ERROR")

    nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)

    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
    # setup the color depth to the current windows setting
        ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_BAYER: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
        # for color camera models use RGB32 mode
        m_nColorMode = ueye.IS_CM_BGRA8_PACKED
        nBitsPerPixel = ueye.INT(32)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_CBYCRY: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
        # for color camera models use RGB32 mode
        m_nColorMode = ueye.IS_CM_MONO8
        nBitsPerPixel = ueye.INT(8)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_MONOCHROME: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    
    else:
    # for monochrome camera models use Y8 mode
        m_nColorMode = ueye.IS_CM_MONO8
        nBitsPerPixel = ueye.INT(8)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("else")
    
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    if nRet != ueye.IS_SUCCESS:
        print("is_AOI ERROR")

    width = rectAOI.s32Width
    height = rectAOI.s32Height

    print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
    print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
    print("Maximum image width:\t", width)
    print("Maximum image height:\t", height)
    print()

    nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_AllocImageMem ERROR")
    else:
        # Makes the specified image memory the active memory
        nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetImageMem ERROR")
        else:
            # Set the desired color mode
            nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

    # Activates the camera's live video mode (free run mode)
    nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")

    # Enables the queue mode for existing image memory sequences
    nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
    if nRet != ueye.IS_SUCCESS:
        print("is_InquireImageMem ERROR")
    else:
        print("Press q to leave the programm")


     # 0 represents the default webcam
    while(nRet == ueye.IS_SUCCESS):

        array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
        frame = np.reshape(array,(height.value, width.value, bytes_per_pixel))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
        

        # Encode the annotated frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)

        # Send the frame as a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    

@app.route('/stream_yolo')
def stream_yolo():
    return Response(webcam_yolo(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_anom')
def stream_anom():
    return Response(webcam_anom(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_webcam')
def stream_webcam():
    return Response(webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3333)
