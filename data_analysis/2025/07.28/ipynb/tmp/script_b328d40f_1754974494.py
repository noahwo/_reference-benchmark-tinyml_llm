"""
Extracted from: raw_codestral_c8f6_tpusg_batch
Entry ID: 6820a937
Entry Name: 6820_tpu_sketch_generator
Session ID: codestral_c8f6_tpusg_batch
Timestamp: 2025-07-29T14:13:06.536000+00:00
Tags: benchmark, codestral:latest, tpu_sketch_generator
"""

import numpy as np

try:
    from tensorflow.lite import Interpreter
    from tensorflow.lite.experimental import load_delegate
except ImportError:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
import cv2

# Define paths/parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_6820.mp4"
threshold = 0.5  # Confidence threshold for detection

# Load Interpreter with EdgeTPU
interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Preprocess Data
    resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(resized, axis=0)
    # Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    # Post-processing (Confidence Thresholding, Coordinate Scaling, Bounding Box Clipping)
    for i in range(len(scores)):
        if ((scores[i] > threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * frame.shape[0])))
            xmin = int(max(1,(boxes[i][1] * frame.shape[1])))
            ymax = int(min(frame.shape[0],(boxes[i][2] * frame.shape[0])))
            xmax = int(min(frame.shape[1],(boxes[i][3] * frame.shape[1])))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
    # Handle Output (In this case, we just display the frame with bounding boxes)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()