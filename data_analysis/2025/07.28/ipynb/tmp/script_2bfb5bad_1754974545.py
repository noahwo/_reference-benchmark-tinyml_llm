"""
Extracted from: raw_codestral_c8f6_tpusg_batch
Entry ID: d7f970f5
Entry Name: d7f9_tpu_sketch_generator
Session ID: codestral_c8f6_tpusg_batch
Timestamp: 2025-07-29T14:08:27.386000+00:00
Tags: benchmark, codestral:latest, tpu_sketch_generator
"""

import cv2
import numpy as np
try:
    from tensorflow.lite import Interpreter
    from tensorflow.lite.experimental import load_delegate
except ImportError:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate

# Function to preprocess input image
def preprocess_input(image, input_size=(300, 300)):
    image = cv2.resize(image, input_size)
    image = np.expand_dims(image, axis=0)
    if floating_model:
        image = (np.float32(image) - 127.5) / 127.5
    return image

# Function to load labels from text file
def load_labels(path):
    with open(path, 'r') as f:
        labels = [line.strip() for line in f]
    return labels

# Paths to model and label files, and input video
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path   = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_video_path   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_video_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_d7f9.mp4"

# Load the model with EdgeTPU delegate
interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = (input_details[0]['dtype'] == np.float32)

# Load labels and open input video
labels = load_labels(label_path)
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    input_data = preprocess_input(frame)

    # Set input tensor for the interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference with the interpreter and retrieve output data (detection boxes)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process output data here to draw bounding boxes on frame, then write frame to output video
    out.write(frame)

cap.release()
out.release()