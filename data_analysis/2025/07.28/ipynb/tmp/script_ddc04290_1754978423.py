"""
Extracted from: raw_qwen2.5-coder:32b_ae24_tpusg_batch
Entry ID: b9b901f7
Entry Name: b9b9_tpu_sketch_generator
Session ID: qwen2.5-coder:32b_ae24_tpusg_batch
Timestamp: 2025-07-29T11:37:38.969000+00:00
Tags: experiment, qwen2.5-coder:32b, tpu_sketch_generator
"""

import os
import time

import cv2  # Import OpenCV
import numpy as np  # Corrected import for numpy
from tflite_runtime.interpreter import Interpreter, load_delegate

# Phase 1: Setup
# Paths/Parameters
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path   = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_b9b9.mp4"
confidence_threshold   = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Failed to load EdgeTPU delegate: {e}")
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])

interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = (input_details[0]['dtype'] == np.float32)
height, width = input_details[0]['shape'][1:3]

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    boxes = output_data[0][0]
    classes = output_data[1][0]
    scores = output_data[2][0]
    num_detections = int(output_data[3][0])

    # Interpret Results and Apply Confidence Threshold
    for i in range(num_detections):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * cap.get(3), xmax * cap.get(3),
                                          ymin * cap.get(4), ymax * cap.get(4))
            
            # Draw bounding box
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detection to output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()