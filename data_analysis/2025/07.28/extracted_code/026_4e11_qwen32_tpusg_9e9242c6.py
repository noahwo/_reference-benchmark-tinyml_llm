"""
Extracted from: raw_qwen2.5-coder:32b_4e11_tpusg_batch
Entry ID: 9e9242c6
Entry Name: 9e92_tpu_sketch_generator
Session ID: qwen2.5-coder:32b_4e11_tpusg_batch
Timestamp: 2025-07-29T12:12:40.909000+00:00
Tags: experiment, qwen2.5-coder:32b, tpu_sketch_generator
"""

import os

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Phase 1: Setup
# 1.1 Imports are already handled above
# 1.2 Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_9e92.mp4"
confidence_threshold  = 0.5

# 1.3 Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 1.4 Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])

interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]
floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess Data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    
    # Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    
    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Phase 4: Output Interpretation & Handling Loop
    # Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    # Interpret Results and Post-processing
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            label = labels[class_id]
            
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label_text = f'{label}: {scores[i]:.2f}'
            cv2.putText(frame, label_text, (int(left), int(top) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()