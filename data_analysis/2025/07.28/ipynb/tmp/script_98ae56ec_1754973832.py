"""
Extracted from: raw_qwen2.5-coder:32b_ae24_tpusg_batch
Entry ID: aebfebdb
Entry Name: aebf_tpu_sketch_generator
Session ID: qwen2.5-coder:32b_ae24_tpusg_batch
Timestamp: 2025-07-29T11:11:10.431000+00:00
Tags: experiment, qwen2.5-coder:32b, tpu_sketch_generator
"""

import os
import time

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Paths and parameters
model_path    = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path    = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path    = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_aebf.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Failed to load delegate: {e}. Falling back to alternative path.")
    try:
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        raise RuntimeError("Could not initialize Coral TPU delegate.") from e

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = (input_details[0]['dtype'] == np.float32)
batch_size, height, width, channels = input_details[0]['shape']

# Phase 2: Input Acquisition & Preprocessing Loop
import cv2

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    resized_frame = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(resized_frame, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    for i in range(num_detections):
        # Ensure scores[i] is a scalar value
        if scores[0][i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[0][i]
            class_id = int(classes[0][i])
            label = labels[class_id]
            
            # Scale coordinates
            im_height, im_width, _ = frame.shape
            (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
            
            cv2.rectangle(frame, (int(xminn), int(yminn)), (int(xmaxx), int(ymaxx)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[0][i]:.2f}', (int(xminn), int(yminn) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detection to output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()