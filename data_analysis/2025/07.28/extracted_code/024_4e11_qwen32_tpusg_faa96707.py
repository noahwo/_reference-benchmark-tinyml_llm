"""
Extracted from: raw_qwen2.5-coder:32b_4e11_tpusg_batch
Entry ID: faa96707
Entry Name: faa9_tpu_sketch_generator
Session ID: qwen2.5-coder:32b_4e11_tpusg_batch
Timestamp: 2025-07-29T12:15:45.214000+00:00
Tags: experiment, qwen2.5-coder:32b, tpu_sketch_generator
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Paths and parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_faa9.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Error loading EdgeTPU delegate: {e}. Falling back to default location.")
    try:
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        print(f"Failed to load EdgeTPU delegate: {e}")
        exit(1)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = (input_details[0]['dtype'] == np.float32)
height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

# Input acquisition & preprocessing loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit(1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    # Interpret results and apply confidence threshold
    for i in range(num_detections):
        if scores[0][i] > confidence_threshold:
            class_id = int(classes[0][i])
            box = boxes[0][i]
            ymin, xmin, ymax, xmax = box

            # Scale coordinates to frame dimensions
            ymin = int(max(1, ymin * frame.shape[0]))
            xmin = int(max(1, xmin * frame.shape[1]))
            ymax = int(min(frame.shape[0], ymax * frame.shape[0]))
            xmax = int(min(frame.shape[1], xmax * frame.shape[1]))

            # Draw bounding box and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f'{labels[class_id]}: {int(scores[0][i] * 100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Handle output
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()