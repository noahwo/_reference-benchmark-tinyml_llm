"""
Extracted from: raw_qwen2.5-coder:32b_4e11_tpusg_batch
Entry ID: c344c50d
Entry Name: c344_tpu_sketch_generator
Session ID: qwen2.5-coder:32b_4e11_tpusg_batch
Timestamp: 2025-07-29T12:27:56.630000+00:00
Tags: experiment, qwen2.5-coder:32b, tpu_sketch_generator
"""

import time

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Phase 1: Setup
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path   = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_c344.mp4"
confidence_threshold   = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path, 
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Failed to load EdgeTPU delegate: {e}")
    try:
        interpreter = Interpreter(model_path=model_path, 
                                  experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        print(f"Failed to load alternative EdgeTPU delegate: {e}")
        raise

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = (input_details[0]['dtype'] == np.float32)
_, height, width, _ = input_details[0]['shape']  # Adjusted to handle batch dimension

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = [interpreter.get_tensor(output_detail['index']) for output_detail in output_details]
    
    # Print the shapes of each output tensor to understand their structure
    for i, data in enumerate(output_data):
        print(f"Output {i} shape: {data.shape}")

    # Assuming detection_boxes, detection_classes, and detection_scores are the outputs we need
    detection_boxes = output_data[0]
    detection_classes = output_data[1]
    detection_scores = output_data[2]

    # Number of detections might be a fixed number or part of metadata; let's assume it's the size of the first dimension
    num_detections = detection_scores.shape[1]  # Adjust based on actual model output

    for i in range(num_detections):
        score = float(detection_scores[0][i])
        if score > confidence_threshold:
            class_id = int(detection_classes[0][i]) - 1  # Adjusting class ID to match label index
            box = detection_boxes[0][i]

            ymin, xmin, ymax, xmax = box
            im_height, im_width, _ = frame.shape

            (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width,
                                            ymin * im_height, ymax * im_height)

            cv2.rectangle(frame, (int(xminn), int(yminn)), (int(xmaxx), int(ymaxx)), (0, 255, 0), 2)
            label = f'{labels[class_id]}: {score:.2f}'
            cv2.putText(frame, label, (int(xminn), int(yminn) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()