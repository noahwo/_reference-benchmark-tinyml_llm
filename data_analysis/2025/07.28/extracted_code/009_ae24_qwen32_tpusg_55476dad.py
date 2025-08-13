"""
Extracted from: raw_qwen2.5-coder:32b_ae24_tpusg_batch
Entry ID: 55476dad
Entry Name: 5547_tpu_sketch_generator
Session ID: qwen2.5-coder:32b_ae24_tpusg_batch
Timestamp: 2025-07-29T11:24:04.705000+00:00
Tags: experiment, qwen2.5-coder:32b, tpu_sketch_generator
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Phase 1: Setup
## 1.1 Imports
## 1.2 Paths/Parameters
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path   = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_5547.mp4"
confidence_threshold  = 0.5

## 1.3 Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

## 1.4 Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Failed to load delegate from libedgetpu.so.1.0: {e}")
    try:
        interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        print(f"Failed to load delegate from /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0: {e}")
        interpreter = Interpreter(model_path=model_path)

interpreter.allocate_tensors()

## 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assuming input is a single image of shape (height, width, channels)
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
## 2.1 Acquire Input Data
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

# Get video properties to write output video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

## 2.2 Preprocess Data
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize image
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)

    ## 2.3 Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    ## 4.1 Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    ## 4.2 Interpret Results
    for i in range(num_detections):
        if scores[0][i] > confidence_threshold:  # Correctly access the score
            # Get bounding box coordinates and scale them to the original frame size
            ymin, xmin, ymax, xmax = boxes[0][i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)

            # Draw bounding box and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f"{labels[int(classes[0][i])]}: {scores[0][i]:.2f}"
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()