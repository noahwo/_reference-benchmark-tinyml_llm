"""
Extracted from: raw_codestral_c8f6_tpusg_batch
Entry ID: 6536d7f7
Entry Name: 6536_tpu_sketch_generator
Session ID: codestral_c8f6_tpusg_batch
Timestamp: 2025-07-29T14:14:27.086000+00:00
Tags: benchmark, codestral:latest, tpu_sketch_generator
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import load_delegate, Interpreter

# Define paths and parameters
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
input_video   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_video   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_6536.mp4"
label_path   = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
confidence_threshold   = 0.5

# Load labels from file
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter with EdgeTPU delegate
interpreter = Interpreter(model_path=model_path)  # Updated to use a generic delegate loading statement
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]
floating_model = (input_details[0]['dtype'] == np.float32)

# Open input video and initialize output video writer
cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input image by resizing and normalizing
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Perform inference on the image
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results and post-process them
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])  # Updated to cast the result to int
    for i in range(num_detections):
        if scores[i] > confidence_threshold:
            class_id = classes[i]
            label = labels[class_id]
            ymin, xmin, ymax, xmax = boxes[i]
            left = int(xmin * width)
            top = int(ymin * height)
            right = int(xmax * width)
            bottom = int(ymax * height)

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Write the frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()