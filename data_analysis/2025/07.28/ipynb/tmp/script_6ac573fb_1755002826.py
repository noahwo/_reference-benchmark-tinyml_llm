"""
Extracted from: raw_codestral_c8f6_tpusg_batch
Entry ID: ce243473
Entry Name: ce24_tpu_sketch_generator
Session ID: codestral_c8f6_tpusg_batch
Timestamp: 2025-07-29T14:19:17.139000+00:00
Tags: benchmark, codestral:latest, tpu_sketch_generator
"""

import numpy as np
import cv2
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter

# Define paths and parameters
model_path      = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path      = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path      = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path     = "/home/mendel/tinyml_autopilot/results/sheeps_detections_ce24.mp4"
confidence_threshold      = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load EdgeTPU delegate
delegate = load_delegate('libedgetpu.so.1')

# Load interpreter with the model and delegate
interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

# Open video capture
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image
    resized_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    input_data = np.expand_dims(resized_img, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Interpret results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num = interpreter.get_tensor(output_details[3]['index'])[0]

    # Post-processing
    for i in range(int(num)):
        if scores[i] > confidence_threshold:
            box = boxes[i] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            (startY, startX, endY, endX) = box.astype("int")
            label = "{}: {:.2f}".format(labels[int(classes[i])], scores[i])
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()