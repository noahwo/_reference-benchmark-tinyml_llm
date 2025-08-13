"""
Extracted from: raw_codestral_c8f6_tpusg_batch
Entry ID: 64baf4ca
Entry Name: 64ba_tpu_sketch_generator
Session ID: codestral_c8f6_tpusg_batch
Timestamp: 2025-07-29T15:07:03.410000+00:00
Tags: benchmark, codestral:latest, tpu_sketch_generator
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Define necessary variables using the model path and other parameters provided in the main prompt
model_path    = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path    = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path    = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_64ba.mp4"
threshold    = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter with EdgeTPU
interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Acquire Input Data and Preprocessing Loop
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = np.expand_dims(cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2])), axis=0)
    floating_model = (input_details[0]['dtype'] == np.float32)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output Interpretation & Handling
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > threshold) and (scores[i] <= 1.0)):
            ymin = int((boxes[i][0]) * frame_height)
            xmin = int((boxes[i][1]) * frame_width)
            ymax = int((boxes[i][2]) * frame_height)
            xmax = int((boxes[i][3]) * frame_width)
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            label = labels[int(classes[i])] + ': ' + str(round((scores[i]*100), 2)) + '%'
            cv2.putText(frame, label, (xmin,ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()