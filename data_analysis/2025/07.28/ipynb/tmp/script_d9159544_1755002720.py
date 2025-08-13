"""
Extracted from: raw_codestral_c8f6_tpusg_batch
Entry ID: c202cbd1
Entry Name: c202_tpu_sketch_generator
Session ID: codestral_c8f6_tpusg_batch
Timestamp: 2025-07-29T14:33:57.865000+00:00
Tags: benchmark, codestral:latest, tpu_sketch_generator
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Parameters
model_path    = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
input_path    = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
label_path    = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_c202.mp4"
confidence_threshold    = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load EdgeTPU delegate
delegate = load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')
interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

# Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(resized_frame, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            ymin = int((boxes[i][0] * frame.shape[0]))
            xmin = int((boxes[i][1] * frame.shape[1]))
            ymax = int((boxes[i][2] * frame.shape[0]))
            xmax = int((boxes[i][3] * frame.shape[1]))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 0), 2)
            label = labels[int(classes[i])] + ': ' + str(round(scores[i]*100, 2)) + '%'
            cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()