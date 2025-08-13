"""
Extracted from: raw_codestral_c8f6_tpusg_batch
Entry ID: 09cb2e13
Entry Name: 09cb_tpu_sketch_generator
Session ID: codestral_c8f6_tpusg_batch
Timestamp: 2025-07-29T14:23:06.723000+00:00
Tags: benchmark, codestral:latest, tpu_sketch_generator
"""

import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# Define paths/parameters
model_path     = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
input_path     = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
label_path     = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
output_path    = "/home/mendel/tinyml_autopilot/results/sheeps_detections_09cb.mp4"

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter with EdgeTPU
interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get model details and resize input shape for EdgeTPU
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]

# Input acquisition & preprocessing loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data and resize for EdgeTPU
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    floating_model = (input_details[0]['dtype'] == np.float32)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output interpretation & handling loop
    output_data = [interpreter.get_tensor(output['index']) for output in output_details]

    # Interpret results and post-processing
    boxes, scores, labels_idx = output_data[0], output_data[1], output_data[2]
    for i in range(len(boxes)):
        if scores[i][0] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i][0]
            label = labels[int(labels_idx[i][0])]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1], ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(left), int(top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Handle results and write frame to output video
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()