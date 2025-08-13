"""
Extracted from: raw_codestral_c8f6_tpusg_batch
Entry ID: 3f29855c
Entry Name: 3f29_tpu_sketch_generator
Session ID: codestral_c8f6_tpusg_batch
Timestamp: 2025-07-29T14:03:30.337000+00:00
Tags: benchmark, codestral:latest, tpu_sketch_generator
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Load labels and paths
labels = load_labels('/home/mendel/tinyml_autopilot/models/labelmap.txt')
input_video_path   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_video_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_3f29.mp4"
confidence_threshold   = 0.5
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"

# Load interpreter with EdgeTPU
interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

# Acquire input data
cap = cv2.VideoCapture(input_video_path)
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

# Loop over input data
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    input_data = np.expand_dims(cv2.resize(frame, (width, height)), axis=0)
    input_data = input_data if floating_model else input_data.astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Postprocess output
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            ymin = int((boxes[i][0]*frame.shape[0]))
            xmin = int((boxes[i][1]*frame.shape[1]))
            ymax = int((boxes[i][2]*frame.shape[0]))
            xmax = int((boxes[i][3]*frame.shape[1]))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    out.write(frame)
cap.release()
out.release()
cv2.destroyAllWindows()