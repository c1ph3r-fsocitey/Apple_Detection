import cv2
import argparse
import numpy as np
import time

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True,
                help='Path to input video file')
ap.add_argument('-c', '--config', required=True,
                help='/home/c1ph3r/TensorFlow_Projects/Apple-Detection/yolov3.cfg')
ap.add_argument('-w', '--weights', required=True,
                help='/home/c1ph3r/TensorFlow_Projects/Apple-Detection/yolov3.weights')
ap.add_argument('-cl', '--classes', required=True,
                help='/home/c1ph3r/TensorFlow_Projects/Apple-Detection/yolov3.txt')
args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getUnconnectedOutLayersNames()
    return layer_names

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label + " " + str(round(confidence, 2)), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Load classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the neural network
net = cv2.dnn.readNet(args.weights, args.config)

# Load the video
video = cv2.VideoCapture(args.video)

# Initialize variables for FPS calculation
prev_time = time.time()
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.4
    nms_threshold = 0.3

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices:
            box = boxes[i]  # Access the box using the index directly
            x, y, w, h = box[0], box[1], box[2], box[3]
            draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    else:
        print("No objects detected")

    # Calculate FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = 1 / elapsed_time
    prev_time = current_time
    frame_count += 1

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("object detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop ends, calculate the average FPS
avg_fps = frame_count / (time.time() - start_time)

# Print the average FPS
print(f'Average FPS: {avg_fps:.2f}')

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
