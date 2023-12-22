import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
x
image = cv2.imread("https://github.com/kishorravi/yolotest/blob/master/cow.jpg")  # Replace with the path to your image

# Get image dimensions
height, width, _ = image.shape

# Create a new image with RGB color order for displaying with Matplotlib
new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(new_image)

# Load YOLOv5 ONNX model
yolo = cv2.dnn.readNet("yolov5s.onnx")  # Replace with the path to your yolov5s.onnx file

# Load classes
classes = []
with open("coco.names", 'r') as f:
    classes = f.read().splitlines()

# Preprocess the image
blob = cv2.dnn.blobFromImage(image, 1 / 255, (640, 640), (0, 0, 0), swapRB=True, crop=False)
yolo.setInput(blob)

# Get YOLOv5 output
output_layer_name = yolo.getUnconnectedOutLayersNames()
layer_output = yolo.forward(output_layer_name)

# Initialize lists to store bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

# Process each detection
for output in layer_output:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.7:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
indexes_array = np.array(indexes).flatten()

# Draw bounding boxes on the image
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes_array:
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i], 2))
    color = colors[i]
    cv2.rectangle(new_image, (x, y), (x + w, y + h), color, 7)
    cv2.putText(new_image, label + " " + confi, (x, y + 20), font, 2, (255, 255, 255), 5)

# Display the image with bounding boxes
plt.imshow(new_image)
plt.show()
