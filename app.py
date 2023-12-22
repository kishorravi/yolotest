import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("/content/car.jpeg")
height, width, _ = image.shape  # Added to get height and width of the image

new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(new_image)

yolo = cv2.dnn.readNet("/content/yolov3-tiny.weights", "/content/yolov3-tiny.cfg")

classes = []
with open("/content/coco.names", 'r') as f:
    classes = f.read().splitlines()

blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
yolo.setInput(blob)
output_layer_name = yolo.getUnconnectedOutLayersNames()
layer_output = yolo.forward(output_layer_name)

boxes = []
confidences = []
class_ids = []

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

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
indexes_array = np.array(indexes).flatten()

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes_array:
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i], 2))
    color = colors[i]
    cv2.rectangle(new_image, (x, y), (x + w, y + h), color, 7)
    cv2.putText(new_image, label + " " + confi, (x, y + 20), font, 2, (255, 255, 255), 5)

plt.imshow(new_image)
plt.show()
boxes
