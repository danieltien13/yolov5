import cv2
import supervision as sv
import torch

model = torch.hub.load("../yolov5", "custom", path="runs/train/exp2/weights/best.pt", source="local")

image_name = "000000000009"

results = model(f"../datasets/coco128/images/train2017/{image_name}.jpg")

detections = sv.Detections.from_yolov5(results)

# detections .txt file has this format:
# 45 0.769337 0.270366 0.461326 0.443452 0.829141
# 49 0.672552 0.221547 0.121519 0.140393 0.71401

print(f"detections: {detections}")

# Load image
image = cv2.imread(f"../datasets/coco128/images/train2017/{image_name}.jpg")

# Annotate image
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

print(type(detections))

annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)

labels = []
# detections is some sort of iterable/list
for i in range(len(detections)):
    class_id = detections.class_id[i]
    confidence = detections.confidence[i]
    label = f"{model.names[class_id]} {confidence:.2f}"
    labels.append(label)

print(f"labels: {labels}" )

annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# Display image
sv.plot_image(image=annotated_image, size=(16,16))
cv2.imwrite("error_mining.jpg", annotated_image)
