import cv2
import numpy as np

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
dimension = 640
confThreshold = 0.5
nmsThreshold = 0.2

classFiles = 'classes.txt'
with open(classFiles, 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNet('Model/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def draw_bboxes(detections, img):
    class_ids = []
    confidences = []
    bboxes = []
    rows = detections.shape[0]

    img_height, img_width, _ = img.shape
    x_scale, y_scale = img_width / dimension, img_height / dimension

    for i in range(rows):
        row = detections[i]
        confidence = row[4]

        if confidence > 0.5:
            classes_score = row[5:]  # Accuracy of classes
            index = np.argmax(classes_score)

            if classes_score[index] > 0.5:
                class_ids.append(index)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x, y = int((cx - w / 2) * x_scale), int((cy - h / 2) * y_scale)
                width, height = int(w * x_scale), int(h * y_scale)
                bbox = np.array([x, y, width, height])
                bboxes.append(bbox)

    indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.2)

    for i in indices:
        x, y, w, h = bboxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))

        if label == 'mask':
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 1, (0, 255, 0), 2)
        elif label == 'no-mask':
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, label + " " + confidence, (x, y - 20), font, 1, (0, 0, 255), 2)


def detect(img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (dimension, dimension), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]
    return detections


def detect_mask_video(video):
    cap = cv2.VideoCapture(video)

    while True:
        success, img = cap.read()
        if img is None:
            break

        detections = detect(img)
        draw_bboxes(detections, img)
        cv2.imshow('Video', img)
        cv2.waitKey(1)

    cap.release()


def detect_mask_image(image):
    img = cv2.imread(image)
    detections = detect(img)
    draw_bboxes(detections, img)
    cv2.imshow('Video', img)
    cv2.waitKey(0)


#detect_mask_image('Testing/no-mask.jpg')
detect_mask_video('Testing/mask.mp4')
