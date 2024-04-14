import cv2
from ultralytics import YOLO
import supervision as sv
import urllib.request
import numpy as np


def main():
    url = 'http://192.168.137.123/cam-hi.jpg'
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Failed to open the IP camera stream")
        exit()

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)

        frame = cv2.imdecode(imgnp, -1)
        result = model(frame)[-1]
        detections = sv.Detections.from_ultralytics(result)

        labels = [
            f"{model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow("SafeVision", frame)

        if cv2.waitKey(30) == 27:
            break


if __name__ == "__main__":
    main()
