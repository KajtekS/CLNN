import cv2
from ultralytics import YOLO

class Preproc:
    def __init__(self) -> None:
        self.model = YOLO("TOOLS/yolov8/yolov8n-face.pt")


    def proces(self, vid):
        self.face_detector(vid)

    def face_detector(self, img):
        img = cv2.imread("DATA/IMG_2998.png")

        results = self.model(img)
        faces = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                })
        x1, y1, x2, y2 = faces[0]["bbox"]

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )
        cv2.imshow("Face", img)
        cv2.waitKey(0)
        print(faces)
