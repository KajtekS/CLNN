import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO


class Preproc:
    def __init__(self) -> None:
        self.model = YOLO("TOOLS/yolov8/yolov8n-face.pt")

    def process(self, vid, recenter, square_size):
        count = recenter
        cor = None
        frames = []

        # Extracting faces
        while True:
            success, img = vid.read()

            if not success:
                break

            if count == 0:
                cor = self.detect_face(img)
                count = recenter

            if cor is not None:
                face = img[cor[1] : cor[3], cor[0] : cor[2]]
                face = cv2.resize(face, square_size)
                frames.append(face)

            count -= 1

        # Diffrences counting and normalization
        frames = np.asarray(frames, dtype=np.float32)
        diff = (frames[1:] - frames[:-1]) / (frames[1:] + frames[:-1] + 1e-10)

        # Standarization
        diff = diff / np.std(diff)

        return diff

    def detect_face(self, img: NDArray[np.uint8]) -> tuple[int, int, int, int] | None:
        results = self.model(img)
        faces = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                faces.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": confidence,
                    }
                )
        if faces is None:
            return None

        face = max(
            faces,
            key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
        )
        x1, y1, x2, y2 = face["bbox"]

        return (x1, y1, x2, y2)
