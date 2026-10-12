import time
from typing import ClassVar

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult

import mediapipe as mp


class FaceRoiExtractor:
    FACE_REGION_TO_INDEX: ClassVar[dict[str, int]] = {
        "FOREHEAD": 0,
        "CHEEK_L": 1,
        "CHEEK_R": 2
    }
    FACE_REGIONS: ClassVar[dict[str, list[int]]] = {
        "FOREHEAD": [9, 10, 66, 67, 69, 103, 104, 105, 107, 108, 109, 151, 296, 297, 299, 332, 333, 334, 336, 337, 338],
        "CHEEK_L": [36, 50, 101, 111, 116, 117, 118, 119, 123, 135, 137, 138, 147, 177, 187,192, 205, 206, 207, 212, 213, 214, 215, 216, 227],
        "CHEEK_R": [266, 280, 330, 340, 345, 346, 347, 348, 352, 364, 366, 367, 376, 401, 411, 416, 425, 426, 427, 432, 433, 434, 435, 436, 447]
    }

    def __init__(self, path_to_model: str):
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path_to_model),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=False,
            num_faces=1,    
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        
    def detect_landmarks(self, frame_bgr: np.ndarray) -> FaceLandmarkerResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        now = int(time.monotonic() * 1000)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(image, now)
        return result

    def draw_landmarks(self, frame_bgr: np.ndarray, result: FaceLandmarkerResult, draw_indices:bool=False) -> np.ndarray:
        landmarked_frame = frame_bgr.copy()

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                if not landmarks:
                    continue
                nearest_z = min(point.z for point in landmarks)
                depth_range = max(point.z for point in landmarks) - nearest_z
                for i, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_bgr.shape[1])
                    y = int(landmark.y * landmarked_frame.shape[0])
                    depth = (landmark.z - nearest_z) / depth_range if depth_range > 0 else 0.0
                    intensity = round(50 + 205 * (1.0 - depth) ** 2.5)

                    if draw_indices:
                        cv2.putText(landmarked_frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    if any(i in indices for indices in self.FACE_REGIONS.values()):
                        cv2.circle(landmarked_frame, (x, y), 2, (intensity, 0, 0), -1)
                        continue
                    cv2.circle(landmarked_frame, (x, y), 2, (0, intensity, 0), -1)

        return landmarked_frame

    def extract_face_roi(self, frame_bgr: np.ndarray, result: FaceLandmarkerResult) -> np.ndarray | None:
        if not result.face_landmarks or not result.face_landmarks[0]:
            return None

        landmarks = result.face_landmarks[0]
        height, width = frame_bgr.shape[:2]
        face_rois = {}
        for region, indices in self.FACE_REGIONS.items():
            if any(index >= len(landmarks) for index in indices):
                return None
            xs = [int(landmarks[index].x * width) for index in indices]
            ys = [int(landmarks[index].y * height) for index in indices]
            x_min, x_max = max(0, min(xs)), min(width, max(xs) + 1)
            y_min, y_max = max(0, min(ys)), min(height, max(ys) + 1)
            if x_max <= x_min or y_max <= y_min:
                return None
            face_rois[region] = frame_bgr[y_min:y_max, x_min:x_max]

        forehead = cv2.resize(face_rois["FOREHEAD"], (256, 128))
        left_cheek = cv2.resize(face_rois["CHEEK_L"], (128, 128))
        right_cheek = cv2.resize(face_rois["CHEEK_R"], (128, 128))
        cheeks = np.concatenate((left_cheek, right_cheek), axis=1)
        return np.concatenate((forehead, cheeks), axis=0)
