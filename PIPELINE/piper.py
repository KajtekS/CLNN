from pathlib import Path
from typing import Any
import numpy as np

from cv2 import VideoCapture

from MODELS.model import Model
from PIPELINE.pipeline import Pipeline
from TOOLS.resampling import video_resampler, gt_resampler
from TOOLS.face_roi_extractor import FaceRoiExtractor, FaceLandmarkerResult


class Piper(Pipeline):
    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path)
        self.extractor = FaceRoiExtractor("TOOLS/mediapipe/face_landmarker.task")

    def bronze(
        self, data: np.ndarray, gt: np.ndarray, fs_start, fs_gt_start, fs_end,
    ) -> tuple[np.ndarray, np.ndarray]:
        data_resampled = video_resampler(data, fs_start, fs_end)
        gt_resampled = gt_resampler(data, fs_gt_start, fs_end)

        return (data_resampled, gt_resampled)

    def silver(self, data) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        data = []
        for frame in data:
            result = self.extractor.detect_landmarks(frame)
            roi = self.extractor.extract_face_roi(frame, result)
            if roi is None:
                data.append(np.zeros((256, 256, 3)))
            else:
                data.append(roi)

        video = np.stack(data).astype(np.float32) / 255.0

        #Diffrences
        video = np.diff(video, axis=0, prepend=video[:1])
        
        return video

    def gold(self, data) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        return super().gold(data)
