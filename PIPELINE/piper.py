from pathlib import Path
from typing import Any
import numpy as np

from cv2 import VideoCapture

from MODELS.model import Model
from PIPELINE.pipeline import Pipeline
from TOOLS.resampling import video_resampler, gt_resampler
from TOOLS.face_roi_extractor import FaceRoiExtractor, FaceLandmarkerResult


class Piper(Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = FaceRoiExtractor("TOOLS/mediapipe/face_landmarker.task")

    def bronze(
        self,
        data: np.ndarray,
        gt: np.ndarray,
        fs_start,
        fs_gt_start,
        fs_end,
    ) -> tuple[np.ndarray, np.ndarray]:
        data_resampled = video_resampler(data, fs_start, fs_end)
        gt_resampled = gt_resampler(gt, fs_gt_start, fs_end)

        return (data_resampled, gt_resampled)

    def silver(
        self, data, extractor: FaceRoiExtractor | None = None
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        if extractor is None:
            extractor = self.extractor

        processed_frames = []
        for frame in data:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            result = extractor.detect_landmarks(frame)
            roi = extractor.extract_face_roi(frame, result)
            if roi is None:
                processed_frames.append(np.zeros((256, 256, 3)))
            else:
                processed_frames.append(roi)

        video = np.stack(processed_frames).astype(np.float32) / 255.0

        # Diffrences
        video = np.diff(video, axis=0, prepend=video[:1])

        return video

    def gold(
        self, data: np.ndarray, gt, chunk_size
    ) -> tuple[np.ndarray[Any], np.ndarray[Any]]:
        # Input shape [N, H, W, C]
        total_frames = (len(data) // chunk_size) * chunk_size
        data = data[:total_frames]
        gt = gt[:total_frames]

        # Spliting into chunks, we get [Batch, N, H, W, C]
        split_points = np.arange(chunk_size, len(data), chunk_size)
        chunked_data = np.split(data, split_points)
        chunked_gt = np.split(gt, split_points)

        return (np.stack(chunked_data), np.stack(chunked_gt))
