from pathlib import Path
from typing import Any
import numpy as np

from cv2 import VideoCapture

from MODELS.model import Model
from PIPELINE.pipeline import Pipeline
from TOOLS.resampling import video_resampler, gt_resampler


class Piper(Pipeline):
    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path)

    def bronze(
        self, data: np.ndarray, gt: np.ndarray, fs_start, fs_end
    ) -> tuple[np.ndarray, np.ndarray]:
        data_resampled = video_resampler(data, fs_start, fs_end)
        gt_resampled = gt_resampler(data, fs_start, fs_end)

        return (data_resampled, gt_resampled)

    def silver(self, data) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        return super().silver(data)

    def gold(self, data) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        return super().gold(data)
