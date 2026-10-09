from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class Reader(ABC):
    @staticmethod
    @abstractmethod
    def read_video(path: Path) -> np.ndarray:
        pass


class AviReader(Reader):
    @staticmethod
    def read_video(path: Path) -> np.ndarray:
        frames = []

        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frames.append(frame)
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames found in video: {path}")

        return np.stack(frames)


class PngReader(Reader):
    @staticmethod
    def read_video(path: Path) -> np.ndarray:
        if not path.is_dir():
            raise ValueError(f"Expected a directory: {path}")

        png_files = sorted(path.glob("*.png"))

        if not png_files:
            raise ValueError(f"No PNG files found in directory: {path}")

        frames = []

        for png_path in png_files:
            frame = cv2.imread(str(png_path), cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError(f"Could not read image: {png_path}")

            frames.append(frame)

        shapes = {frame.shape for frame in frames}

        if len(shapes) != 1:
            raise ValueError("PNG frames have different dimensions")

        return np.stack(frames)

def get_reader(video_type: str) -> type[Reader]:
    """Factory method to run reading videos"""
    readers = {
        "AVI": AviReader,
        "PNG": PngReader
    }
    video_type = video_type.upper()

    if video_type not in readers:
        raise ValueError(
            f"Unsupported reader filetype: {video_type}. Available: {list(readers.keys())}"
        )
    
    return readers[video_type]
