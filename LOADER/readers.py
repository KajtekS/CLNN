
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np
import pandas as pd

from LOADER.vid_readers import get_reader


class Reader(ABC):
    @staticmethod
    @abstractmethod
    def read(
        video_path: Path,
        gt_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        pass


class SumsReader(Reader):
    @staticmethod
    def read(
        video_path: Path,
        gt_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:

        matrix_video = get_reader("AVI").read_video(video_path)

        # SUMS: columns "timestamp" and "bvp"
        bvp_data = pd.read_csv(gt_path)

        bvp_timestamps = bvp_data["timestamp"].to_numpy()
        signal = bvp_data["bvp"].to_numpy(dtype=np.float64)

        timestamp_path = gt_path.parent / "frames_timestamp.csv"
        frame_timestamps = pd.read_csv(timestamp_path)["timestamp"].to_numpy()

        # Find the nearest BVP sample for every video frame
        indices = np.searchsorted(bvp_timestamps, frame_timestamps)

        indices = np.clip(indices, 1, len(bvp_timestamps) - 1)

        left = indices - 1
        right = indices

        choose_right = (
            np.abs(bvp_timestamps[right] - frame_timestamps)
            < np.abs(bvp_timestamps[left] - frame_timestamps)
        )

        nearest_indices = np.where(choose_right, right, left)
        signal_gt = signal[nearest_indices]

        return matrix_video, signal_gt


class UbfcPhysReader(Reader):

    @staticmethod
    def read(
        video_path: Path,
        gt_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:

        matrix_video = get_reader("AVI").read_video(video_path)

        # UBFC-PHYS: one BVP value per CSV row
        signal_gt = []

        with open(gt_path, "r", newline="") as file:
            reader = csv.reader(file)

            for row in reader:
                if row:
                    signal_gt.append(float(row[0]))

        return matrix_video, np.asarray(signal_gt, dtype=np.float64)


class UbfcRppgReader(Reader):

    @staticmethod
    def read(
        video_path: Path,
        gt_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:

        matrix_video = get_reader("AVI").read_video(video_path)

        # UBFC-rPPG: BVP values are on the first line,
        # separated by whitespace.
        with open(gt_path, "r", encoding="utf-8") as file:
            first_line = file.readline()

        signal_gt = np.fromstring(first_line, sep=" ", dtype=np.float64)

        return matrix_video, signal_gt


class PureReader(Reader):

    @staticmethod
    def read(
        video_path: Path,
        gt_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:

        matrix_video = get_reader("PNG").read_video(video_path)

        # PURE: BVP is stored under Value.waveform
        with open(gt_path, "r", encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)

        package = data["/FullPackage"]

        signal_gt = np.asarray(
            [entry["Value"]["waveform"] for entry in package],
            dtype=np.float64
        )

        return matrix_video, signal_gt


def get_dataset_reader(dataset_name: str) -> type[Reader]:
    """Return the reader for the selected dataset."""

    readers = {
        "SUMS": SumsReader,
        "UBFC-PHYS": UbfcPhysReader,
        "UBFC-RPPG": UbfcRppgReader,
        "PURE": PureReader,
    }

    dataset_name = dataset_name.upper()

    if dataset_name not in readers:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Available: {list(readers.keys())}"
        )

    return readers[dataset_name]