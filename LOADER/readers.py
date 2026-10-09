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
    def read(video_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
        pass


class SumsReader(Reader):
    @staticmethod
    def read(video_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:

        matrix_video = get_reader("AVI").read_video(video_path)

        bvp_data = pd.read_csv(gt_path)
        signal_gt = bvp_data["bvp"].to_numpy(dtype=np.float64)

        return matrix_video, signal_gt


class UbfcPhysReader(Reader):
    @staticmethod
    def read(video_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:

        matrix_video = get_reader("AVI").read_video(video_path)
        signal_gt = []

        with open(gt_path, "r", newline="") as file:
            reader = csv.reader(file)

            for row in reader:
                if row:
                    signal_gt.append(float(row[0]))

        return matrix_video, np.asarray(signal_gt, dtype=np.float64)


class UbfcRppgReader(Reader):
    @staticmethod
    def read(video_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:

        matrix_video = get_reader("AVI").read_video(video_path)

        with open(gt_path, "r", encoding="utf-8") as file:
            first_line = file.readline()

        signal_gt = np.fromstring(first_line, sep=" ", dtype=np.float64)

        return matrix_video, signal_gt


class PureReader(Reader):
    @staticmethod
    def read(video_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:

        matrix_video = get_reader("PNG").read_video(video_path)

        with open(gt_path, "r", encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)

        package = data["/FullPackage"]

        signal_gt = np.asarray(
            [entry["Value"]["waveform"] for entry in package], dtype=np.float64
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
            f"Unsupported dataset: {dataset_name}. Available: {list(readers.keys())}"
        )

    return readers[dataset_name]
