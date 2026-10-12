import cv2
import numpy as np
import os
from abc import ABC, abstractmethod
from MODELS.model import Model
from pathlib import Path
from LOADER.loader import get_loader
from LOADER.readers import get_dataset_reader
from CONFIGS.yaml_class import Config, ConfigLoader


class Pipeline(ABC):
    def __init__(self, config_path: Path) -> None:
        super().__init__()
        # self.config = ConfigParser.parse(config_path)

    def run(self, config_path: Path) -> np.ndarray:
        config = ConfigLoader.load(config_path)

        match config.MAIN.PROCESS_LAYER:
            case 'bronze':
                data = self.matrix_loader(config.BRONZE.PATH)

                for video_data, gt_data in zip(*data):
                    pass
                
            case 'silver':
                pass
            case 'gold':
                pass

        return np.zeros(0)

    # Functions to load data from diffrent stages of processing
    def raw_loader(self, dataset_name, data_path) -> tuple[np.ndarray, np.ndarray]:
        samples = get_loader(dataset_name).load_data(data_path)

        matrixes = []
        gts = []

        for sample in samples:
            video_path = sample["video_path"]
            gt_path = sample["gt_path"]

            m, gt = get_dataset_reader(dataset_name).read(video_path, gt_path)

            matrixes.append(m)
            gts.append(gt)

        return (np.array(matrixes), np.array(gts))

    def matrix_loader(self, data_path: Path) -> tuple[np.ndarray, np.ndarray]:
        root = Path(data_path)

        video = []
        gt = []

        for cat in root.iterdir():
            if not cat.is_dir():
                continue

            video.append(np.load(next(cat.glob("video.npy"))))
            gt.append(np.load(next(cat.glob("gt.npy"))))

        return (np.array(video), np.array(gt))

    def save_stage(
        self, data: list[tuple[np.ndarray, np.ndarray]], data_path: Path, hash, stage
    ) -> None:
        """Save processed data to a pipeline stage.

        Args:
            data: List of video matrices and corresponding ground-truth signals.
            data_path: Root directory where the data should be saved.
            hash: Hash identifying the configuration used to generate the data.
            stage: Pipeline stage name, e.g. "BRONZE".

        """
        dir_path = data_path / stage / hash

        for i, (matrix, gt) in enumerate(data):
            subject_dir = dir_path / f"subject_{i}"
            subject_dir.mkdir(parents=True, exist_ok=True)

            np.save(subject_dir / "video.npy", matrix)
            np.save(subject_dir / "gt.npy", gt)

    # Functions to process data at diffrent stages
    @abstractmethod
    def bronze(
        self, data: np.ndarray, gt: np.ndarray, fs_start: int, fs_end: int
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def silver(self, data) -> np.ndarray:
        pass

    @abstractmethod
    def gold(self, data) -> np.ndarray:
        pass
