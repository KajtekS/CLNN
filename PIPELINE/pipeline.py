import numpy as np
from abc import ABC, abstractmethod
from MODELS.model import Model
from pathlib import Path
from LOADER.loader import get_loader
from LOADER.readers import get_dataset_reader
from CONFIGS.yaml_class import Config, ConfigLoader

from typing import Any


from TOOLS.hash import hash_yaml


def _is_truthy(val: Any) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in ("yes", "true", "1")
    return bool(val)


class Pipeline(ABC):
    def __init__(self, config_path: Path) -> None:
        super().__init__()
        # self.config = ConfigParser.parse(config_path)

    def run(self, config_path: Path) -> np.ndarray:
        config = ConfigLoader.load(config_path)

        path_obj = Path(config_path)
        if path_obj.is_file():
            config_hash = hash_yaml(path_obj)
        else:
            config_hash = "0x00"

        option = config.MAIN.PROCESS_LAYER.lower()

        match option:
            case "raw":
                videos, gts = self.raw_loader(
                    config.RAW.DATASET_NAME, Path(config.RAW.PATH)
                )
                current_data = list(zip(videos, gts))
            case "bronze":
                videos, gts = self.matrix_loader(Path(config.BRONZE.PATH))
                current_data = list(zip(videos, gts))
            case "silver":
                videos, gts = self.matrix_loader(Path(config.SILVER.PATH))
                current_data = list(zip(videos, gts))
            case "gold":
                videos, gts = self.matrix_loader(Path(config.GOLD.PATH))
                current_data = list(zip(videos, gts))
            case _:
                raise ValueError(f"Invalid PROCESS_LAYER: {option}")

        # BRONZE STAGE
        should_downsample = _is_truthy(config.BRONZE.DOWN_SAMPLE.DO)

        if option == "raw":
            bronze_data = []
            for video_data, gt_data in current_data:
                if should_downsample:
                    video_res, gt_res = self.bronze(
                        video_data,
                        gt_data,
                        config.BRONZE.FS,
                        config.BRONZE.GT_FS,
                        config.BRONZE.DOWN_SAMPLE.FS,
                    )
                    bronze_data.append((video_res, gt_res))
                else:
                    bronze_data.append((video_data, gt_data))

            self.save_stage(bronze_data, Path(config.BRONZE.PATH), config_hash, "BRONZE")
            current_data = bronze_data

        elif option == "bronze" and should_downsample:
            bronze_data = []
            for video_data, gt_data in current_data:
                video_res, gt_res = self.bronze(
                    video_data,
                    gt_data,
                    config.BRONZE.FS,
                    config.BRONZE.GT_FS,
                    config.BRONZE.DOWN_SAMPLE.FS,
                )
                bronze_data.append((video_res, gt_res))

            self.save_stage(bronze_data, Path(config.BRONZE.PATH), config_hash, "BRONZE")
            current_data = bronze_data

        # SILVER STAGE
        if option in ("raw", "bronze", "silver"):
            silver_data = []
            for video_data, gt_data in current_data:
                video_silver = self.silver(video_data)
                silver_data.append((video_silver, gt_data))

            self.save_stage(silver_data, Path(config.SILVER.PATH), config_hash, "SILVER")
            current_data = silver_data

        # GOLD STAGE
        gold_data = []
        for video_data, gt_data in current_data:
            video_gold, gt_gold = self.gold(
                video_data, gt_data, config.GOLD.CHUNK_SIZE
            )
            gold_data.append((video_gold, gt_gold))

        self.save_stage(gold_data, Path(config.GOLD.PATH), config_hash, "GOLD")
        current_data = gold_data

        if current_data:
            try:
                return np.array([item[0] for item in current_data])
            except ValueError:
                return np.array([item[0] for item in current_data], dtype=object)

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

        subject_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "video.npy").exists()]
        if not subject_dirs:
            subject_dirs = [d for d in root.rglob("subject_*") if d.is_dir() and (d / "video.npy").exists()]

        for cat in sorted(subject_dirs, key=lambda p: p.name):
            video.append(np.load(cat / "video.npy"))
            gt.append(np.load(cat / "gt.npy"))

        return (np.array(video), np.array(gt))

    def save_stage(
        self, data: list[tuple[np.ndarray, np.ndarray]], data_path: Path, hash, stage: str
    ) -> None:
        """Save processed data to a pipeline stage.

        Args:
            data: List of video matrices and corresponding ground-truth signals.
            data_path: Root directory where the data should be saved.
            hash: Hash identifying the configuration used to generate the data.
            stage: Pipeline stage name, e.g. "BRONZE".

        """
        stage_name = stage.upper()
        if data_path.name == str(hash):
            dir_path = data_path
        elif data_path.name.upper() == stage_name:
            dir_path = data_path / str(hash)
        else:
            dir_path = data_path / stage_name / str(hash)

        for i, (matrix, gt) in enumerate(data):
            subject_dir = dir_path / f"subject_{i}"
            subject_dir.mkdir(parents=True, exist_ok=True)

            np.save(subject_dir / "video.npy", matrix)
            np.save(subject_dir / "gt.npy", gt)

    # Functions to process data at diffrent stages
    @abstractmethod
    def bronze(
        self,
        data: np.ndarray,
        gt: np.ndarray,
        fs_start: int,
        fs_gt_start: int,
        fs_end: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def silver(self, data) -> np.ndarray:
        pass

    @abstractmethod
    def gold(self, data, gt, chunk_size) -> tuple[np.ndarray[Any], np.ndarray[Any]]:
        pass
