import cv2
import numpy as np
from abc import ABC, abstractmethod
from TOOLS.config_parser import ConfigParser
from TOOLS.config_parser import Layer
from MODELS.model import Model
from pathlib import Path


class Pipeline(ABC):
    def __init__(self, model: Model, config_path: Path) -> None:
        super().__init__()
        self.config = ConfigParser.parse(config_path)

    def run(self) -> np.ndarray:
        layer = self.config.main.PROCESS_LAYER

        # Stage of loading and processing data
        match layer:
            case 0:
                data = self.raw_loader(
                    self.config.main.DATASET,
                    self.config.raw.PATH
                )
                data = self.bronze(data)
                data = self.silver(data)
                data = self.gold(data)

            case 1:
                data = self.bronze_loader(
                    self.config.bronze.PATH
                )
                data = self.silver(data)
                data = self.gold(data)

            case 2:
                data = self.silver_loader(
                    self.config.silver.PATH
                )
                data = self.gold(data)

            case 3:
                data = self.gold_loader(
                    self.config.gold.PATH
                )

            case _:
                raise ValueError(f"Invalid PROCESS_LAYER: {layer}")

        return data

    # Functions to load data from diffrent stages of processing
    def raw_loader(self, dataset_type, data_path) -> cv2.VideoCapture:
        match(dataset_type):
            case 'PURE':
                pass
            case 'rPPG':
                pass
            case 'PHYS':
                pass
            case 'SUMS':
                pass
            case _:
                raise ValueError(f"{dataset_type} is invalid DATASET name")

    def bronze_loader(self, data_path) -> cv2.VideoCapture:
        pass
    def silver_loader(self, data_path) -> np.ndarray:
        pass
    def gold_loader(self, data_path) -> np.ndarray:
        pass

    # Functions to process data at diffrent stages
    @abstractmethod
    def bronze(self, data) -> list[tuple[cv2.VideoCapture, int]]:
        pass
    @abstractmethod
    def silver(self, data) -> np.ndarray:
        pass
    @abstractmethod
    def gold(self, data) -> np.ndarray:
        pass