import cv2
from abc import ABC, abstractmethod
from TOOLS.config_parser import ConfigParser
from TOOLS.config_parser import Layer
from MODELS.model import Model
from pathlib import Path

class Pipeline(ABC):
    def __init__(self, model: Model, config_path: Path) -> None:
        super().__init__()
        self.model = model
        self.config = ConfigParser.parse(config_path)

    def run(self) -> None:
        layer = self.config.main.PROCESS_LAYER

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

        self.model.train()
        self.model.test()

    def raw_loader(self, dataset_type, data_path):
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

    def bronze_loader(self, data_path):
        pass
    def silver_loader(self, data_path):
            pass
    def gold_loader(self, data_path):
            pass

    @abstractmethod
    def bronze(self, data) -> None:
        pass
    @abstractmethod
    def silver(self, data) -> None:
        pass
    @abstractmethod
    def gold(self, data) -> None:
        pass