from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import yaml


@dataclass
class MainConfig:
    PROCESS_LAYER: Literal["raw", "bronze", "silver", "gold"]

@dataclass
class RawConfig:
    PATH: Path
    DATASET_NAME: str

@dataclass
class DownSampleConfig:
    DO: bool
    FS: int

@dataclass
class BronzeConfig:
    PATH: Path
    FS: int
    DOWN_SAMPLE: DownSampleConfig

@dataclass
class SilverConfig:
    PATH: Path
    SCALE: float
    MATRIX_SIZE: int
    NORM: bool
    STANDARIZATION: bool

@dataclass
class GoldConfig:
    PATH: Path
    CHUNK_SIZE: int
    AXIS_REORDER: tuple[str, ...]

@dataclass
class ModelConfig:
    GPU: int
    NAME: str

@dataclass
class Config:
    MAIN: MainConfig
    RAW: RawConfig
    BRONZE: BronzeConfig
    SILVER: SilverConfig
    GOLD: GoldConfig
    MODEL: ModelConfig


class ConfigLoader:
    @staticmethod
    def load(path: Path) -> Config:
        with open(path, "r") as f:
            data = yaml.full_load(f)
        return Config(
            MAIN=MainConfig(
                PROCESS_LAYER=data["MAIN"]["PROCESS_LAYER"]
            ),
            RAW=RawConfig(
                PATH=Path(data["RAW"]["PATH"]),
                DATASET_NAME=data["RAW"]["DATASET_NAME"]
            ),
            BRONZE=BronzeConfig(
                PATH=Path(data["BRONZE"]["PATH"]),
                FS=data["BRONZE"]["FS"],
                DOWN_SAMPLE=DownSampleConfig(
                    DO=data["BRONZE"]["DOWN_SAMPLE"]["DO"],
                    FS=data["BRONZE"]["DOWN_SAMPLE"]["FS"]
                )
            ),
            SILVER=SilverConfig(
                PATH=Path(data["SILVER"]["PATH"]),
                SCALE=data["SILVER"]["SCALE"],
                MATRIX_SIZE=data["SILVER"]["MATRIX_SIZE"],
                NORM=data["SILVER"]["NORM"],
                STANDARIZATION=data["SILVER"]["STANDARIZATION"]
            ),
            GOLD=GoldConfig(
                PATH=Path(data["GOLD"]["PATH"]),
                CHUNK_SIZE=data["GOLD"]["CHUNK_SIZE"],
                AXIS_REORDER=tuple(data["GOLD"]["AXIS_REORDER"])
            ),
            MODEL=ModelConfig(
                GPU=data["MODEL"]["GPU"],
                NAME=data["MODEL"]["NAME"]
            )
        ) 