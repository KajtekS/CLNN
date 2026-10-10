from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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