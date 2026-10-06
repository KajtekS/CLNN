import yaml
from enum import Enum


class ConfigParser:
    @staticmethod
    def parse(path):
        with open(path, "r") as f:
            data = yaml.full_load(f)
        return data


class Layer(Enum):
    RAW = 0
    BRONZE = 1
    SILVER = 2
    GOLD = 3
