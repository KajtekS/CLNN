from pathlib import Path
from PIPELINE.pipeline import Pipeline
from CONFIGS.yaml_class import ConfigLoader
from PIPELINE.piper import Piper

class ExperimentRunner:
    def __init__(self, config_path: Path) -> None:
        self.config = ConfigLoader.load(config_path)
        self.config_path = config_path
    def run(self):
        pipeline = Piper()
        pipeline.run(self.config, self.config_path)