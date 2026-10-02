import yaml

class ConfigParser:
    @staticmethod
    def parse(path):
        with open(path, 'r') as f:
            data = yaml.full_load(f)
        return data