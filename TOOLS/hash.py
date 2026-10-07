from pathlib import Path
import hashlib
import yaml

def hash_yaml(path: Path, size = 16) -> str:
    m = hashlib.sha256()
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    m.update(str(data).encode('utf-8'))
    hash = m.hexdigest()

    return hash[:size]

print(hash_yaml(Path('CONFIGS/HOW_CONFIG_SHOOULD_LOOK_LIKE.yaml')))