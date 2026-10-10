import hashlib
import json
from pathlib import Path

import yaml


def hash_yaml(path: Path, size = 16) -> str:
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    serialized = json.dumps(data, sort_keys=True).encode("utf-8")
    hash = hashlib.sha256(serialized).hexdigest()

    return hash[:size]