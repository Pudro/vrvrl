from dataclasses import dataclass
from pathlib import Path


@dataclass
class Parameters:
    MODEL_DIR: Path = Path.cwd()/'models'
