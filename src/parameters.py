from dataclasses import dataclass
from pathlib import Path


@dataclass
class Parameters:
    MODEL_DIR: Path = Path.cwd()/'models'
    RESULTS_DIR: Path = Path.cwd()/'results'
    TENSORBOARD_DIR: Path = RESULTS_DIR / 'tensorboard'
