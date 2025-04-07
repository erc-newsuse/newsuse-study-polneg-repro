from pathlib import Path

import dvc.api
import dvc.config
from newsuse.config import Config

from .__about__ import __version__

__all__ = ("__version__", "config", "paths")

root = Path(__file__).parent.parent
config = Config(dvc.api.params_show()).resolve()
paths = config.pop("paths")(root=root)
