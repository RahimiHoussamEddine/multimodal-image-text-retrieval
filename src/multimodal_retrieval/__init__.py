"""License-aware multimodal image-text retrieval workbench."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
