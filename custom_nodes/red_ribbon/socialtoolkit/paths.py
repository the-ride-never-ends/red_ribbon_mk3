
from pathlib import Path

THIS_DIR: Path 
THIS_DIR = ROOT = Path(__file__)
CUSTOM_NODES_DIR: Path = THIS_DIR.parent
COMFY_UI_DIR: Path = CUSTOM_NODES_DIR.parent
