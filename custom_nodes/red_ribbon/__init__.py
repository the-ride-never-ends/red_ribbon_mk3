"""
Red Ribbon - A collection of custom nodes for ComfyUI
"""
import easy_nodes
import os
from .__version__ import __version__

# NOTE This only needs to be called once.
easy_nodes.initialize_easy_nodes(default_category="Red Ribbon", auto_register=False)

# Import all modules - this must come after calling initialize_easy_nodes
from .main import *  # noqa: F403, E402

# Get the combined node mappings for ComfyUI
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = easy_nodes.get_node_mappings()
WEB_DIRECTORY = "./js"

# Export so that ComfyUI can pick them up.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

easy_nodes.save_node_list(
    os.path.join(os.path.dirname(__file__), "node_list.json")
)

# Version information
print(f"Red Ribbon v{__version__}: Successfully loaded {len(NODE_CLASS_MAPPINGS)} nodes")