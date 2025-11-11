"""
Red Ribbon - A collection of custom nodes for ComfyUI
"""
from .custom_easy_nodes import easy_nodes


import os
from .__version__ import __version__ as red_ribbon_version

class Counter:
    _initialized = False
    count = 0

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Counter, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # if Counter._initialized:
        #     return
        self.count += 1
        self._initialized = True

# Initialize Easy Nodes (has built-in protection against multiple calls)
counter = Counter()
print(f"Red Ribbon __init__.py CALL_COUNT: {counter.count}")
try:
    easy_nodes.initialize_easy_nodes(default_category="Red Ribbon", auto_register=False)
except Exception as e:
    print(f"Error initializing Easy Nodes: {e}")
    raise e

# Import all modules - this must come after calling initialize_easy_nodes
from .main import *  # type: ignore # noqa: F403, E402

# Get the combined node mappings for ComfyUI
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = easy_nodes.get_node_mappings()
WEB_DIRECTORY = "./js"

# Export so that ComfyUI can pick them up.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

easy_nodes.save_node_list(
    os.path.join(os.path.dirname(__file__), "node_list.json")
)

# Version information
print(f"Red Ribbon v{red_ribbon_version}: Successfully loaded {len(NODE_CLASS_MAPPINGS)} nodes")
