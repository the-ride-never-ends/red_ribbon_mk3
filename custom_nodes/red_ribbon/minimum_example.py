


from easy_nodes import (
    NumberInput,
    ComfyNode,
    StringInput,
    Choice,
)


from .configs import Configs
from .__version__ import __version__

@ComfyNode(
    category="Red Ribbon",
    display_name="Red Ribbon Core",
    description="Core functionality for Red Ribbon",
)
def test_red_ribbon(
    str_input: str
) -> str:
    """Test Red Ribbon Core"""
    pass