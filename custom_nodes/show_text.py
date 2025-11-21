
import time
from typing import Any


_curr_preview: dict[str, Any] = {}


class ShowText:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "show_text"

    OUTPUT_NODE = True

    CATEGORY = "red_ribbon/utils"

    def show_text(self, text):
        """Add a preview text to the ComfyUI node.

        Args:
            text (str): The text to display.
        """
        text: str
        if "text" not in _curr_preview:
            _curr_preview["text"] = []

        # If the current text ends with some sort of punctuation.
        # Just append it on. Otherwise, add a newline.
        if len(_curr_preview["text"]) > 0:
            if text.endswith((".", "!", "?", ":", ";", "...", "—")):
                print(_curr_preview["text"][-1])
                _curr_preview["text"][-1] += f"\n{text}"
            else:
                _curr_preview["text"][-1] += text
        else:
            _curr_preview["text"].append(text)

    @classmethod
    def IS_CHANGED(s, text):
        return time.time()

NODE_CLASS_MAPPINGS = {
    "ShowText": ShowText,
}
