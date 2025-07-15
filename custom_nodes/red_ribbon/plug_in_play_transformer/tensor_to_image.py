

class TensorToImage:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "model_size": (["tiny", "small", "medium", "large", "xl"], {"default": "small"}),
            },
            "optional": {
                "num_layers": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "custom_config": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "transformer/models"

    def __init__(self, tensor):
        self.tensor = tensor

    def to_image(self):
        """
        Converts a tensor to an image format.
        Assumes the tensor is in the shape (C, H, W) or (H, W, C).
        """
        if self.tensor.dim() == 3:
            if self.tensor.size(0) == 3:  # RGB
                return self.tensor.permute(1, 2, 0).numpy()
            elif self.tensor.size(0) == 1:  # Grayscale
                return self.tensor.squeeze(0).numpy()
        raise ValueError("Tensor must be of shape (C, H, W) or (H, W, C)")

