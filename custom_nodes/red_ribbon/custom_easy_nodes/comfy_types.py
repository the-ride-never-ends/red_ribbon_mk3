from typing import Type, Any


import torch


from comfy.clip_vision import ClipVisionModel # type: ignore
from comfy.sd import VAE # type: ignore


class ImageTensor(torch.Tensor):
    pass
class MaskTensor(torch.Tensor): 
    pass
class LatentTensor(torch.Tensor):
    pass
class ConditioningTensor(torch.Tensor):
    pass
class ModelTensor(torch.Tensor):
    pass
class SigmasTensor(torch.Tensor):
    pass

# Maybe there's an actual class for this?
class PhotoMaker:
    pass

# Abstract type, not for instantiating.
class NumberType:
    pass

class _VerifyTypes:

    _initialized = False

    def __new__(cls):
        """Singleton pattern to prevent multiple instances."""
        if not hasattr(cls, 'instance'):
            cls.instance = super(_VerifyTypes, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        if _VerifyTypes._initialized:
            return

        from .easy_nodes import AnythingVerifier, TensorVerifier, TypeVerifier, register_type, AnyType, any_type

        type_list: list[tuple[Type, str, TypeVerifier]] = [
            # ComfyUI will get the special string that AnyType is registered with, which is hardcoded to match anything.
            (AnyType,            "ANY",          Any),
            # Primitive types
            (int,                "INT",          None),
            (float,              "FLOAT",        [float, int]),
            (str,                "STRING",       None),
            (bool,               "BOOLEAN",      None),
            (NumberType,         "NUMBER",       [float, int]),
            # Custom types
            (ClipVisionModel,    "CLIP_VISION",  Any),
            (VAE,                "VAE",          Any),
            (PhotoMaker,         "PHOTOMAKER",   Any),
            (ImageTensor,        "IMAGE",        ("IMAGE", {"allowed_shapes":[4]}, {"allowed_channels": [1, 3, 4]})),
            (MaskTensor,         "MASK",         ("MASK",  {"allowed_shapes":[3]}, {"allowed_range": [0, 1]})),
            (LatentTensor,       "LATENT",       ("LATENT")),
            (ConditioningTensor, "CONDITIONING", ("CONDITIONING")),
            (ModelTensor,        "MODEL",        ("MODEL")),
            (SigmasTensor,       "SIGMAS",       ("SIGMAS")),
        ]
        for cls, name, verifier in type_list:
            match verifier:
                case list():
                    # If verifier is a list, it means we have a TypeVerifier
                    verifier = TypeVerifier(verifier)
                case tuple():
                    # If verifier is a tuple, it means we have a TensorVerifier
                    args = (arg for arg in verifier if not isinstance(arg, dict))
                    kwargs = kwargs in verifier if isinstance(verifier, dict) else {}
                    verifier = TensorVerifier(*verifier, **kwargs)
                case _:
                    verifier = AnythingVerifier()
            try:
                register_type(cls, name, verifier=verifier)
            except Exception as e:
                print(f"Failed to register type {cls.__name__} as {name}: {e}")
        _VerifyTypes._initialized = True


_VerifyTypes()  # Initialize and register types


# Define __all__ for easy imports
__all__ = [
    "ImageTensor",
    "MaskTensor",
    "LatentTensor",
    "ConditioningTensor",
    "ModelTensor",
    "SigmasTensor",
    "PhotoMaker",
    "NumberType",
    "NumberInput"
]
