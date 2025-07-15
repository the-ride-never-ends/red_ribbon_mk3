
#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path
from typing import Annotated as Ann, Any


from PIL import Image
from pydantic import AfterValidator as AV, BaseModel, DirectoryPath, Field, NonNegativeInt, ValidationError
import yaml


def _load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict[str, Any]: Parsed configuration dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the YAML file cannot be parsed.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return dict(config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{config_path}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading configuration file '{config_path}': {e}") from e


def _check_ext(extension: str) -> str:
    """Validate image file extension.
    
    Args:
        extension (str): Image file extension to validate.
        
    Returns:
        str: The validated extension.
        
    Raises:
        ValueError: If the extension is not supported.
    """
    valid_extensions = ["png", "jpg", "jpeg"]
    if extension not in valid_extensions:
        raise ValueError(f"Unsupported image extension '{extension}'. Use 'png', 'jpg', or 'jpeg'.")
    return extension


def _ends_with_gif(output_path: str) -> str:
    """Ensure the output path ends with '.gif'.
    
    Args:
        output_path (str): The output file path to validate.
        
    Returns:
        str: The validated output path.
        
    Raises:
        ValueError: If the output path doesn't end with '.gif'.
    """
    if not output_path.endswith(".gif"):
        raise ValueError("Output path must end with '.gif'.")
    return output_path


class _MakeGifArgs(BaseModel):
    """Configuration model for GIF creation arguments."""
    output_path:  Ann[str, AV(_ends_with_gif)] = Field(..., description="Path to the GIF file to be created")
    frame_folder: DirectoryPath                = Field(..., description="Folder containing the frames for the GIF")
    name_pattern: str                          = Field("*", description="Pattern to match frame files (default: '*', matches all files in folder)")
    extension:    Ann[str, AV(_check_ext)]     = Field("png", description="Image file extension of input frames (default: png)")
    duration:     NonNegativeInt               = Field(100, description="Duration between frames in milliseconds (default: 100)")
    loop:         bool                         = Field(True, description="Whether the GIF should loop (default: True)")


def _make_gif(
        output_path: str,
        frame_folder: str, 
        name_pattern: str = "*",
        extension: str = "png",
        duration: int = 100,
        loop: bool = True
        ) -> None:
    """Create a GIF from a folder of images.
    
    Args:
        output_path (str): Path where the GIF will be saved.
        frame_folder (str): Directory containing the frame images.
        name_pattern (str): Glob pattern to match frame files (default: "*").
        extension (str): Image file extension (default: "png").
        duration (int): Duration between frames in milliseconds (default: 100).
        loop (bool): Whether the GIF should loop infinitely (default: True).
        
    Raises:
        ValueError: If no images are found matching the pattern.
        RuntimeError: If the GIF cannot be saved.
    """
    glob_string = f"{frame_folder}/{name_pattern}.{extension}"
    # Prevent unbounded variable errors
    frame_one = None
    frames = []

    # Sort the file paths to ensure correct frame order
    image_paths = sorted(glob.glob(glob_string))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in folder '{frame_folder}' with pattern '{glob_string}'.")

    print(f"Found {len(image_paths)} frames in folder '{frame_folder}' with pattern '{glob_string}'.")

    # Load images with proper resource management
    try:
        frames = [Image.open(path) for path in image_paths]
        
        if not frames:
            raise ValueError(f"No valid images found in folder '{frame_folder}' with pattern '{glob_string}'.")

        frame_one = frames.pop(0)  # Use the first frame as the base

        # Convert loop boolean to PIL's expected format:
        try:
            frame_one.save(
                output_path, 
                format="GIF", 
                append_images=frames,
                save_all=True, 
                duration=duration,
                loop=0 if loop else 1 # 0 = infinite loop, 1 = play once
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save GIF to '{output_path}': {e}") from e

        print(f"GIF saved as '{output_path}' with {len(frames)} frames, duration {duration}ms, loop={loop}.")
        
    finally:
        if frame_one is not None:
            frame_one.close()
        _ = [frame.close() for frame in frames if isinstance(frame, Image.Image)]


def _args_from_argparse() -> _MakeGifArgs:
    """Parse command line arguments and return validated _MakeGifArgs.
    
    Returns:
        _MakeGifArgs: Validated arguments for GIF creation.
        
    Raises:
        ValueError: If the provided arguments are invalid.
    """
    parser = argparse.ArgumentParser(description="Create a GIF from a folder of images.")
    parser.add_argument("--output_path", type=str, required=True, help="Name of the GIF file to be created")
    parser.add_argument("--frame_folder", type=str, required=True, help="Folder containing the frames for the GIF")
    parser.add_argument("--name_pattern", type=str, default="*", help="Pattern to match frame files (default: '*', matches all files in folder)")
    parser.add_argument("--extension", type=_check_ext, default="png", help="Image file extension (default: png). Options: png, jpg, jpeg")
    parser.add_argument("--duration", type=int, default=100, help="Duration between frames in milliseconds (default: 100)")
    parser.add_argument("--loop", action='store_true', help="Whether the GIF should loop (default: True)")

    args = parser.parse_args()

    # Validate and convert arguments using Pydantic
    try:
        validated_args = _MakeGifArgs(**vars(args))
    except ValidationError as e:
        raise ValueError(f"Invalid arguments: {e}") from e
    return validated_args


def main() -> None:
    """Main function to create GIF from configuration file.
    
    Raises:
        ValueError: If configuration is invalid.
        FileNotFoundError: If configuration file is not found.
    """
    # Load configuration file
    config_path = Path(__file__).parent / "make_gif_configs.yaml"
    config = _load_config(config_path)

    # Validate and convert arguments using Pydantic
    try:
        args = _MakeGifArgs(**config)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}") from e

    _make_gif(
        output_path=args.output_path,
        frame_folder=args.frame_folder,
        name_pattern=args.name_pattern,
        extension=args.extension,
        duration=args.duration,
        loop=args.loop
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    else:
        print("GIF created successfully.")
        exit(0)