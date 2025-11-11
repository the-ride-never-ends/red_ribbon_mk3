# !/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


def _make_filepath() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(__file__).parent / "logs" / "mypy" / f"mypy_check_{timestamp}.log"


def type_check(save_results: bool = True, timeout: int = 120) -> bool:
    """
    Run mypy type checking on the 'custom_nodes/red_ribbon' directory.

    Args:
        save_results: If True, saves the mypy output to 'mypy_check_{timestamp}.txt' in this script's directory.
        timeout: Maximum time in seconds to wait for mypy to complete. Defaults to 120 seconds.

    Returns:
        True if type checking passed without errors, False otherwise.

    Raises:
        TypeError: If save_results is not a boolean or timeout is not an integer.
        ValueError: If timeout is not positive.
        IOError: If there is an error writing to the output file.
        subprocess.TimeoutExpired: If mypy command does not complete within the specified timeout.
        subprocess.CalledProcessError: If mypy returns a non-zero exit code.
        RuntimeError: If an unexpected error occurs.
    """
    if not isinstance(save_results, bool):
        raise TypeError(f"save_results must be a boolean, got {type(save_results).__name__}")
    if not isinstance(timeout, int):
        raise TypeError(f"timeout must be an integer, got {type(timeout).__name__}")
    if timeout <= 0:
        raise ValueError(f"timeout must be a positive integer, got {timeout}")

    comfy_directory = Path(__file__).parent.parent
    rr_directory = comfy_directory / "custom_nodes" / "red_ribbon"

    cmd = [
        "mypy", f"{rr_directory.resolve()}", 
        "--follow-imports=silent",  # Don't follow imports outside your code
    ]

    print(f"Starting mypy type checking on '{rr_directory}' with '{cmd}'...")

    try:
        # Run mypy on the target directory
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        # Return True if mypy passed (return code 0), False otherwise
    except subprocess.TimeoutExpired:
        raise
    except subprocess.CalledProcessError:
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred: {e}") from e

    return_code = result.returncode

    # Save results if requested
    if save_results:
        output_path = _make_filepath()

        if output_path.exists():
            # Generate a slightly different timestamp to avoid overwriting
            time.sleep(0.1)
            output_path =  _make_filepath()

        try:
            with open(output_path, 'w') as f:
                f.write(f"Return code: {return_code}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
        except IOError as e:
            raise IOError(f"Error writing to output file: {e}")

    if return_code == 0:
        print("Type checking passed without errors.")
    else:
        print(f"Type checking found issues. See '{output_path}' for details.")

    return return_code == 0, output_path


def main():
    parser = argparse.ArgumentParser(description="Run mypy type checking on Red Ribbon custom nodes.")
    parser.add_argument(
        "--save_results",
        action="store_true",
        default=True,
        help="Whether to save the mypy output to a file."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Maximum time in seconds to wait for mypy to complete."
    )
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        return 1

    try:
        _ = type_check(save_results=args.save_results, timeout=args.timeout)
    except Exception as e:
        print(f"Error during type checking: {e}")
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
