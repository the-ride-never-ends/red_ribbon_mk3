#!/usr/bin/env python3
"""
Script to inject assertions into test method stubs.

This script reads a JSON file containing test assertions and injects them
into the corresponding test methods in a Python test file.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_assertions(json_path: str) -> Dict[str, List[str]]:
    """
    Load assertions from JSON file.
    
    Args:
        json_path: Path to JSON file containing assertions
        
    Returns:
        Dictionary mapping test names to [assertion, message] pairs
        
    Raises:
        FileNotFoundError: If JSON file does not exist
        json.JSONDecodeError: If JSON file is malformed
        ValueError: If JSON structure is invalid
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Assertions JSON file not found: {json_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in {json_path}: {e.msg}",
            e.doc,
            e.pos
        )
    
    # Validate structure
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    
    for key, value in data.items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(
                f"Invalid assertion format for '{key}': "
                f"expected [assertion, message], got {value}"
            )
    
    return data


def find_test_method(content: str, test_name: str) -> Tuple[int, int]:
    """
    Find the start and end positions of a test method.
    
    Args:
        content: File content as string
        test_name: Name of the test method to find
        
    Returns:
        Tuple of (start_pos, end_pos) for the test method body
    """
    # Pattern to match the test method definition
    pattern = rf'^\s*def {re.escape(test_name)}\(.*?\):'
    
    lines = content.split('\n')
    start_line = None
    
    # Find the method definition
    for i, line in enumerate(lines):
        if re.match(pattern, line):
            start_line = i
            break
    
    if start_line is None:
        return (-1, -1)
    
    # Find the indentation level of the method
    method_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
    
    # Find the end of the method (next method or class end)
    end_line = len(lines)
    for i in range(start_line + 1, len(lines)):
        line = lines[i]
        if line.strip() == '':
            continue
        current_indent = len(line) - len(line.lstrip())
        
        # If we find a line at the same or lower indentation, method is done
        if current_indent <= method_indent and line.strip():
            end_line = i
            break
    
    return (start_line, end_line)


def get_method_body_indent(content: str, start_line: int) -> int:
    """
    Get the indentation level for the method body.
    
    Args:
        content: File content as string
        start_line: Line number where method starts
        
    Returns:
        Number of spaces for method body indentation
    """
    lines = content.split('\n')
    method_line = lines[start_line]
    method_indent = len(method_line) - len(method_line.lstrip())
    
    # Method body should be indented one level more
    return method_indent + 4


def inject_assertion_into_method(
    content: str,
    test_name: str,
    assertion: str,
    message: str
) -> str:
    """
    Inject assertion and message into a test method.
    
    Replaces the FIRST assert statement with the new one, keeps everything else.
    
    Args:
        content: File content as string
        test_name: Name of the test method
        assertion: Assertion statement to inject
        message: Error message f-string to inject
        
    Returns:
        Modified file content
        
    Raises:
        ValueError: If test method not found or content is invalid
    """
    if not content:
        raise ValueError("Content cannot be empty")
    
    lines = content.split('\n')
    start_line, end_line = find_test_method(content, test_name)
    
    if start_line == -1:
        raise ValueError(f"Test method '{test_name}' not found in file")
    
    # Get the body indentation
    body_indent = get_method_body_indent(content, start_line)
    indent_str = ' ' * body_indent
    
    # Find where docstring ends
    docstring_end = start_line + 1
    in_docstring = False
    docstring_quote = None
    
    for i in range(start_line + 1, end_line):
        line = lines[i].strip()
        
        if not in_docstring:
            if line.startswith('"""') or line.startswith("'''"):
                docstring_quote = '"""' if line.startswith('"""') else "'''"
                in_docstring = True
                if line.count(docstring_quote) >= 2:
                    # Single line docstring
                    docstring_end = i + 1
                    break
            elif line and not line.startswith('#'):
                # No docstring, this is body content
                docstring_end = i
                break
        else:
            if docstring_quote in line:
                in_docstring = False
                docstring_end = i + 1
                break
    
    # Build the new assertion line
    new_assertion = f"{indent_str}{assertion}, {message}"
    
    # Find the FIRST assert line in the method body
    first_assert_line = None
    for i in range(docstring_end, end_line):
        line = lines[i].strip()
        if line.startswith('assert '):
            first_assert_line = i
            break
    
    if first_assert_line is not None:
        # Replace the first assert line
        lines[first_assert_line] = new_assertion
        print(f"✓ Replaced assert in '{test_name}'")
    else:
        # No assert found, check if only pass
        has_pass = False
        pass_line = None
        for i in range(docstring_end, end_line):
            line = lines[i].strip()
            if line == 'pass':
                has_pass = True
                pass_line = i
                break
            elif line and not line.startswith('#'):
                break
        
        if has_pass:
            # Replace pass with assertion
            lines[pass_line] = new_assertion
            print(f"✓ Replaced 'pass' with assertion in '{test_name}'")
        else:
            # Insert at beginning of body
            lines.insert(docstring_end, new_assertion)
            print(f"✓ Added assertion to '{test_name}'")
    
    return '\n'.join(lines)


def inject_assertions(
    test_file_path: str,
    assertions_json_path: str,
    output_path: str = None,
    overwrite: bool = False
) -> int:
    """
    Inject assertions from JSON into test file.
    
    Args:
        test_file_path: Path to the test file
        assertions_json_path: Path to JSON file with assertions
        output_path: Optional output path (defaults to overwriting input)
        overwrite: If True, overwrite existing assertions
        
    Returns:
        Number of assertions successfully injected
        
    Raises:
        FileNotFoundError: If test file or JSON file doesn't exist
        PermissionError: If cannot write to output file
        ValueError: If file content or assertions are invalid
    """
    # Validate input file exists
    if not Path(test_file_path).exists():
        raise FileNotFoundError(f"Test file not found: {test_file_path}")
    
    # Load assertions
    try:
        assertions = load_assertions(assertions_json_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to load assertions: {e}")
    
    # Read test file
    try:
        with open(test_file_path, 'r') as f:
            content = f.read()
    except PermissionError:
        raise PermissionError(f"Cannot read test file: {test_file_path}")
    except Exception as e:
        raise IOError(f"Error reading test file: {e}")
    
    # Inject each assertion
    modified_content = content
    injection_count = 0
    failed_injections = []
    
    for test_name, (assertion, message) in assertions.items():
        try:
            old_content = modified_content
            modified_content = inject_assertion_into_method(
                modified_content,
                test_name,
                assertion,
                message
            )
            if old_content != modified_content:
                injection_count += 1
                print(f"✓ Injected assertion into '{test_name}'")
        except ValueError as e:
            failed_injections.append((test_name, str(e)))
            print(f"✗ Failed to inject into '{test_name}': {e}")
    
    # Report failures
    if failed_injections:
        print(f"\nWarning: {len(failed_injections)} assertion(s) could not be injected:")
        for test_name, error in failed_injections:
            print(f"  - {test_name}: {error}")
    
    # Determine output path
    output_path = output_path or test_file_path
    
    # Check if overwrite needed
    if Path(output_path).exists() and output_path != test_file_path and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            f"Use --overwrite to overwrite."
        )
    
    # Write output
    try:
        with open(output_path, 'w') as f:
            f.write(modified_content)
    except PermissionError:
        raise PermissionError(f"Cannot write to output file: {output_path}")
    except Exception as e:
        raise IOError(f"Error writing output file: {e}")
    
    print(f"\nSummary: Injected {injection_count} assertions into {output_path}")
    
    return injection_count


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Inject assertions into test file from JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0 - Success (all assertions injected)
  1 - Partial success (some assertions injected, some failed)
  2 - Failure (no assertions injected or critical error)
  3 - File not found error
  4 - Permission error
  5 - Invalid JSON or data format

Examples:
  # Inject into test file (overwrites original)
  %(prog)s test_file.py assertions.json
  
  # Inject into new file
  %(prog)s test_file.py assertions.json -o output_test.py
  
  # Overwrite existing output file
  %(prog)s test_file.py assertions.json -o output.py --overwrite
        """
    )
    parser.add_argument(
        'test_file',
        help='Path to the test file'
    )
    parser.add_argument(
        'assertions_json',
        help='Path to JSON file containing assertions'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (defaults to overwriting input)',
        default=None
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output file if it exists'
    )
    
    args = parser.parse_args()
    
    try:
        injection_count = inject_assertions(
            args.test_file,
            args.assertions_json,
            args.output,
            args.overwrite
        )
        
        # Exit code based on success
        if injection_count > 0:
            sys.exit(0)  # Success
        else:
            print("ERROR: No assertions were injected", file=sys.stderr)
            sys.exit(2)  # Failure - no injections
            
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)
    except PermissionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(4)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(5)
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()