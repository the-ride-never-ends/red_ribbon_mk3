#!/usr/bin/env python3
"""
Code Line Counter

This script walks through a directory structure and counts the number of lines in each code file,
excluding data files like JSON, TXT, CSV, etc.

Usage:
    python code_line_counter.py [-h] [--verbose] [--output OUTPUT] 
                               [--exclude PATTERN [PATTERN ...]] 
                               [--exclude-dir DIR [DIR ...]]
                               [directory]

Arguments:
    directory           Directory to analyze (default: current directory)

Options:
    -h, --help          Show this help message and exit
    --verbose, -v       Show detailed progress while counting
    --output, -o        Output file path to save results (default: print to console)
    --exclude, -e       Additional file patterns to exclude (e.g., '*.lock' '*.toml')
    --exclude-dir, -d   Directories to exclude (e.g., 'venv' '.git')

Example:
    python code_line_counter.py /path/to/project --verbose --exclude-dir venv .github node_modules
"""

import os
import sys
import argparse
from collections import defaultdict
import fnmatch


def is_code_file(filename):
    """
    Determine if a file is likely to be a code file based on its extension.
    
    Args:
        filename (str): The name of the file to check
        
    Returns:
        bool: True if the file is likely a code file, False otherwise
    """
    # Data file extensions to exclude
    data_extensions = [
        '*.json', '*.txt', '*.csv', '*.tsv', '*.xml', '*.yaml', '*.yml', 
        '*.properties', '*.config', '*.cfg', '*.ini', '*.dat', '*.data',
        '*.db', '*.sqlite', '*.sqlite3', '*.log', '*.md', '*.markdown',
        '*.rst', '*.docx', '*.pdf', '*.xlsx', '*.xls', '*.pptx', '*.ppt',
        '*.jpg', '*.jpeg', '*.png', '*.gif', '*.svg', '*.bmp', '*.tiff',
        '*.mp3', '*.mp4', '*.wav', '*.avi', '*.mov', '*.zip', '*.tar',
        '*.gz', '*.7z', '*.rar', '*.bin', '*.exe', '*.dll', '*.so', '*.pyc',
        '.gitignore', '.gitattributes', '.gitkeep', '*.lock', '*.key', '*.pem',
        '*.identifier'
    ]
    
    # Check if the file matches any of the excluded patterns
    for pattern in data_extensions:
        if fnmatch.fnmatch(filename.lower(), pattern):
            return False
    
    return True


def count_lines(file_path):
    """
    Count the number of lines in a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        int: Number of lines in the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def walk_directory(directory_path, verbose=False, additional_exclusions=None, excluded_dirs=None):
    """
    Walk through a directory structure and count lines in code files.
    
    Args:
        directory_path (str): Path to the directory to analyze
        verbose (bool): Whether to print detailed progress
        additional_exclusions (list): Additional file patterns to exclude
        excluded_dirs (list): Directories to exclude from analysis
        
    Returns:
        dict: Dictionary with statistics about the files analyzed
    """
    # Initialize counters
    stats = {
        'total_lines': 0,
        'total_files': 0,
        'extension_counts': defaultdict(int),
        'extension_lines': defaultdict(int),
        'files': [],
        'skipped_dirs': set()
    }
    
    if verbose:
        print(f"Analyzing directory: {directory_path}")
    
    # Default excluded directories if not provided
    if excluded_dirs is None:
        excluded_dirs = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory_path, topdown=True):
        # Skip excluded directories by modifying dirs in-place (topdown=True required)
        # This technique prevents os.walk from descending into these directories
        dirs_to_remove = []
        for d in dirs:
            # Check each directory against excluded patterns
            if any(fnmatch.fnmatch(d.lower(), pattern) for pattern in excluded_dirs):
                dirs_to_remove.append(d)
                if verbose:
                    rel_path = os.path.relpath(os.path.join(root, d), directory_path)
                    print(f"Skipping directory: {rel_path}")
                stats['skipped_dirs'].add(os.path.join(root, d))
        
        # Remove the excluded directories from the dirs list
        for d in dirs_to_remove:
            dirs.remove(d)
        
        # Process files in current directory
        for file in files:
            file_path = os.path.join(root, file)

            # Skip the file that runs the script.
            if file_path in [__file__, "*.gitignore"]:
                continue
            
            # Skip if it's not a code file
            if not is_code_file(file):
                continue
                
            # Check against additional exclusions if provided
            if additional_exclusions:
                if any(fnmatch.fnmatch(file.lower(), pattern) for pattern in additional_exclusions):
                    if verbose:
                        print(f"Skipping excluded file: {file_path}")
                    continue
            
            # Get file extension (or filename if no extension)
            _, ext = os.path.splitext(file)
            ext = ext.lower() if ext else 'no_extension'
            
            # Count lines
            if verbose:
                print(f"Counting lines in: {file_path}")
            line_count = count_lines(file_path)
            
            # Update statistics
            stats['total_lines'] += line_count
            stats['total_files'] += 1
            stats['extension_counts'][ext] += 1
            stats['extension_lines'][ext] += line_count
            stats['files'].append({
                'path': file_path,
                'extension': ext,
                'lines': line_count
            })
    
    return stats


def format_stats(stats):
    """
    Format the statistics for display.
    
    Args:
        stats (dict): Dictionary with statistics about the files analyzed
        
    Returns:
        str: Formatted string with statistics
    """
    result = []
    result.append("=" * 80)
    result.append(f"CODE LINE COUNT SUMMARY")
    result.append("=" * 80)
    result.append(f"Total files analyzed: {stats['total_files']}")
    result.append(f"Total lines of code: {stats['total_lines']}")
    
    # Add skipped directories if any
    if 'skipped_dirs' in stats and stats['skipped_dirs']:
        result.append(f"\nSkipped directories: {len(stats['skipped_dirs'])}")
        for skipped_dir in sorted(stats['skipped_dirs']):
            result.append(f"  - {skipped_dir}")
    
    result.append("\nBreakdown by file extension:")
    result.append("-" * 40)
    
    # Sort extensions by line count (descending)
    sorted_extensions = sorted(
        stats['extension_lines'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for ext, lines in sorted_extensions:
        count = stats['extension_counts'][ext]
        avg = lines / count if count > 0 else 0
        result.append(f"{ext:<10} {count:>6} files  {lines:>10} lines  {avg:>10.2f} avg lines/file")
    
    result.append("\nTop 10 largest files:")
    result.append("-" * 80)
    
    # Sort files by line count (descending) and get top 10
    sorted_files = sorted(
        stats['files'],
        key=lambda x: x['lines'],
        reverse=True
    )[:10]
    
    for i, file_info in enumerate(sorted_files, 1):
        result.append(f"{i:>2}. {file_info['lines']:>8} lines  {file_info['extension']:<10}  {file_info['path']}")
    
    return "\n".join(result)


def main():
    """Main function to parse arguments and run the analysis."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Count lines of code in a directory structure, excluding data files.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        'directory',
        nargs='?',
        default=os.getcwd(),
        help='Directory to analyze (default: current directory)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress while counting'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path to save results (default: print to console)'
    )
    parser.add_argument(
        '--exclude', '-e',
        nargs='+',
        metavar='PATTERN',
        help='Additional file patterns to exclude (e.g., *.lock *.toml)'
    )
    parser.add_argument(
        '--exclude-dir', '-d',
        nargs='+',
        metavar='DIR',
        default=[
            'venv', '.venv', '.git', '.github', 'node_modules', '__pycache__', '.idea', '.vs', '.vscode',
            
            ],
        help='Directories to exclude (default: venv .venv .git .github node_modules __pycache__ .idea .vs .vscode)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure the directory path exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        sys.exit(1)
    
    # Analyze the directory
    if args.verbose:
        print(f"Starting analysis with the following settings:")
        print(f"  Directory: {args.directory}")
        print(f"  Additional exclusions: {args.exclude or 'None'}")
        print(f"  Excluded directories: {args.exclude_dir}")
        print(f"  Output: {args.output or 'Console'}")
        print("-" * 40)
    
    stats = walk_directory(
        args.directory,
        verbose=args.verbose,
        additional_exclusions=args.exclude,
        excluded_dirs=args.exclude_dir
    )
    
    # Format the results
    formatted_stats = format_stats(stats)
    
    # Output the results
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(formatted_stats)
            if args.verbose:
                print(f"Results saved to: {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
            print(formatted_stats)  # Fall back to console output
    else:
        print(formatted_stats)


if __name__ == "__main__":
    main()