#!/usr/bin/env python3
"""
CV Training Data Train-Test Split Script
This script efficiently distributes image-JSON pairs from a source directory to train and test directories
with a specified split ratio (default 95% train, 5% test). It ensures paired files stay together,
empties destination directories before processing, and uses multithreading for performance.

Author: Rokawoo
"""

import os
import shutil
import random
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to script location
base_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))  # Go back 4 folders
DEFAULT_SOURCE = os.path.join(base_dir, "CVHelperScripts", "labeled_cv_training_data")
DEFAULT_TRAIN_DIR = os.path.join(SCRIPT_DIR, "train")
DEFAULT_TEST_DIR = os.path.join(SCRIPT_DIR, "test")

DEFAULT_TRAIN_RATIO = 0.97
DEFAULT_MIN_TEST_SAMPLES = 1
DEFAULT_THREADS = 4

def find_image_json_pairs(source_dir):
    """Find all image-JSON pairs in the source directory."""
    source_path = Path(source_dir)
    
    # Create dictionaries to store files by stem (filename without extension)
    image_files = {}
    json_files = {}
    
    # Categorize files by extension
    for file in source_path.glob('*'):
        stem = file.stem
        if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            image_files[stem] = file
        elif file.suffix.lower() == '.json':
            json_files[stem] = file
    
    # Find files that have both image and JSON
    pairs = []
    for stem in set(image_files.keys()) & set(json_files.keys()):
        pairs.append((image_files[stem], json_files[stem]))
    
    print(f"Found {len(pairs)} image-JSON pairs in {source_dir}")
    return pairs

def empty_directory(directory_path, force=False):
    """Empty a directory of all files and subdirectories."""
    dir_path = Path(directory_path)
    
    # Check if directory exists
    if not dir_path.exists():
        return 0, True  # Nothing to delete
    
    # Check if directory is empty
    if not any(dir_path.iterdir()):
        return 0, True  # Already empty
    
    # Count files for reporting
    file_count = sum(1 for _ in dir_path.glob('**/*'))
    
    # Prompt for confirmation if not forced
    if not force:
        response = input(f"Directory {directory_path} contains {file_count} items. Empty it? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return 0, False
    
    print(f"Emptying directory: {directory_path}")
    start_time = time.time()
    
    try:
        # Remove all contents
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        duration = time.time() - start_time
        print(f"Removed {file_count} items in {duration:.2f} seconds")
        return file_count, True
    
    except Exception as e:
        print(f"Error emptying directory {directory_path}: {str(e)}")
        return 0, False

def copy_file_pair(pair, dest_dir):
    """Copy an image-JSON pair to the destination directory."""
    image_file, json_file = pair
    
    try:
        # Copy files
        shutil.copy2(image_file, Path(dest_dir) / image_file.name)
        shutil.copy2(json_file, Path(dest_dir) / json_file.name)
        return True, 2  # 2 files copied
    except Exception as e:
        print(f"Error copying {image_file.name} or {json_file.name}: {e}")
        return False, 0

def split_and_copy_data(
    source_dir=None,
    train_dir=None,
    test_dir=None,
    train_ratio=DEFAULT_TRAIN_RATIO,
    min_test_samples=DEFAULT_MIN_TEST_SAMPLES,
    threads=DEFAULT_THREADS,
    force_empty=False,
    seed=None
):
    """Main function to split and copy data between train and test directories."""
    # Use default paths if none provided
    source_dir = source_dir if source_dir is not None else DEFAULT_SOURCE
    train_dir = train_dir if train_dir is not None else DEFAULT_TRAIN_DIR
    test_dir = test_dir if test_dir is not None else DEFAULT_TEST_DIR
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Print configuration
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Source: {source_dir}")
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Train/Test ratio: {train_ratio:.2f}/{1-train_ratio:.2f}")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return None
    
    # Empty destination directories
    empty_directory(train_dir, force=force_empty)
    empty_directory(test_dir, force=force_empty)
    
    # Create destination directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Find all image-JSON pairs
    pairs = find_image_json_pairs(source_dir)
    if not pairs:
        print("No image-JSON pairs found. Exiting.")
        return None
    
    # Calculate split
    num_pairs = len(pairs)
    num_test = max(min_test_samples, math.ceil(num_pairs * (1 - train_ratio)))
    num_train = num_pairs - num_test
    
    # Shuffle and split pairs
    random.shuffle(pairs)
    train_pairs = pairs[:num_train]
    test_pairs = pairs[num_train:]
    
    print(f"\nSplitting {num_pairs} pairs:")
    print(f"  - {len(train_pairs)} pairs ({len(train_pairs)/num_pairs:.1%}) to train directory")
    print(f"  - {len(test_pairs)} pairs ({len(test_pairs)/num_pairs:.1%}) to test directory")
    
    # Function to copy all pairs to a destination
    def copy_all_to_dest(pair_list, dest_dir):
        success_count = 0
        files_copied = 0
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(copy_file_pair, pair, dest_dir) 
                for pair in pair_list
            ]
            
            for future in futures:
                success, copied = future.result()
                if success:
                    success_count += 1
                files_copied += copied
        
        return success_count, files_copied
    
    # Copy files to destinations
    print(f"Copying files to train directory...")
    train_success, train_copied = copy_all_to_dest(train_pairs, train_dir)
    
    print(f"Copying files to test directory...")
    test_success, test_copied = copy_all_to_dest(test_pairs, test_dir)
    
    # Return summary
    results = {
        'total_pairs': num_pairs,
        'train_pairs': len(train_pairs),
        'test_pairs': len(test_pairs),
        'train_success': train_success,
        'test_success': test_success,
        'train_copied': train_copied,
        'test_copied': test_copied
    }
    
    print(f"\nSummary:")
    print(f"  Train set: {results['train_success']}/{results['train_pairs']} pairs")
    print(f"  Test set: {results['test_success']}/{results['test_pairs']} pairs")
    print(f"  Total files copied: {results['train_copied'] + results['test_copied']}")
    
    return results

if __name__ == "__main__":
    # Execute the function when script is run directly
    split_and_copy_data(force_empty=True)