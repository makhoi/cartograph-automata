#!/usr/bin/env python3
"""
LabelMe JSON Validator
---------------------
This script validates LabelMe JSON annotation files to ensure they have proper
shapes and labels for computer vision training.
"""

import os
import json
import sys

def validate_json_files(directory_path):
    """
    Checks all JSON files in the specified directory to ensure they:
    1. Have shapes with 'shape_type' == 'rectangle' only
    2. Have labels that are only 'shelf', 'person', or 'line'
    
    Args:
        directory_path: Path to the directory containing JSON files
    
    Returns:
        A tuple of (valid_files, invalid_files) where invalid_files is a dict
        mapping filenames to lists of errors
    """
    valid_files = []
    invalid_files = {}
    valid_labels = {'shelf', 'person', 'line'}
    
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return [], {"directory_error": ["Invalid directory path"]}
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"Warning: No JSON files found in {directory_path}")
        return [], {"no_files": ["No JSON files found in directory"]}
    
    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        errors = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if 'shapes' key exists
            if 'shapes' not in data:
                errors.append("Missing 'shapes' key")
                continue
                
            # Check each shape
            for i, shape in enumerate(data['shapes']):
                # Check if shape has valid type
                if 'shape_type' not in shape:
                    errors.append(f"Shape #{i+1} is missing 'shape_type'")
                elif shape['shape_type'] != 'rectangle':
                    errors.append(f"Shape #{i+1} has invalid shape_type: '{shape['shape_type']}'. Only 'rectangle' is allowed")
                
                # Check if shape has valid label
                if 'label' not in shape:
                    errors.append(f"Shape #{i+1} is missing 'label'")
                elif shape['label'] not in valid_labels:
                    errors.append(f"Shape #{i+1} has invalid label: '{shape['label']}'. Valid labels are {', '.join(valid_labels)}")
            
            # Add to appropriate list
            if errors:
                invalid_files[json_file] = errors
            else:
                valid_files.append(json_file)
                
        except json.JSONDecodeError:
            invalid_files[json_file] = ["Invalid JSON format"]
        except Exception as e:
            invalid_files[json_file] = [f"Error processing file: {str(e)}"]
    
    return valid_files, invalid_files

def print_results(valid_files, invalid_files):
    """Print validation results in a readable format"""
    total_files = len(valid_files) + len(invalid_files)
    
    print(f"\nValidation Results ({total_files} files processed):")
    print("=" * 60)
    
    print(f"\n✅ Valid Files: {len(valid_files)}/{total_files}")
    if valid_files:
        for file in valid_files:
            print(f"  - {file}")
    
    print(f"\n❌ Invalid Files: {len(invalid_files)}/{total_files}")
    if invalid_files:
        for file, errors in invalid_files.items():
            print(f"  - {file}:")
            for error in errors:
                print(f"      * {error}")
    
    print("\n" + "=" * 60)

def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use command line argument if provided, otherwise use directory relative to script
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = os.path.join(script_dir, "labeled_cv_training_data_boxed")
    
    print(f"Validating JSON files in: {directory_path}")
    valid_files, invalid_files = validate_json_files(directory_path)
    print_results(valid_files, invalid_files)
    
    # Return non-zero exit code if any invalid files found
    if invalid_files:
        sys.exit(1)

if __name__ == "__main__":
    main()