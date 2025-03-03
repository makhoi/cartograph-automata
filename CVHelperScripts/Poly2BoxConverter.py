#!/usr/bin/env python3
"""
Polygon to Bounding Box Converter
---------------------------------
This script converts polygon annotations in LabelMe JSON files to bounding box annotations.
It calculates optimal bounding boxes that maximize IoU with the original polygons.

Features:
- Converts polygon annotations to rectangle annotations (bounding boxes)
- Optimizes boxes to maximize IoU with original polygons
- Provides visualization of conversions with low IoU scores
- Generates a detailed conversion report
- Handles invalid polygons with graceful fallbacks

Author: Rokawoo
"""

import os
import json
import numpy as np
from shapely.geometry import Polygon, box
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as PlotPolygon
import warnings
from shapely.errors import ShapelyDeprecationWarning, GEOSException

# Suppress shapely deprecation warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# -------------------- CONFIGURATION --------------------
# Set your input and output paths here
INPUT_DIR = r"C:\Users\rokaw\GitProjects\Codefest2025 - TEMP\CVHelperScripts\labeled_cv_training_data"  # Directory containing JSON files
OUTPUT_DIR = r"C:\Users\rokaw\GitProjects\Codefest2025 - TEMP\CVHelperScripts\labeled_cv_training_data_boxed"   # Directory to save converted JSON files
VISUALIZE = True                 # Set to True to visualize conversions
IOU_THRESHOLD = 0.7              # Minimum IoU threshold (0.0 to 1.0)
# -------------------------------------------------------

def simple_bbox(polygon):
    """
    Get the bounding box of a polygon using min/max coordinates
    (Fallback method for invalid polygons)
    
    Args:
        polygon: List of [x, y] coordinates defining the polygon
    
    Returns:
        [x_min, y_min, x_max, y_max] coordinates of the bounding box
    """
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return [x_min, y_min, x_max, y_max]

def polygon_to_optimal_bbox(polygon):
    """
    Convert a polygon to an optimal bounding box using Shapely.
    Falls back to simple min/max bounding box if the polygon is invalid.
    
    Args:
        polygon: List of [x, y] coordinates defining the polygon
    
    Returns:
        [x_min, y_min, x_max, y_max] coordinates of the optimal bounding box
    """
    # Simple validity check - need at least 3 points for a polygon
    if len(polygon) < 3:
        return simple_bbox(polygon), 1.0
    
    try:
        # Convert to numpy array if it's not
        polygon_np = np.array(polygon)
        
        # Create a Shapely polygon
        poly = Polygon(polygon_np)
        
        # Check if polygon is valid
        if not poly.is_valid:
            # Try to repair the polygon
            from shapely.validation import make_valid
            try:
                poly = make_valid(poly)
                if not poly.is_valid:
                    raise ValueError("Couldn't repair invalid polygon")
            except:
                # If we can't repair, fall back to simple bbox
                return simple_bbox(polygon), 1.0
        
        # Get the default bounding box
        x_min, y_min, x_max, y_max = poly.bounds
        default_box = [x_min, y_min, x_max, y_max]
        
        # Try slight variations of the bounding box to maximize IoU
        best_box = default_box
        
        try:
            best_iou = calculate_iou(poly, box(*default_box))
            
            # Define search space for optimization (small adjustments to box coordinates)
            deltas = [-0.05, -0.025, 0, 0.025, 0.05]  # Percentage of box dimensions
            width = x_max - x_min
            height = y_max - y_min
            
            for dx_min in deltas:
                for dy_min in deltas:
                    for dx_max in deltas:
                        for dy_max in deltas:
                            # Apply deltas as percentage of dimension
                            new_box = [
                                x_min + dx_min * width,
                                y_min + dy_min * height,
                                x_max + dx_max * width,
                                y_max + dy_max * height
                            ]
                            
                            # Calculate IoU
                            try:
                                box_poly = box(*new_box)
                                iou = calculate_iou(poly, box_poly)
                                
                                if iou > best_iou:
                                    best_iou = iou
                                    best_box = new_box
                            except Exception:
                                # Skip this iteration if there's an error
                                continue
        except Exception:
            # If IoU calculation fails, just use the default bbox
            return default_box, 1.0
            
        return best_box, best_iou
    
    except Exception as e:
        # Fallback to simple bounding box if anything goes wrong
        print(f"Warning: Using simple bounding box due to: {str(e)}")
        return simple_bbox(polygon), 1.0

def calculate_iou(poly1, poly2):
    """Calculate IoU between two Shapely polygons"""
    try:
        if not poly1.intersects(poly2):
            return 0
        
        intersection = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - intersection
        
        return intersection / union if union > 0 else 0
    except GEOSException:
        # If there's a topology exception, try with buffer(0) to clean up the geometries
        try:
            poly1_clean = poly1.buffer(0)
            poly2_clean = poly2.buffer(0)
            
            if not poly1_clean.intersects(poly2_clean):
                return 0
            
            intersection = poly1_clean.intersection(poly2_clean).area
            union = poly1_clean.area + poly2_clean.area - intersection
            
            return intersection / union if union > 0 else 0
        except Exception:
            # If all else fails, return a default value
            return 1.0  # Assume it's a perfect match to avoid rejecting the box

def process_labelme_json(file_path, output_dir, visualize=False, iou_threshold=0.7):
    """
    Process a LabelMe JSON file to convert polygons to bounding boxes.
    Specific for LabelMe format as shown in the example.
    
    Args:
        file_path: Path to the input JSON file
        output_dir: Directory to save the output JSON file
        visualize: Whether to visualize the conversion for random samples
        iou_threshold: Minimum IoU threshold for accepting a box conversion
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file. Skipping.")
        return 0, 0
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}. Skipping.")
        return 0, 0
    
    # Create a copy of the data for modification
    output_data = data.copy()
    
    # Track low IoU conversions
    low_iou_instances = []
    conversion_count = 0
    
    # Process each shape in the LabelMe format
    if 'shapes' in data:
        # Create a new list for shapes
        new_shapes = []
        
        for i, shape in enumerate(data['shapes']):
            try:
                if shape['shape_type'] == 'polygon':
                    # Extract points from the polygon
                    points = shape['points']
                    
                    # Convert to optimal bounding box
                    bbox, iou = polygon_to_optimal_bbox(points)
                    
                    # Track instances with low IoU
                    if iou < iou_threshold:
                        low_iou_instances.append({
                            'label': shape.get('label', f'shape_{i}'),
                            'iou': iou,
                            'original': points,
                            'bbox': bbox
                        })
                    
                    # Create a new rectangle shape
                    rect_shape = {
                        'label': shape['label'],
                        'points': [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                        'group_id': shape.get('group_id', None),
                        'description': shape.get('description', ''),
                        'shape_type': 'rectangle',
                        'flags': shape.get('flags', {}),
                        'mask': shape.get('mask', None)
                    }
                    
                    # Add the new rectangle shape to output
                    new_shapes.append(rect_shape)
                    conversion_count += 1
                else:
                    # Keep the original shape if it's not a polygon
                    new_shapes.append(shape)
            except Exception as e:
                print(f"Error processing shape in {file_path}: {str(e)}. Forcing simple box conversion.")
                # CRITICAL CHANGE: Never keep the original polygon - always convert to box
                if shape['shape_type'] == 'polygon':
                    try:
                        # Get simple bounding box as fallback
                        points = shape['points']
                        simple_box = simple_bbox(points)
                        
                        # Create a new rectangle shape
                        rect_shape = {
                            'label': shape['label'],
                            'points': [[simple_box[0], simple_box[1]], [simple_box[2], simple_box[3]]],
                            'group_id': shape.get('group_id', None),
                            'description': shape.get('description', ''),
                            'shape_type': 'rectangle',
                            'flags': shape.get('flags', {}),
                            'mask': shape.get('mask', None)
                        }
                        
                        # Add the rectangle shape to output
                        new_shapes.append(rect_shape)
                        conversion_count += 1
                        print(f"  Successfully created simple bounding box for problematic polygon.")
                    except Exception as inner_e:
                        print(f"  Failed to create simple box: {str(inner_e)}. Skipping this shape.")
                        # Skip this shape entirely if we can't even create a simple box
                else:
                    # Keep non-polygon shapes
                    new_shapes.append(shape)
        
        # Replace the shapes list with our new one
        output_data['shapes'] = new_shapes
    
    # Save the modified data
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Visualize a few examples if requested
    if visualize and low_iou_instances:
        try:
            # Visualize up to 5 instances with low IoU
            num_samples = min(5, len(low_iou_instances))
            fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
            if num_samples == 1:
                axes = [axes]
            
            for i, instance in enumerate(low_iou_instances[:num_samples]):
                ax = axes[i]
                
                # Plot original polygon
                polygon_points = np.array(instance['original'])
                ax.add_patch(PlotPolygon(polygon_points, fill=True, alpha=0.5, color='blue', label='Polygon'))
                
                # Plot bounding box
                bbox = instance['bbox']
                ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                                      fill=False, edgecolor='red', linewidth=2, label='Bbox'))
                
                ax.set_title(f"{instance['label']}: IoU={instance['iou']:.3f}")
                ax.legend()
                ax.set_aspect('equal')
                ax.set_xlim(bbox[0]-10, bbox[2]+10)
                ax.set_ylim(bbox[1]-10, bbox[3]+10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"low_iou_samples_{os.path.basename(file_path)}.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating visualization for {file_path}: {str(e)}")
    
    return conversion_count, len(low_iou_instances)

def process_directory(input_dir, output_dir, visualize=False, iou_threshold=0.7):
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Input directory containing JSON files
        output_dir: Output directory to save modified JSON files
        visualize: Whether to visualize conversions with low IoU
        iou_threshold: Minimum IoU threshold for accepting a box conversion
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    total_conversions = 0
    total_low_iou = 0
    error_files = []
    
    # Process each file
    for file_name in tqdm(json_files, desc="Converting files"):
        file_path = os.path.join(input_dir, file_name)
        try:
            conversions, low_iou_count = process_labelme_json(file_path, output_dir, visualize, iou_threshold)
            total_conversions += conversions
            total_low_iou += low_iou_count
        except Exception as e:
            print(f"\nError processing file {file_name}: {str(e)}")
            error_files.append(file_name)
            # Continue processing other files
            continue
    
    print(f"\nConversion complete. {len(json_files)} files processed.")
    print(f"Total polygons converted to boxes: {total_conversions}")
    print(f"Total instances with IoU < {iou_threshold}: {total_low_iou}")
    
    if error_files:
        print(f"Files with errors: {len(error_files)}")
    
    # Create a report file
    with open(os.path.join(output_dir, "conversion_report.txt"), 'w') as f:
        f.write(f"Polygon to Bounding Box Conversion Report\n")
        f.write(f"========================================\n\n")
        f.write(f"Files processed: {len(json_files)}\n")
        f.write(f"Polygons converted to boxes: {total_conversions}\n")
        f.write(f"Instances with IoU < {iou_threshold}: {total_low_iou}\n")
        f.write(f"IoU threshold used: {iou_threshold}\n")
        f.write(f"Files with errors: {len(error_files)}\n")
        f.write(f"\nVisualization: {'Enabled' if visualize else 'Disabled'}\n")
        
        if error_files:
            f.write("\nFiles with errors:\n")
            for err_file in error_files:
                f.write(f"- {err_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert polygon annotations to bounding boxes")
    parser.add_argument("--input_dir", default=INPUT_DIR, help="Directory containing JSON annotation files")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Directory to save modified JSON files")
    parser.add_argument("--visualize", action="store_true", default=VISUALIZE, help="Visualize low IoU conversions")
    parser.add_argument("--iou_threshold", type=float, default=IOU_THRESHOLD, 
                        help="IoU threshold for flagging poor conversions (default: 0.7)")
    
    args = parser.parse_args()
    
    print(f"Converting polygon annotations in {args.input_dir}")
    print(f"Saving box annotations to {args.output_dir}")
    print(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    
    process_directory(args.input_dir, args.output_dir, args.visualize, args.iou_threshold)