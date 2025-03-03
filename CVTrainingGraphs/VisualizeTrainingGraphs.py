"""
TensorFlow Training Metrics Visualizer
--------------------------------------
This script creates comprehensive visualizations of TensorFlow training metrics
from CSV files in the specified directory and displays them in interactive windows.

Author: Rokawoo
Date: March 2, 2025
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_training_metrics(csv_dir, smooth_factor=0.8):
    """
    Creates interactive visualizations for TensorFlow training metrics from CSV files.
    
    Args:
        csv_dir (str): Directory containing CSV files
        smooth_factor (float): Factor for exponential moving average smoothing (0-1)
    """
    # Get all CSV files in the directory
    csv_files = list(Path(csv_dir).glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    # Create figure for combined plot
    plt.figure(figsize=(12, 8))
    
    # Color palette for consistent colors
    colors = plt.cm.tab10.colors
    
    # Track min/max values for y-axis scaling
    all_min, all_max = float('inf'), float('-inf')
    
    # Process each CSV file
    metrics_data = []
    
    for idx, csv_path in enumerate(csv_files):
        try:
            # Extract metric name from filename
            metric_name = csv_path.stem
            
            # Load data
            df = pd.read_csv(csv_path)
            
            # Some TensorFlow CSV files use different column names, try to handle common variations
            step_col = next((col for col in df.columns if col.lower() in ['step', 'steps', 'iteration', 'iterations']), df.columns[0])
            value_col = next((col for col in df.columns if col.lower() in ['value', 'values', 'loss']), df.columns[1])
            
            # Extract steps and values
            steps = df[step_col].values
            values = df[value_col].values
            
            # Apply exponential moving average for smoothing
            smoothed_values = apply_smoothing(values, smooth_factor)
            
            # Update min/max for y-axis scaling
            all_min = min(all_min, np.min(values))
            all_max = max(all_max, np.max(values))
            
            # Add to combined plot
            color = colors[idx % len(colors)]
            plt.plot(steps, values, color=color, alpha=0.3, linewidth=1)
            plt.plot(steps, smoothed_values, color=color, linewidth=2, label=metric_name)
            
            # Store data for individual plots
            metrics_data.append({
                'steps': steps,
                'values': values,
                'smoothed_values': smoothed_values,
                'metric_name': metric_name,
                'color': color
            })
            
        except Exception as e:
            print(f"Error processing {csv_path}: {str(e)}")
    
    # Finalize combined plot
    finalize_combined_plot(all_min, all_max)
    
    # Create individual plots for each metric
    for metric in metrics_data:
        create_individual_plot(
            metric['steps'], 
            metric['values'], 
            metric['smoothed_values'], 
            metric['metric_name'], 
            metric['color']
        )
    
    # Show all plots
    plt.show()


def apply_smoothing(values, smooth_factor):
    """Apply exponential moving average smoothing to values"""
    smoothed = np.copy(values)
    for i in range(1, len(values)):
        smoothed[i] = smooth_factor * smoothed[i-1] + (1 - smooth_factor) * values[i]
    return smoothed


def create_individual_plot(steps, values, smoothed_values, metric_name, color):
    """Create individual plot for a single metric in a new figure"""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, color=color, alpha=0.3, linewidth=1, label='Raw')
    plt.plot(steps, smoothed_values, color=color, linewidth=2, label='Smoothed')
    
    plt.title(f"{format_title(metric_name)}")
    plt.xlabel("Training Steps")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistical information
    add_stats_annotation(values, steps)
    
    plt.tight_layout()


def finalize_combined_plot(all_min, all_max):
    """Finalize the combined plot with all metrics"""
    plt.title("Training Metrics Overview")
    plt.xlabel("Training Steps")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    
    # Add padding to y-axis limits for better visualization
    y_range = all_max - all_min
    plt.ylim(all_min - 0.1 * y_range, all_max + 0.1 * y_range)
    
    # Place legend outside plot for better visibility with many metrics
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()


def add_stats_annotation(values, steps):
    """Add statistical information to the plot"""
    # Calculate statistics
    final_value = values[-1]
    min_value = np.min(values)
    max_value = np.max(values)
    
    # Find step where minimum and maximum values occur
    min_step = steps[np.argmin(values)]
    max_step = steps[np.argmax(values)]
    
    # Calculate improvement (assuming lower is better, common for loss metrics)
    improvement = ((values[0] - final_value) / values[0]) * 100 if values[0] != 0 else 0
    
    # Add text annotation with statistics
    stats_text = (
        f"Final value: {final_value:.4f}\n"
        f"Min: {min_value:.4f} (step {min_step})\n"
        f"Max: {max_value:.4f} (step {max_step})\n"
        f"Change: {improvement:.1f}%"
    )
    
    plt.annotate(
        stats_text,
        xy=(0.02, 0.97),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        va='top'
    )


def format_title(filename):
    """Format filename into a proper title"""
    return ' '.join(word.capitalize() for word in filename.replace('_', ' ').split())


if __name__ == "__main__":
    # Configuration
    CSV_DIRECTORY = "RawCsvData"  # Directory containing CSV files
    SMOOTHING_FACTOR = 0.8  # Smoothing factor for EMA (0-1)
    
    # Run visualization with interactive display
    visualize_training_metrics(CSV_DIRECTORY, SMOOTHING_FACTOR)
    
    print("Interactive visualization complete. Close the windows to exit.")