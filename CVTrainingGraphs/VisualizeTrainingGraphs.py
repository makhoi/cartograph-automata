"""
TensorFlow Training Metrics Visualizer
--------------------------------------
This script creates a single window with scrollable, uniformly-sized visualizations
of TensorFlow training metrics from CSV files in the specified directory.

Author: Rokawoo
Date: March 2, 2025
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from pathlib import Path


def visualize_training_metrics(csv_dir, smooth_factor=0.8):
    """
    Creates a single window with all training metrics visualizations.
    
    Args:
        csv_dir (str): Directory containing CSV files
        smooth_factor (float): Factor for exponential moving average smoothing (0-1)
    """
    # Get all CSV files in the directory
    csv_files = list(Path(csv_dir).glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    # Calculate layout dimensions - one additional row for the combined view
    num_metrics = len(csv_files)
    num_rows = num_metrics + 1
    
    # Create a tall figure with fixed-width subplots
    # Height is calculated based on number of plots (x inches per plot)
    fig = plt.figure(figsize=(12, 7 * num_rows), constrained_layout=True)
    
    # Create grid layout
    gs = GridSpec(num_rows, 1, figure=fig)
    
    # Color palette for consistent colors
    colors = plt.cm.tab10.colors
    
    # Store all data for combined plot
    all_data = []
    all_min, all_max = float('inf'), float('-inf')
    
    # Process each CSV file for individual plots
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
            
            # Update min/max for scaling
            all_min = min(all_min, np.min(values))
            all_max = max(all_max, np.max(values))
            
            # Create individual subplot
            ax = fig.add_subplot(gs[idx + 1, 0])  # +1 because first row is for combined plot
            
            # Plot individual metric
            color = colors[idx % len(colors)]
            ax.plot(steps, values, color=color, alpha=0.3, linewidth=1, label='Raw')
            ax.plot(steps, smoothed_values, color=color, linewidth=2, label='Smoothed')
            
            # Add annotations and styling
            ax.set_title(format_title(metric_name), fontsize=12, x=0.01, y=0.95)
            ax.set_xlabel("", labelpad=-1)
            ax.text(1, -0.25, "Training Steps", fontsize=10, transform=ax.transAxes, ha='right')
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Add statistical information
            add_stats_annotation(ax, values, steps)
            
            # Store data for combined plot
            all_data.append({
                'steps': steps,
                'values': values,
                'smoothed_values': smoothed_values,
                'metric_name': metric_name,
                'color': color
            })
            
        except Exception as e:
            print(f"Error processing {csv_path}: {str(e)}")
    
    # Create combined plot at the top
    ax_combined = fig.add_subplot(gs[0, 0])
    create_combined_plot(ax_combined, all_data, all_min, all_max)
    
    # Add overall title to the figure
    fig.suptitle("TensorFlow Training Metrics Visualization", fontsize=16, x=0.1, y=1)
    
    # Try to set window to fullscreen in a cross-platform way
    try:
        figManager = plt.get_current_fig_manager()
        
        # Different backends have different methods for fullscreen
        if hasattr(figManager, 'window'):
            if hasattr(figManager.window, 'showMaximized'):  # Qt backend
                figManager.window.showMaximized()
            elif hasattr(figManager.window, 'state'):  # Tk backend
                figManager.window.state('zoomed')  # Windows
                # figManager.window.attributes('-zoomed', True)  # Linux
        elif hasattr(figManager, 'full_screen_toggle'):  # Newer matplotlib versions
            figManager.full_screen_toggle()
        elif hasattr(figManager, 'frame'):  # wxPython
            figManager.frame.Maximize(True)
        
    except Exception as e:
        print(f"Note: Could not set fullscreen mode. This doesn't affect functionality.")
    
    # Show the complete figure
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust for suptitle
    plt.subplots_adjust(hspace=0.4)  # Add more space between subplots
    plt.show()


def apply_smoothing(values, smooth_factor):
    """Apply exponential moving average smoothing to values"""
    smoothed = np.copy(values)
    for i in range(1, len(values)):
        smoothed[i] = smooth_factor * smoothed[i-1] + (1 - smooth_factor) * values[i]
    return smoothed


def create_combined_plot(ax, all_data, all_min, all_max):
    """Create the combined overview plot on the provided axis"""
    for data in all_data:
        ax.plot(
            data['steps'], 
            data['values'], 
            color=data['color'], 
            alpha=0.2, 
            linewidth=1
        )
        ax.plot(
            data['steps'], 
            data['smoothed_values'], 
            color=data['color'], 
            linewidth=2, 
            label=data['metric_name']
        )
    
    # Style the combined plot
    ax.set_title("Combined Metrics Overview", fontsize=12)
    ax.set_xlabel("", labelpad=-1)
    ax.text(1, -0.25, "Training Steps", fontsize=10, transform=ax.transAxes, ha='right')
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    
    # Add padding to y-axis limits for better visualization
    y_range = all_max - all_min
    ax.set_ylim(all_min - 0.1 * y_range, all_max + 0.1 * y_range)
    
    # Place legend outside plot for better visibility with many metrics
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


def add_stats_annotation(ax, values, steps):
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
    
    ax.annotate(
        stats_text,
        xy=(1.06, 0.91),
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
    
    # Run visualization
    visualize_training_metrics(CSV_DIRECTORY, SMOOTHING_FACTOR)
    
    print("Visualization complete. Close the window to exit.")