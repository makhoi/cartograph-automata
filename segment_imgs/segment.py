import json
import pandas as pd
import numpy as np

# Load JSON file
json_file = "seq47_left_20250228-043027.json"  # Replace with your JSON file
with open(json_file, "r") as file:
    data = json.load(file)

# Get image dimensions (fallback to defaults if not available)
image_width = data.get("imageWidth", 1024)
image_height = data.get("imageHeight", 768)

# Extract labeled objects
shapes = data.get("shapes", [])

# Initialize lists for storing object coordinates
persons = []
shelves = []
lines = []

# Process each labeled object
for shape in shapes:
    label = shape["label"].lower()  # Convert to lowercase for consistency
    points = shape["points"]

    # Convert polygon points to bounding box (min_x, min_y, max_x, max_y)
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    # Categorize objects
    if label == "person":
        persons.append(bbox)
    elif label == "shelf":
        shelves.append(bbox)
    elif label == "line":
        lines.append(bbox)

# Create an empty obstacle map
obstacle_map = np.zeros((image_height, image_width), dtype=np.uint8)

# Mark persons and shelves as obstacles
for (x1, y1, x2, y2) in persons + shelves:
    x1, x2 = int(x1), int(x2)
    y1, y2 = int(y1), int(y2)
    obstacle_map[y1:y2, x1:x2] = 255  # Mark as obstacle

# Extract the original path along the line
line_path = []
for (x1, y1, x2, y2) in lines:
    for x in range(int(x1), int(x2) + 1, 10):  # Sample points along the line
        mid_y = int((y1 + y2) / 2)
        line_path.append((x, mid_y))

# Function to find the safest detour around obstacles
def find_safe_detour(start_x, start_y, obstacle_map, step_size=10, max_shift=100):
    """
    Finds a safe detour by scanning left and right for free space.
    Returns the best (x, y) detour point.
    """
    best_x = start_x
    best_space = 0

    for shift in range(10, max_shift + 10, step_size):
        left_x = max(start_x - shift, 0)
        right_x = min(start_x + shift, image_width - 1)

        # If both left and right are free, choose the wider one
        if obstacle_map[start_y, left_x] == 0 and obstacle_map[start_y, right_x] == 0:
            return (right_x if right_x - start_x > start_x - left_x else left_x, start_y)

        # Check individual spaces and choose the widest one
        if obstacle_map[start_y, left_x] == 0 and start_x - left_x > best_space:
            best_x, best_space = left_x, start_x - left_x

        if obstacle_map[start_y, right_x] == 0 and right_x - start_x > best_space:
            best_x, best_space = right_x, right_x - start_x

    return (best_x, start_y)  # Return the best available detour point

# Compute the adjusted path
safe_path = []
avoiding = False

for x, y in line_path:
    if obstacle_map[y, x] == 255:  # Obstacle detected on the line
        avoiding = True
        detour_x, detour_y = find_safe_detour(x, y, obstacle_map)
        safe_path.append((detour_x, detour_y))
    else:
        if avoiding:  # If we were avoiding, check if we can return to the line
            avoiding = False
        safe_path.append((x, y))

# Convert safe path to a DataFrame and print coordinates
df_safe_path = pd.DataFrame(safe_path, columns=["X", "Y"])

# Save safe path coordinates as CSV
df_safe_path.to_csv("safe_path_coordinates.csv", index=False)

# Print output
print("\nSafe Path Coordinates:")
print(df_safe_path)

print("\nSafe path coordinates saved to 'safe_path_coordinates.csv'")
