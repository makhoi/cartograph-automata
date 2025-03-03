"""
LabelMe to Pascal VOC XML Converter
-----------------------------------
This script converts LabelMe JSON annotation files to Pascal VOC XML format.
It processes both ./test and ./train directories relative to the script location.

Author: Rokawoo
"""

import os
import json
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

def create_pascal_voc_xml(json_data, image_filename):
    """
    Create Pascal VOC format XML from LabelMe JSON data
    
    Args:
        json_data: The JSON data containing annotation information
        image_filename: Filename of the image being annotated
        
    Returns:
        String containing formatted XML in Pascal VOC format
    """
    # Create the root element
    annotation = Element('annotation')
    
    # Add basic image information
    folder = SubElement(annotation, 'folder')
    folder.text = 'images'
    
    filename_elem = SubElement(annotation, 'filename')
    filename_elem.text = image_filename
    
    path = SubElement(annotation, 'path')
    path.text = os.path.join('images', image_filename)
    
    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'
    
    # Extract image size
    size = SubElement(annotation, 'size')
    width_elem = SubElement(size, 'width')
    width_elem.text = str(json_data.get('imageWidth', 0))
    height_elem = SubElement(size, 'height')
    height_elem.text = str(json_data.get('imageHeight', 0))
    depth_elem = SubElement(size, 'depth')
    depth_elem.text = '3'  # Assuming RGB images
    
    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'
    
    # Process shapes (objects)
    if 'shapes' in json_data:
        for shape in json_data['shapes']:
            if shape['shape_type'] == 'rectangle':
                object_elem = SubElement(annotation, 'object')
                
                # Get class name (label)
                name = SubElement(object_elem, 'name')
                name.text = shape['label']
                
                pose = SubElement(object_elem, 'pose')
                pose.text = 'Unspecified'
                
                truncated = SubElement(object_elem, 'truncated')
                truncated.text = '0'
                
                difficult = SubElement(object_elem, 'difficult')
                difficult.text = '0'
                
                # Get bounding box coordinates from the rectangle points
                # LabelMe format has points as [[x1,y1], [x2,y2]]
                bndbox = SubElement(object_elem, 'bndbox')
                
                # Extract coordinates
                if len(shape['points']) >= 2:
                    x_values = [point[0] for point in shape['points']]
                    y_values = [point[1] for point in shape['points']]
                    
                    xmin = SubElement(bndbox, 'xmin')
                    xmin.text = str(int(min(x_values)))
                    ymin = SubElement(bndbox, 'ymin')
                    ymin.text = str(int(min(y_values)))
                    xmax = SubElement(bndbox, 'xmax')
                    xmax.text = str(int(max(x_values)))
                    ymax = SubElement(bndbox, 'ymax')
                    ymax.text = str(int(max(y_values)))
    
    # Convert the ElementTree to a string and return it
    rough_string = tostring(annotation, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def convert_json_to_pascal_voc(json_path, xml_path):
    """
    Convert a LabelMe JSON file to Pascal VOC format XML
    
    Args:
        json_path (str): Path to the JSON file
        xml_path (str): Path to save the XML file
    """
    try:
        # Read JSON file
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        
        # Get image filename from the JSON
        image_filename = json_data.get('imagePath', os.path.basename(json_path).replace('.json', '.jpg'))
        
        # Create Pascal VOC XML
        xml_content = create_pascal_voc_xml(json_data, image_filename)
        
        # Write to file
        with open(xml_path, 'w') as file:
            file.write(xml_content)
            
        return True
    except Exception as e:
        print(f"Error converting {json_path}: {str(e)}")
        return False

def process_directory(input_dir):
    """
    Process all JSON files in a directory and convert them to Pascal VOC XML
    
    Args:
        input_dir (str): Directory containing JSON files
    """
    # Ensure the directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        return
    
    # Count for statistics
    total_files = 0
    successful_conversions = 0
    
    # Process all JSON files
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            total_files += 1
            json_path = os.path.join(input_dir, filename)
            xml_path = os.path.join(input_dir, filename.replace('.json', '.xml'))
            
            print(f"Converting {filename}...")
            success = convert_json_to_pascal_voc(json_path, xml_path)
            
            if success:
                successful_conversions += 1
    
    # Print summary
    print(f"\nConversion complete for {input_dir}!")
    print(f"Total JSON files found: {total_files}")
    print(f"Successfully converted: {successful_conversions}")
    print(f"Failed conversions: {total_files - successful_conversions}")

def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the directories to process
    directories = [
        os.path.join(script_dir, "train"),
        os.path.join(script_dir, "test")
    ]
    
    # Process each directory
    for directory in directories:
        print(f"\nProcessing directory: {directory}")
        process_directory(directory)

if __name__ == "__main__":
    main()