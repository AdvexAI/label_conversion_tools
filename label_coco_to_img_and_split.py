import json
import os
import shutil
import numpy as np
from PIL import Image
import cv2
import argparse
from collections import defaultdict

def create_binary_mask(segmentation, height, width):
    """Create a binary mask from COCO segmentation."""
    mask = np.zeros((height, width), dtype=np.uint8)
    # Convert segmentation to numpy array of polygon vertices
    poly = np.array(segmentation, dtype=np.int32).reshape((-1, 2))
    # Fill polygon with ones
    cv2.fillPoly(mask, [poly], 1)
    return mask

def process_coco_dataset(json_path, images_dir, output_base_dir):
    # Dictionary to keep track of images per class
    class_counts = defaultdict(int)
    
    # Read COCO JSON file
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    image_id_to_size = {img['id']: (img['height'], img['width']) for img in coco_data['images']}
    
    # Map category IDs to names
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by image_id
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Create clean directory for images without labels
    clean_dir = os.path.join(output_base_dir, 'clean')
    os.makedirs(clean_dir, exist_ok=True)
    
    # Find images without annotations and move them to clean directory
    for img in coco_data['images']:
        if img['id'] not in image_annotations:
            image_filename = img['file_name']
            src_image_path = os.path.join(images_dir, image_filename)
            dst_image_path = os.path.join(clean_dir, image_filename)
            shutil.copy2(src_image_path, dst_image_path)
            class_counts['clean'] += 1
            print(f"Moved unlabeled image {image_filename} to clean directory")
    
    # Process each image with annotations
    for image_id, annotations in image_annotations.items():
        # Get unique category IDs for this image
        category_ids = set(ann['category_id'] for ann in annotations)
        
        # Skip images with multiple different classes
        if len(category_ids) > 1:
            print(f"Skipping image {image_id_to_file[image_id]} - multiple classes detected")
            continue
        
        category_id = category_ids.pop()  # Get the single category ID
        category_name = category_map[category_id]
        
        # Create directory structure
        class_dir = os.path.join(output_base_dir, category_name)
        images_dir_dest = os.path.join(class_dir, 'images')
        masks_dir = os.path.join(class_dir, 'masks')
        
        os.makedirs(images_dir_dest, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        # Copy image
        image_filename = image_id_to_file[image_id]
        src_image_path = os.path.join(images_dir, image_filename)
        dst_image_path = os.path.join(images_dir_dest, image_filename)
        shutil.copy2(src_image_path, dst_image_path)
        
        # Increment counter for this class
        class_counts[category_name] += 1
        
        # Create and save masks
        height, width = image_id_to_size[image_id]
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        for ann in annotations:
            for segmentation in ann['segmentation']:
                mask = create_binary_mask(segmentation, height, width)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Save mask with same filename as image but with .png extension
        mask_filename = os.path.splitext(image_filename)[0] + '.png'
        mask_path = os.path.join(masks_dir, mask_filename)
        cv2.imwrite(mask_path, combined_mask * 255)  # Multiply by 255 for proper binary image
        
        print(f"Processed {image_filename} for class {category_name}")
    
    # Print summary
    print("\nDataset Summary:")
    print("-" * 40)
    total_images = 0
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count} images")
        total_images += count
    print("-" * 40)
    print(f"Total images processed: {total_images}")
    
    return class_counts

def main():
    parser = argparse.ArgumentParser(description='Process COCO dataset and split by class')
    parser.add_argument('--images', required=True, help='Path to the directory containing images')
    parser.add_argument('--labels', required=True, help='Path to the COCO labels JSON file')
    parser.add_argument('--output', default='split_dataset', help='Output directory for the split dataset')
    
    args = parser.parse_args()
    
    # Verify paths exist
    if not os.path.exists(args.images):
        raise FileNotFoundError(f"Images directory not found: {args.images}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    process_coco_dataset(args.labels, args.images, args.output)

if __name__ == "__main__":
    main()
