# COCO Dataset Class Splitter

A Python script that splits a COCO-formatted dataset into separate folders by class, creating binary masks from segmentation annotations.

## Installation

```bash
pip install numpy pillow opencv-python
```

## Overview

This script processes a COCO dataset by:
- Splitting images into class-specific directories
- Creating binary masks from COCO segmentations 
- Moving unlabeled images into a separate 'clean' folder
- Skipping images with multiple different class labels
- Combining multiple masks of the same class
- Providing image count summary per class

## Usage

```bash
python coco_split.py --images /path/to/images/folder --labels /path/to/labels_coco.json [--output /path/to/output]
```

### Arguments
- `--images`: Path to directory containing source images
- `--labels`: Path to COCO format JSON labels file 
- `--output`: Output directory for split dataset (default: 'split_dataset')

### Output Structure
```
output_directory/
    clean/              # Images without labels
        image1.jpg
        image2.jpg
    class1/
        images/        # Original class1 images
            image3.jpg
            image4.jpg
        masks/         # Binary masks for class1
            image3.png
            image4.png
    class2/
        images/
            image5.jpg
        masks/
            image5.png
```

### Binary Masks
- Saved as PNG files
- White pixels (255) = object
- Black pixels (0) = background
- Multiple masks of same class are combined

### Special Cases

The script handles several special cases:

1. **Unlabeled Images**
   - Moved to 'clean' directory
   - Original filename preserved

2. **Multiple Classes**
   - Images with different class labels are skipped
   - Warning message printed

3. **Multiple Masks**
   - Same-class masks combined into single binary mask
   - Named same as source image (with .png extension)

### Example Output

```
Dataset Summary:
----------------------------------------
clean: 45 images
class1: 120 images
class2: 85 images
----------------------------------------
Total images processed: 250
```

### Notes
- Preserves image metadata during copying
- Creates directories automatically
- Validates input paths
- Logs progress during processing

## Example

```bash
python coco_split.py \
    --images ./dataset/images \
    --labels ./dataset/labels_coco.json \
    --output ./split_by_class
```
