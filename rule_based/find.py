#!/usr/bin/env python3
"""
Script to organize image files from /var/share/sdata into the same folder structure
as their corresponding JSON files in grouped_manual (early, late, unrelated folders).

The script matches image files like:
  00000020-PHOTO-2024-11-19-07-54-53.jpg
with JSON files like:
  00000020-PHOTO-2024-11-19-07-54-53_OCR2_phi-4_sampling.json
"""

import os
import shutil
import glob
from pathlib import Path


def extract_base_filename(filename):
    """Extract the base filename without extension and OCR suffix."""
    # Remove extension
    base = os.path.splitext(filename)[0]
    # Remove _OCR2_phi-4_sampling suffix if present
    if base.endswith('_OCR2_phi-4_sampling'):
        base = base.replace('_OCR2_phi-4_sampling', '')
    return base


def find_matching_json_folder(base_filename, grouped_manual_path):
    """Find which folder (early, late, unrelated) contains the matching JSON file."""
    folders = ['early', 'late', 'unrelated']
    
    for folder in folders:
        folder_path = os.path.join(grouped_manual_path, folder)
        json_filename = f"{base_filename}_OCR2_phi-4_sampling.json"
        json_path = os.path.join(folder_path, json_filename)
        
        if os.path.exists(json_path):
            return folder
    
    return None


def organize_images():
    """Main function to organize image files."""
    source_dir = "/var/share/sdata"
    json_base_dir = "/home/med_data/<user>/rule_based/manual/grouped_manual"
    target_base_dir = "/home/med_data/<user>/rule_based/manual/grouped_images"
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist!")
        return
    
    # Check if JSON directory exists
    if not os.path.exists(json_base_dir):
        print(f"Error: JSON directory {json_base_dir} does not exist!")
        return
    
    # Create target directory structure if it doesn't exist
    os.makedirs(target_base_dir, exist_ok=True)
    for folder in ['early', 'late', 'unrelated']:
        os.makedirs(os.path.join(target_base_dir, folder), exist_ok=True)
    
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, ext)))
    
    print(f"Found {len(image_files)} image files in {source_dir}")
    
    # Statistics
    copied_count = 0
    not_found_count = 0
    error_count = 0
    
    # Process each image file
    for image_path in image_files:
        try:
            image_filename = os.path.basename(image_path)
            base_filename = extract_base_filename(image_filename)
            
            # Find the matching folder
            target_folder = find_matching_json_folder(base_filename, json_base_dir)
            
            if target_folder:
                # Create target path in grouped_images
                target_dir = os.path.join(target_base_dir, target_folder)
                target_path = os.path.join(target_dir, image_filename)
                
                # Copy the file
                if not os.path.exists(target_path):
                    shutil.copy2(image_path, target_path)
                    print(f"Copied: {image_filename} -> {target_folder}/")
                    copied_count += 1
                else:
                    print(f"Skipped (already exists): {image_filename} -> {target_folder}/")
                    copied_count += 1
            else:
                print(f"No matching JSON found for: {image_filename}")
                not_found_count += 1
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            error_count += 1
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Total image files processed: {len(image_files)}")
    print(f"Successfully copied/found: {copied_count}")
    print(f"No matching JSON found: {not_found_count}")
    print(f"Errors: {error_count}")
    print("="*50)


if __name__ == "__main__":
    organize_images()