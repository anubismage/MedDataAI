#!/usr/bin/env python3

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

def copy_random_audio_files(source_dir, dest_dir, num_files=17, audio_extensions=None):
    """
    Copy a specified number of random audio files from source directory to destination directory.
    
    Args:
        source_dir (str): Source directory containing audio files
        dest_dir (str): Destination directory where files will be copied to
        num_files (int): Number of files to copy (default: 17)
        audio_extensions (list): List of audio file extensions to consider
                                 (default: ['.wav', '.mp3', '.flac', '.aac', '.ogg'])
    """
    if audio_extensions is None:
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg']

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all audio files from source directory
    audio_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"No audio files found in {source_dir}")
        return

    # Select random files
    num_to_copy = min(num_files, len(audio_files))
    selected_files = random.sample(audio_files, num_to_copy)
    
    print(f"Found {len(audio_files)} audio files. Copying {num_to_copy} files to {dest_dir}...")
    
    # Copy the selected files
    for file in selected_files:
        filename = os.path.basename(file)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(file, dest_path)
        print(f"Copied: {filename}")

    print(f"Successfully copied {num_to_copy} audio files to {dest_dir}")

def main():
    parser = argparse.ArgumentParser(description='Copy random audio files from source to destination.')
    parser.add_argument('source_dir', help='Source directory containing audio files')
    parser.add_argument('dest_dir', help='Destination directory where files will be copied to')
    parser.add_argument('-n', '--num_files', type=int, default=17, help='Number of files to copy (default: 17)')
    parser.add_argument('-e', '--extensions', nargs='+', 
                        default=['.wav', '.mp3', '.flac', '.aac', '.ogg','.m4a'],
                        help='Audio file extensions to consider')
    
    args = parser.parse_args()
    
    copy_random_audio_files(args.source_dir, args.dest_dir, args.num_files, args.extensions)

if __name__ == "__main__":
    main()