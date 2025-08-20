#!/usr/bin/env python3
"""
Script to compare the results between exact matching and semantic matching.
"""

import os
import json

def compare_grouping_results():
    """Compare the results between exact and semantic matching."""
    
    # Define directories
    exact_dir = '/home/med_data/<user>/rule_based/grouped_output'
    semantic_dir = '/home/med_data/<user>/rule_based/grouped_output_semantic'
    
    categories = ['early', 'late', 'unrelated']
    
    print("=" * 60)
    print("COMPARISON: EXACT vs SEMANTIC MATCHING")
    print("=" * 60)
    
    print(f"{'Category':<15} {'Exact Match':<12} {'Semantic Match':<15} {'Difference':<12}")
    print("-" * 60)
    
    total_improvement = 0
    
    for category in categories:
        exact_files = set(os.listdir(os.path.join(exact_dir, category)))
        semantic_files = set(os.listdir(os.path.join(semantic_dir, category)))
        
        exact_count = len(exact_files)
        semantic_count = len(semantic_files)
        difference = semantic_count - exact_count
        
        print(f"{category.capitalize():<15} {exact_count:<12} {semantic_count:<15} {difference:+d}")
        
        if category != 'unrelated':
            total_improvement += max(0, difference)
    
    # Calculate files moved out of unrelated
    exact_unrelated = set(os.listdir(os.path.join(exact_dir, 'unrelated')))
    semantic_unrelated = set(os.listdir(os.path.join(semantic_dir, 'unrelated')))
    
    files_moved_from_unrelated = exact_unrelated - semantic_unrelated
    
    print("-" * 60)
    print(f"Files moved from 'unrelated' to categorized: {len(files_moved_from_unrelated)}")
    print(f"Overall categorization improvement: {len(files_moved_from_unrelated)} files")
    print("=" * 60)
    
    # Show some examples of files that were reclassified
    print("\nEXAMPLES OF RECLASSIFIED FILES:")
    print("-" * 40)
    
    for category in ['early', 'late']:
        exact_files = set(os.listdir(os.path.join(exact_dir, category)))
        semantic_files = set(os.listdir(os.path.join(semantic_dir, category)))
        
        new_files = semantic_files - exact_files
        if new_files:
            print(f"\nFiles newly classified as '{category}' with semantic matching:")
            for i, file in enumerate(sorted(list(new_files))[:5]):  # Show first 5
                print(f"  {i+1}. {file}")
            if len(new_files) > 5:
                print(f"  ... and {len(new_files) - 5} more files")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    compare_grouping_results()
