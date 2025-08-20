import json
import os
import shutil
import yaml
from pathlib import Path
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


def load_yml_phrases(file_path):
    """Load phrases from YAML file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        # Split by newlines and filter out empty lines
        phrases = [phrase.strip() for phrase in content.split('\n') if phrase.strip()]
    return phrases


def search_phrases_in_json(json_content, phrases):
    """Search for phrases in JSON content (case-insensitive)."""
    # Convert JSON to string for searching
    json_str = json.dumps(json_content, ensure_ascii=False).lower()
    
    found_phrases = []
    for phrase in phrases:
        # Use word boundaries to match whole words/phrases
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        if re.search(pattern, json_str):
            found_phrases.append(phrase)
    
    return found_phrases


def extract_text_from_json(json_content):
    """Extract all text content from JSON for semantic analysis."""
    text_parts = []
    
    def extract_values(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                extract_values(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_values(item)
        elif isinstance(obj, str):
            text_parts.append(obj.lower())
    
    extract_values(json_content)
    return ' '.join(text_parts)


def semantic_search_phrases(json_content, phrases, model, similarity_threshold=0.6):
    """Search for phrases using semantic similarity."""
    json_text = extract_text_from_json(json_content)
    
    if not json_text.strip():
        return []
    
    # Split JSON text into sentences/chunks for better semantic matching
    sentences = [s.strip() for s in re.split(r'[.!?;]', json_text) if len(s.strip()) > 10]
    if not sentences:
        sentences = [json_text]  # Fallback to full text
    
    try:
        # Get embeddings
        phrase_embeddings = model.encode(phrases)
        sentence_embeddings = model.encode(sentences)
        
        # Calculate similarities
        similarities = cosine_similarity(phrase_embeddings, sentence_embeddings)
        
        found_phrases = []
        for i, phrase in enumerate(phrases):
            max_similarity = np.max(similarities[i])
            if max_similarity >= similarity_threshold:
                found_phrases.append(phrase)
        
        return found_phrases
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []


def load_semantic_model():
    """Load the sentence transformer model for semantic matching."""
    try:
        # Use a lightweight, medical-domain friendly model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded semantic model: all-MiniLM-L6-v2")
        return model
    except Exception as e:
        print(f"Error loading semantic model: {e}")
        return None


def categorize_json_file(json_file_path, early_phrases, late_phrases, model=None, min_matches=2, semantic_threshold=0.6):
    """Categorize a JSON file based on exact and semantic phrase matches.
    
    Exclusion rule: Files containing "USG PELVIS" or "USG_PELVIS" matches are immediately categorized as "unrelated".
    For 'early' categorization: File must contain "early" as text AND have at least min_matches phrase matches.
    For 'late' categorization: File must have at least min_matches phrase matches.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_content = json.load(file)
        
        # Exact matching
        early_exact_matches = search_phrases_in_json(json_content, early_phrases)
        late_exact_matches = search_phrases_in_json(json_content, late_phrases)
        
        # Semantic matching (if model is available)
        early_semantic_matches = []
        late_semantic_matches = []
        
        if model is not None:
            early_semantic_matches = semantic_search_phrases(json_content, early_phrases, model, semantic_threshold)
            late_semantic_matches = semantic_search_phrases(json_content, late_phrases, model, semantic_threshold)
        
        # Combine exact and semantic matches (remove duplicates)
        early_matches = list(set(early_exact_matches + early_semantic_matches))
        late_matches = list(set(late_exact_matches + late_semantic_matches))
        
        # Check for exclusion patterns - immediately categorize as unrelated
        all_matches = early_matches + late_matches
        exclusion_patterns = ["USG PELVIS", "USG_PELVIS"]
        for pattern in exclusion_patterns:
            if any(pattern.lower() in match.lower() for match in all_matches):
                return 'unrelated', [], {'exact': [], 'semantic': []}
        
        # print(f"File: {os.path.basename(json_file_path)}")
        # print(f"  Early exact matches ({len(early_exact_matches)}): {early_exact_matches}")
        # print(f"  Early semantic matches ({len(early_semantic_matches)}): {early_semantic_matches}")
        # print(f"  Late exact matches ({len(late_exact_matches)}): {late_exact_matches}")
        # print(f"  Late semantic matches ({len(late_semantic_matches)}): {late_semantic_matches}")
        # print(f"  Total early matches ({len(early_matches)}): {early_matches}")
        # print(f"  Total late matches ({len(late_matches)}): {late_matches}")
        
        # Check for "early" text match requirement
        has_early_text = False
        json_text = extract_text_from_json(json_content)
        if "early" in json_text.lower():
            has_early_text = True
        
        # Determine category based on matches
        # For early category: must have "early" text match AND at least 2 other matches
        early_qualifies = has_early_text and len(early_matches) >= min_matches
        late_qualifies = len(late_matches) >= min_matches
        
        if early_qualifies and late_qualifies:
            # If both qualify, prefer the one with more matches
            if len(early_matches) > len(late_matches):
                return 'early', early_matches, {'exact': early_exact_matches, 'semantic': early_semantic_matches}
            elif len(late_matches) > len(early_matches):
                return 'late', late_matches, {'exact': late_exact_matches, 'semantic': late_semantic_matches}
            else:
                # If equal matches, prefer early since it has the "early" text requirement
                return 'early', early_matches, {'exact': early_exact_matches, 'semantic': early_semantic_matches}
        elif early_qualifies:
            return 'early', early_matches, {'exact': early_exact_matches, 'semantic': early_semantic_matches}
        elif late_qualifies:
            return 'late', late_matches, {'exact': late_exact_matches, 'semantic': late_semantic_matches}
        else:
            return 'unrelated', [], {'exact': [], 'semantic': []}
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return 'unrelated', [], {'exact': [], 'semantic': []}


def group_json_files(input_dir, output_base_dir, early_yml, late_yml, min_matches=2, use_semantic=True, semantic_threshold=0.6):
    """Group JSON files into early, late, or unrelated categories using exact and semantic matching."""
    
    # Load phrases from YAML files
    early_phrases = load_yml_phrases(early_yml)
    late_phrases = load_yml_phrases(late_yml)
    
    print(f"Loaded {len(early_phrases)} early phrases")
    print(f"Loaded {len(late_phrases)} late phrases")
    print(f"Minimum matches required: {min_matches}")
    
    # Load semantic model
    model = None
    if use_semantic:
        print("Loading semantic model...")
        model = load_semantic_model()
        if model:
            print(f"Semantic similarity threshold: {semantic_threshold}")
        else:
            print("Semantic model failed to load, using exact matching only")
    else:
        print("Using exact matching only")
    
    print("-" * 50)
    
    # Create output directories
    early_dir = os.path.join(output_base_dir, 'early')
    late_dir = os.path.join(output_base_dir, 'late')
    unrelated_dir = os.path.join(output_base_dir, 'unrelated')
    
    os.makedirs(early_dir, exist_ok=True)
    os.makedirs(late_dir, exist_ok=True)
    os.makedirs(unrelated_dir, exist_ok=True)
    
    # Get all JSON files from input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Statistics
    stats = {'early': 0, 'late': 0, 'unrelated': 0}
    semantic_stats = {'early_exact': 0, 'early_semantic': 0, 'late_exact': 0, 'late_semantic': 0}
    
    # Process each JSON file
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        category, matches, match_details = categorize_json_file(
            json_path, early_phrases, late_phrases, model, min_matches, semantic_threshold
        )
        
        # Check if file has "early" text for debug output
        json_text_contains_early = False
        if category == 'early':
            try:
                with open(json_path, 'r', encoding='utf-8') as file:
                    json_content = json.load(file)
                json_text = extract_text_from_json(json_content)
                json_text_contains_early = "early" in json_text.lower()
            except:
                pass
        
        # Update semantic statistics
        if category == 'early':
            semantic_stats['early_exact'] += len(match_details['exact'])
            semantic_stats['early_semantic'] += len(match_details['semantic'])
        elif category == 'late':
            semantic_stats['late_exact'] += len(match_details['exact'])
            semantic_stats['late_semantic'] += len(match_details['semantic'])
        
        # Copy file to appropriate directory
        if category == 'early':
            dest_path = os.path.join(early_dir, json_file)
        elif category == 'late':
            dest_path = os.path.join(late_dir, json_file)
        else:
            dest_path = os.path.join(unrelated_dir, json_file)
        
        shutil.copy2(json_path, dest_path)
        stats[category] += 1
        
        print(f"File: {os.path.basename(json_file)} -> {category}")
        if category == 'early':
            print(f"  Has 'early' text: {json_text_contains_early}")
        if match_details['exact']:
            print(f"  Exact matches: {match_details['exact']}")
        if match_details['semantic']:
            print(f"  Semantic matches: {match_details['semantic']}")
        print()
    
    # Print summary
    print("=" * 50)
    print("SUMMARY:")
    print(f"Total files processed: {len(json_files)}")
    print(f"Early: {stats['early']} files")
    print(f"Late: {stats['late']} files")
    print(f"Unrelated: {stats['unrelated']} files")
    
    if model is not None:
        print("\nSemantic Matching Statistics:")
        print(f"Early category - Exact matches: {semantic_stats['early_exact']}, Semantic matches: {semantic_stats['early_semantic']}")
        print(f"Late category - Exact matches: {semantic_stats['late_exact']}, Semantic matches: {semantic_stats['late_semantic']}")
    
    print("=" * 50)


def main():
    """Main function to run the grouping process."""
    # Define paths
    input_dir = '/home/med_data/<user>/rule_based/all_output_phi4_sampling'
    output_base_dir = '/home/med_data/<user>/rule_based/grouped_output_semantic'
    early_yml = '/home/med_data/<user>/rule_based/early.yml'
    late_yml = '/home/med_data/<user>/rule_based/late.yml'
    
    # Set parameters (you can adjust these values)
    min_matches = 2
    use_semantic = True  # Set to False to use exact matching only
    semantic_threshold = 0.6  # Similarity threshold for semantic matching
    
    print(f"Grouping JSON files from: {input_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Early phrases file: {early_yml}")
    print(f"Late phrases file: {late_yml}")
    print()
    
    # Run the grouping process
    group_json_files(input_dir, output_base_dir, early_yml, late_yml, min_matches, use_semantic, semantic_threshold)


if __name__ == "__main__":
    main()