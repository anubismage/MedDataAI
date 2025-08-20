# JSON File Clustering System

## Overview
This clustering system processes JSON files from the `all_output_phi4_sampling` directory and categorizes them into Early, Late, and Unrelated clusters based on medical terminology patterns found in `early.yml` and `late.yml` files.

## Features

### 1. **Dual Matching Strategy**
- **Regex Matching**: Exact word/phrase matching using regular expressions with word boundaries
- **Semantic Matching**: Uses sentence-transformers model (all-MiniLM-L6-v2) for similarity-based matching

### 2. **Three Primary Clusters**
- **Early**: Files containing early pregnancy terminology (gestational_sac, yolk_sac, fetal_pole, etc.)
- **Late**: Files containing late pregnancy terminology (presentation, position, placenta, doppler, etc.)
- **Unrelated**: Files that don't meet the minimum threshold for either category

### 3. **Advanced Clustering**
- Optional K-means clustering using feature vectors based on match counts
- Creates additional cluster directories for more granular categorization

## Configuration Parameters

### Key Settings (in `clustering.py` main function):
```python
min_matches = 2                    # Minimum matches required for categorization
use_semantic = True                # Enable/disable semantic matching  
semantic_threshold = 0.6           # Similarity threshold for semantic matches
use_advanced_clustering = True     # Enable K-means clustering
```

## Input Files

### `early.yml` (14 phrases):
```
decidual_reaction, sub_chorionic_hemorrhage, chorionic_hemorrhage, 
uterus, adnexa, gestational_sac, gestational_age, gestational_age_by_crl_length,
fetal_pole, yolk_sac, crl, fetal, pregnancy, embryo
```

### `late.yml` (20 phrases):
```
internal_os, uterine_contractions, adnexal_findings, fetal_biometry,
fetal_anatomy, fetal_presentation_findings, pregnancy_details,
biophysical_profile, presentation, lie, position, placenta,
umbilical_cord, doppler, doppler_ultrasound, uterine_artery,
umbilical_artery, middle_cerebral_artery, cereboplacental_ratio, pregnancy
```

## Results Summary

### Processing Results:
- **Total files processed**: 295
- **Early category**: 180 files (61.0%)
- **Late category**: 40 files (13.6%)
- **Unrelated category**: 75 files (25.4%)

### Matching Effectiveness:
- **Exact matches**: 859 total (727 early + 132 late)
- **Semantic matches**: 134 total (118 early + 16 late)
- **Semantic improvement**: 15.6% additional matches over exact matching alone

### Advanced Clustering:
- **Cluster 0**: 95 files (mixed/low match files)
- **Cluster 1**: 149 files (high early pregnancy matches)
- **Cluster 2**: 51 files (high late pregnancy matches)

## Output Structure

```
clustered_output/
├── early/              # 180 files - Early pregnancy cases
├── late/               # 40 files - Late pregnancy cases  
├── unrelated/          # 75 files - Unrelated cases
├── cluster_0/          # 95 files - Advanced clustering
├── cluster_1/          # 149 files - Advanced clustering
├── cluster_2/          # 51 files - Advanced clustering
└── clustering_results.json  # Detailed results and metadata
```

## Algorithm Logic

### 1. **Feature Extraction**
For each JSON file, the system:
- Converts JSON to text for analysis
- Searches for exact phrase matches using regex
- Calculates semantic similarity using embeddings
- Creates feature vectors with match counts

### 2. **Primary Categorization**
```python
if early_matches >= min_matches and late_matches >= min_matches:
    # Choose category with more matches
    category = "early" if early_matches > late_matches else "late"
elif early_matches >= min_matches:
    category = "early"  
elif late_matches >= min_matches:
    category = "late"
else:
    category = "unrelated"
```

### 3. **Advanced Clustering**
- Uses K-means clustering on 6-dimensional feature vectors:
  - Early exact count, Early semantic count
  - Late exact count, Late semantic count  
  - Early total count, Late total count
- Creates 3 clusters with distinct characteristics

## Usage

### Run Clustering:
```bash
cd /home/med_data/<user>/rule_based
/home/med_data/<user>/.venv/bin/python clustering.py
```

### Test Results:
```bash
/home/med_data/<user>/.venv/bin/python test_clustering.py
```

## Dependencies
- `sentence-transformers`: For semantic similarity matching
- `scikit-learn`: For K-means clustering and similarity calculations
- `numpy`: For numerical operations
- `pyyaml`: For loading phrase files

## Customization

### To modify clustering behavior:
1. **Adjust thresholds**: Change `min_matches` or `semantic_threshold`
2. **Add/remove phrases**: Edit `early.yml` and `late.yml` files
3. **Change semantic model**: Modify model in `load_semantic_model()`
4. **Adjust cluster count**: Change `n_clusters` parameter in advanced clustering

The system provides both traditional rule-based categorization and modern semantic matching, making it robust for medical document classification tasks.
