# MedDataAI - Medical Audio Transcription & Analysis Project

This repository contains a comprehensive medical audio transcription and analysis system designed for sonography audio processing. The project evaluates multiple Automatic Speech Recognition (ASR) models for medical transcription accuracy and includes rule-based clustering systems for pregnancy stage classification.

## 📋 Project Overview

The MedDataAI project focuses on:

- **Medical Audio Transcription**: Evaluating ASR models for sonography audio
- **Performance Analysis**: Comprehensive metrics evaluation (WER, CER, Cosine Similarity, Medical Term Accuracy)
- **Medical Classification**: Rule-based clustering for pregnancy stage classification
- **Model Comparison**: Testing multiple state-of-the-art ASR models

## 🏗️ Repository Structure

```text
MedDataAI/
├── 📊 AudioAnalysis/              # ASR Model Evaluation & Analysis
├── 🤖 transcribe scripts/         # Audio Transcription Tools
├── 📋 rule_based/                 # Medical Classification System
├── 🔧 golden.py                   # Gold standard utilities
├── 🌳 json_node_tree_analyzer.py  # JSON analysis tools
└── 📖 PROJECT_STATUS_REPORT.md    # Project status documentation
```

## 📊 Audio Analysis (`AudioAnalysis/`)

### Core Components

- **`ASR_Evaluation.ipynb`**: Main evaluation notebook with comprehensive ASR model analysis
- **`ASR_Evaluation_Report.md`**: Detailed evaluation report with findings and recommendations
- **`ASR_Evaluation_Report.pdf`**: PDF version of the evaluation report
- **`MetricSelection.py`**: Python script for generating evaluation metrics

### Key Findings

**Best Performing Model**: **Whisper-Large-v3**

- **Medical Term Accuracy**: 85%
- **Word Error Rate**: 0.668 ± 0.659
- **Character Error Rate**: 0.385 ± 0.421
- **Cosine Similarity**: 0.843 ± 0.117

### Models Evaluated

1. **Whisper-Large-v3** - Transformer-based, best overall performance
2. **parakeet-rnnt-1.1b** - RNN-Transducer, good for real-time applications
3. **vosk-model-en-in-0.5** - Lightweight, resource-constrained environments
4. **wav2vec2-base-960h** - Self-supervised, general speech recognition

### Evaluation Metrics

- **Word Error Rate (WER)**: Word-level transcription accuracy
- **Character Error Rate (CER)**: Character-level transcription accuracy
- **Cosine Similarity**: Semantic similarity preservation
- **Medical Term Accuracy**: Domain-specific medical vocabulary recognition

### Output Data (`output/`)

- `metrics_summary.csv`: Aggregated performance metrics
- `sample_metrics.csv`: Per-sample detailed metrics
- `medical_term_accuracy.csv`: Medical terminology recognition accuracy

### Visualizations (`figures/`)

- Performance comparison charts (WER, CER, Cosine Similarity)
- Distribution boxplots for each metric
- Medical terminology recognition analysis
- Correlation analysis between metrics

## 🤖 Transcription Scripts (`transcribe scripts/`)

### Core Transcription Tools

- **`transcribe_audio.py`**: Batch processing script for multiple ASR models
- **`transcribe_whisper.py`**: Whisper-specific transcription implementation
- **`transcribe_parakeet.py`**: Parakeet model transcription
- **`transcribe_vosk.py`**: VOSK model transcription

### Experimental Scripts

- **`transcribe_phi4_manual.py`**: Manual Phi-4 multimodal implementation
- **`transcribe_phi4_fixed.py`**: Fixed version of Phi-4 transcription
- **`transcribe_phi4-multimodal-instruct.py`**: Phi-4 multimodal instruction-based
- **`transcribe_voxtral-mini3b.py`**: VoxTral model implementation

### Features

- **Multi-format Support**: MP3, WAV, FLAC, M4A, AAC, OGG, WMA, AIFF, AU, OPUS, WebM
- **Audio Preprocessing**: Automatic format conversion, stereo to mono, 16kHz normalization
- **High-pass Filtering**: 500Hz Butterworth filter for noise reduction
- **Batch Processing**: Efficient processing of multiple audio files
- **Error Handling**: Robust fallback mechanisms

### Debug Tools (`debug/`)

Comprehensive debugging utilities for troubleshooting transcription issues:

- Model structure analysis
- Audio input validation
- Template debugging
- Performance optimization

## 📋 Rule-Based Classification (`rule_based/`)

### Medical Classification System

- **`clustering.py`**: Main clustering algorithm for pregnancy stage classification
- **`CLUSTERING_DOCUMENTATION.md`**: Detailed system documentation
- **`compare_results.py`**: Results comparison utilities
- **`find.py`** & **`group.py`**: Search and grouping utilities

### Classification Categories

- **Early Pregnancy**: gestational_sac, yolk_sac, fetal_pole, etc.
- **Late Pregnancy**: presentation, position, placenta, doppler, etc.
- **Unrelated**: Files not meeting minimum thresholds

### Configuration Files

- **`early.yml`**: Early pregnancy terminology patterns
- **`late.yml`**: Late pregnancy terminology patterns

### Features

- **Dual Matching Strategy**: Regex + Semantic matching
- **Advanced Clustering**: Optional K-means clustering
- **Configurable Thresholds**: Adjustable matching parameters
- **Semantic Analysis**: sentence-transformers integration

### Key Parameters

```python
min_matches = 2                    # Minimum matches for categorization
semantic_threshold = 0.6           # Similarity threshold
use_semantic = True                # Enable semantic matching
use_advanced_clustering = True     # Enable K-means clustering
```

## 🔧 Utility Scripts

### `golden.py`

Gold standard utilities for creating and managing reference transcriptions.

### `json_node_tree_analyzer.py`

JSON analysis tools for processing structured medical data.

## 📈 Project Status

### ✅ Completed Features

1. **Comprehensive ASR Evaluation**: 4 models tested with multiple metrics
2. **Medical Domain Analysis**: Specialized medical terminology evaluation
3. **Visualization Suite**: Performance charts and distribution analysis
4. **Rule-based Classification**: Pregnancy stage categorization system
5. **Audio Preprocessing Pipeline**: Robust audio handling with multiple fallbacks

### ⚠️ Known Issues

- **Phi-4 Multimodal Compatibility**: PEFT library compatibility issues
- **Performance Variability**: Some models show high variance across samples
- **Resource Requirements**: High-performance models require significant GPU resources

### 🎯 Current Focus

- Model optimization for production deployment
- Real-time transcription capabilities
- Quality assurance workflows for clinical applications

## 🚀 Quick Start

### 1. Audio Transcription

```bash
# Batch transcribe using Whisper (recommended)
python "transcribe scripts/transcribe_audio.py" --input /path/to/audio --output /path/to/output --model openai/whisper-large-v3

# Single file transcription
python "transcribe scripts/transcribe_whisper.py" --audio file.mp3 --output transcription.txt
```

### 2. Performance Evaluation

```bash
# Generate evaluation metrics
python AudioAnalysis/MetricSelection.py

# View results in Jupyter notebook
jupyter notebook AudioAnalysis/ASR_Evaluation.ipynb
```

### 3. Medical Classification

```bash
# Run pregnancy stage clustering
python rule_based/clustering.py
```

## 📊 Results Summary

### ASR Model Rankings

1. **Whisper-Large-v3** (Rank 1 across all metrics)
2. **parakeet-rnnt-1.1b** (Consistent second place)
3. **vosk-model-en-in-0.5** (Third place)
4. **wav2vec2-base-960h** (Fourth place)

### Recommendations

- **Production Use**: Whisper-Large-v3 for highest accuracy
- **Real-time Applications**: parakeet-rnnt-1.1b for streaming
- **Resource-Constrained**: vosk for lightweight deployments

## 🔬 Technical Requirements

### Computational Resources

- **Whisper-Large-v3**: 8GB+ GPU memory recommended
- **parakeet-rnnt-1.1b**: Moderate GPU resources
- **vosk**: CPU-only operation possible
- **Audio Processing**: FFmpeg support for format conversion

### Dependencies

- Python 3.8+
- PyTorch, transformers, whisper
- librosa, soundfile, pydub
- sentence-transformers for semantic analysis
- scikit-learn for clustering
- pandas, numpy for data processing

## 📄 Documentation

- **`ASR_Evaluation_Report.md`**: Comprehensive evaluation findings
- **`CLUSTERING_DOCUMENTATION.md`**: Rule-based classification details
- **`PROJECT_STATUS_REPORT.md`**: Current project status and issues


## Support

For questions about the evaluation methodology, technical issues, or implementation details, refer to the detailed documentation in each module or the comprehensive evaluation report.

---

*This project demonstrates the application of state-of-the-art ASR technology to medical audio transcription, with a focus on accuracy, reliability, and clinical applicability.*
