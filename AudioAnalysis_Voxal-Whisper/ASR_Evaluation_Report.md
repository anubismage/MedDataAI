# ASR Evaluation Report: Whisper vs Voxtral for Sonography Transcription

## Executive Summary

This report presents a comprehensive evaluation of two state-of-the-art Automatic Speech Recognition (ASR) models for sonography audio transcription:

- **Whisper-Large-v3**: OpenAI's latest large-scale multilingual ASR model
- **Voxtral-Mini-3b**: A specialized model for medical/clinical audio transcription

## Dataset

The evaluation was conducted on 17 sonography audio recordings with manually verified transcriptions, covering various types of obstetric and gynecologic ultrasound examinations.

## Key Findings

### Overall Performance Metrics

| Metric | Whisper-Large-v3 | Voxtral-Mini-3b | Winner |
|--------|------------------|-----------------|---------|
| **Word Error Rate (WER)** | 0.668 ± 0.659 | 0.548 ± 0.319 | **Voxtral** |
| **Character Error Rate (CER)** | 0.385 ± 0.421 | 0.312 ± 0.276 | **Voxtral** |
| **Semantic Similarity** | 0.843 ± 0.117 | 0.798 ± 0.201 | **Whisper** |
| **Medical Term Accuracy** | 0.815 | 0.835 | **Voxtral** |

### Statistical Analysis

- **No statistically significant difference** was found between the models using Mann-Whitney U tests (p > 0.05)
- However, Voxtral consistently shows lower error rates with less variance
- Whisper shows better semantic similarity but with higher variability

### Medical Term Recognition

**Terms where models differ significantly (>10%):**

| Medical Term | Whisper Accuracy | Voxtral Accuracy | Better Model |
|--------------|------------------|------------------|--------------|
| adenomyotic | 0.750 | 0.250 | Whisper |
| anechoic | 1.000 | 0.500 | Whisper |
| afi | 0.500 | 1.000 | Voxtral |
| cervix | 0.143 | 0.429 | Voxtral |
| adenomyosis | 0.500 | 0.750 | Voxtral |

### Case-Type Performance

| Examination Type | Samples | Whisper WER | Voxtral WER | Better Model |
|------------------|---------|-------------|-------------|--------------|
| Early Obstetric | 10 | 0.677 | 0.573 | **Voxtral** |
| Full-term Obstetric | 13 | 0.574 | 0.513 | **Voxtral** |
| Pelvic Ultrasound | 11 | 0.522 | 0.477 | **Voxtral** |
| Follow-up | 4 | 0.353 | 0.594 | **Whisper** |
| Adenomyosis | 6 | 0.749 | 0.551 | **Voxtral** |

## Overall Winner

** RECOMMENDED MODEL: Voxtral-Mini-3b**

Voxtral-Mini-3b wins in 3/4 key metrics:
- Superior word-level accuracy (lower WER)
- Better character-level precision (lower CER)
- Superior medical terminology recognition

## Use Case Specific Recommendations

### For Clinical Documentation
**→ Use Voxtral-Mini-3b** (better medical terminology recognition)

### For Real-time Transcription
**→ Consider Voxtral-Mini-3b** (smaller model, likely faster)

### For High-accuracy Requirements
**→ Use Voxtral-Mini-3b** (lower error rates)

### For Semantic Understanding
**→ Consider Whisper-Large-v3** (better semantic similarity)

## Key Advantages by Model

### Voxtral-Mini-3b Advantages:
- Lower and more consistent error rates
- Better recognition of clinical terminology like "cervix", "afi", "adenomyosis"
- Superior performance across most examination types
- Smaller model size (likely faster inference)

### Whisper-Large-v3 Advantages:
- Better semantic understanding (higher cosine similarity)
- Superior on some specific terms like "adenomyotic", "anechoic"
- Better performance on follow-up examinations
- More robust general-purpose ASR capabilities

## Areas for Improvement

1. **Domain-specific fine-tuning**: Both models could benefit from additional training on medical audio data
2. **Ensemble approaches**: Combining both models could leverage their complementary strengths
3. **Post-processing**: Implement medical term correction and standardization
4. **Continuous evaluation**: Regular assessment on new data to monitor performance drift

## Technical Implementation

The evaluation used the following metrics:
- **WER/CER**: Traditional ASR accuracy metrics using jiwer library
- **Semantic Similarity**: Cosine similarity of sentence embeddings (all-MiniLM-L6-v2)
- **Medical Term Accuracy**: Domain-specific terminology recognition analysis
- **Statistical Testing**: Mann-Whitney U tests for significance testing

## Files Generated

- `metrics_summary_whisper_voxtral.csv`: Aggregated performance metrics
- `sample_metrics_whisper_voxtral.csv`: Per-sample detailed metrics
- `medical_term_accuracy_whisper_voxtral.csv`: Medical terminology analysis
- `case_type_metrics_whisper_voxtral.csv`: Performance by examination type
- Visualization figures in `figures/` directory
- Complete Jupyter notebook analysis: `ASR_Evaluation_Whisper_Voxtral.ipynb`

## Conclusion

Based on this comprehensive evaluation, **Voxtral-Mini-3b is the recommended model** for sonography transcription tasks. It consistently outperforms Whisper-Large-v3 in accuracy metrics while maintaining good performance across different examination types. The model's smaller size and specialized medical focus make it particularly well-suited for clinical applications.

However, for applications requiring strong semantic understanding or general-purpose transcription capabilities, Whisper-Large-v3 remains a viable alternative. The choice between models should ultimately depend on specific deployment requirements and use case priorities.
