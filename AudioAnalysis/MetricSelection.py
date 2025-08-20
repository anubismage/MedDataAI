import pandas as pd
import numpy as np
from jiwer import wer, cer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import csv

# Path to the Excel file
excel_path = "Sonography Transcription AI project - Gold.xlsx"

def load_data():
    """Load the transcription data from the Excel file"""
    try:
        # Load the Excel file
        df = pd.read_excel(excel_path)
        print(f"Successfully loaded data with {len(df)} rows")
        # Print column names to verify
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def calculate_metrics(df):
    """
    Calculate WER, CER, and Cosine Similarity for each model
    
    Returns:
        dict: Dictionary with metrics for each model
    """
    # Define the models to evaluate
    models = ["Whisper-Large-v3", "parakeet-rnnt-1.1b", 
              "vosk-model-en-in-0.5", "wav2vec2-base-960h"]
    
    # Initialize sentence transformer model for cosine similarity
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded sentence transformer model successfully")
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}")
        return None
    
    results = {}
    
    # Prepare for per-sample metrics
    sample_metrics = []
    
    # Process each model
    for model in models:
        if model not in df.columns:
            print(f"Warning: Column '{model}' not found in data")
            continue
            
        wer_scores = []
        cer_scores = []
        cosine_scores = []
        
        # Calculate metrics for each sample
        for idx, row in df.iterrows():
            reference = str(row["Manual Transcription"]).lower()
            hypothesis = str(row[model]).lower()
            filename = row.get("Filename", f"Sample_{idx}")
            
            # Skip empty entries
            if pd.isna(reference) or pd.isna(hypothesis) or reference.strip() == "" or hypothesis.strip() == "":
                continue
                
            sample_result = {
                "Filename": filename,
                "Model": model
            }
            
            # Calculate WER
            try:
                wer_score = wer(reference, hypothesis)
                wer_scores.append(wer_score)
                sample_result["WER"] = wer_score
            except Exception as e:
                print(f"Error calculating WER for {model} on sample {filename}: {e}")
                sample_result["WER"] = None
            
            # Calculate CER
            try:
                cer_score = cer(reference, hypothesis)
                cer_scores.append(cer_score)
                sample_result["CER"] = cer_score
            except Exception as e:
                print(f"Error calculating CER for {model} on sample {filename}: {e}")
                sample_result["CER"] = None
            
            # Calculate cosine similarity
            try:
                ref_embedding = sentence_model.encode([reference])
                hyp_embedding = sentence_model.encode([hypothesis])
                similarity = cosine_similarity(ref_embedding, hyp_embedding)[0][0]
                cosine_scores.append(similarity)
                sample_result["Cosine_Similarity"] = similarity
            except Exception as e:
                print(f"Error calculating cosine similarity for {model} on sample {filename}: {e}")
                sample_result["Cosine_Similarity"] = None
                
            # Add to sample metrics list
            sample_metrics.append(sample_result)
        
        # Store metrics
        results[model] = {
            "wer": {
                "mean": np.mean(wer_scores) if wer_scores else None,
                "std": np.std(wer_scores) if wer_scores else None,
                "individual": wer_scores
            },
            "cer": {
                "mean": np.mean(cer_scores) if cer_scores else None,
                "std": np.std(cer_scores) if cer_scores else None,
                "individual": cer_scores
            },
            "cosine": {
                "mean": np.mean(cosine_scores) if cosine_scores else None,
                "std": np.std(cosine_scores) if cosine_scores else None,
                "individual": cosine_scores
            }
        }
    
    # Create sample metrics DataFrame
    sample_metrics_df = pd.DataFrame(sample_metrics)
    
    return results, sample_metrics_df

def analyze_medical_terms(df, medical_terms=None):
    """
    Analyze how well each model handles medical terms
    
    Args:
        df: DataFrame with transcriptions
        medical_terms: List of medical terms to look for (if None, will use default list)
    
    Returns:
        dict: Dictionary with medical term accuracy for each model
    """
    # Define some common sonography/medical terms if not provided
    if medical_terms is None:
        medical_terms = [
            "ultrasound", "sonography", "transducer", "doppler", "echogenic",
            "hyperechoic", "hypoechoic", "anechoic", "abdomen", "gallbladder",
            "liver", "kidney", "spleen", "pancreas", "aorta", "fetus", 
            "gestation", "trimester", "placenta", "amnio", "cardiac",
            "obstetric", "gynecologic", "vascular", "thyroid", "pouch of douglas",
            "cyst", "mass", "lesion", "fluid", "calcification", "biopsy",
            "adenomyoma", "fundal", "myometrium", "endometrium", "ovary",
        ]
    
    models = ["Whisper-Large-v3", "parakeet-rnnt-1.1b", 
              "vosk-model-en-in-0.5", "wav2vec2-base-960h"]
    
    results = {}
    
    for model in models:
        if model not in df.columns:
            continue
            
        term_present_in_reference = {term: 0 for term in medical_terms}
        term_correctly_transcribed = {term: 0 for term in medical_terms}
        
        for _, row in df.iterrows():
            reference = str(row["Manual Transcription"]).lower()
            hypothesis = str(row[model]).lower()
            
            # Skip empty entries
            if pd.isna(reference) or pd.isna(hypothesis):
                continue
                
            # Check each medical term
            for term in medical_terms:
                if term.lower() in reference:
                    term_present_in_reference[term] += 1
                    if term.lower() in hypothesis:
                        term_correctly_transcribed[term] += 1
        
        # Calculate accuracy for each term
        term_accuracy = {}
        for term in medical_terms:
            if term_present_in_reference[term] > 0:
                term_accuracy[term] = term_correctly_transcribed[term] / term_present_in_reference[term]
            else:
                term_accuracy[term] = None
                
        # Overall medical term accuracy
        total_mentions = sum(term_present_in_reference.values())
        total_correct = sum(term_correctly_transcribed.values())
        overall_accuracy = total_correct / total_mentions if total_mentions > 0 else 0
        
        results[model] = {
            "term_accuracy": term_accuracy,
            "overall_accuracy": overall_accuracy
        }
    
    return results

def visualize_results(metrics, output_dir="./figures"):
    """
    Create visualizations of the metrics
    
    Args:
        metrics: Dictionary with metrics for each model
        output_dir: Directory to save figures
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(metrics.keys())
    
    # WER comparison
    wer_means = [metrics[model]["wer"]["mean"] for model in models]
    plt.figure(figsize=(10, 6))
    plt.bar(models, wer_means)
    plt.title("Word Error Rate (WER) by Model")
    plt.ylabel("WER (lower is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wer_comparison.png"))
    plt.close()
    
    # CER comparison
    cer_means = [metrics[model]["cer"]["mean"] for model in models]
    plt.figure(figsize=(10, 6))
    plt.bar(models, cer_means)
    plt.title("Character Error Rate (CER) by Model")
    plt.ylabel("CER (lower is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cer_comparison.png"))
    plt.close()
    
    # Cosine similarity comparison
    cosine_means = [metrics[model]["cosine"]["mean"] for model in models]
    plt.figure(figsize=(10, 6))
    plt.bar(models, cosine_means)
    plt.title("Cosine Similarity by Model")
    plt.ylabel("Cosine Similarity (higher is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cosine_similarity_comparison.png"))
    plt.close()

def export_to_csv(metrics, sample_metrics_df, medical_metrics, output_dir="./output"):
    """
    Export all metrics to CSV files
    
    Args:
        metrics: Dictionary with aggregated metrics for each model
        sample_metrics_df: DataFrame with per-sample metrics
        medical_metrics: Dictionary with medical term accuracy for each model
        output_dir: Directory to save CSV files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export aggregated metrics summary
    summary_rows = []
    for model, model_metrics in metrics.items():
        summary_rows.append({
            "Model": model,
            "WER_Mean": model_metrics["wer"]["mean"],
            "WER_StdDev": model_metrics["wer"]["std"],
            "CER_Mean": model_metrics["cer"]["mean"],
            "CER_StdDev": model_metrics["cer"]["std"],
            "Cosine_Similarity_Mean": model_metrics["cosine"]["mean"],
            "Cosine_Similarity_StdDev": model_metrics["cosine"]["std"],
            "Medical_Term_Accuracy": medical_metrics[model]["overall_accuracy"]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    print(f"Exported aggregated metrics to {os.path.join(output_dir, 'metrics_summary.csv')}")
    
    # 2. Export per-sample metrics
    sample_metrics_df.to_csv(os.path.join(output_dir, "sample_metrics.csv"), index=False)
    print(f"Exported per-sample metrics to {os.path.join(output_dir, 'sample_metrics.csv')}")
    
    # 3. Export detailed medical term accuracy
    medical_rows = []
    for model, term_metrics in medical_metrics.items():
        for term, accuracy in term_metrics["term_accuracy"].items():
            if accuracy is not None:
                medical_rows.append({
                    "Model": model,
                    "Medical_Term": term,
                    "Accuracy": accuracy
                })
    
    medical_df = pd.DataFrame(medical_rows)
    medical_df.to_csv(os.path.join(output_dir, "medical_term_accuracy.csv"), index=False)
    print(f"Exported medical term accuracy to {os.path.join(output_dir, 'medical_term_accuracy.csv')}")

def main():
    """Main function to run the analysis"""
    print("Starting transcription metrics analysis...")
    
    # Load data from Excel file
    df = load_data()
    if df is None:
        return
        
    # Calculate metrics
    print("Calculating metrics...")
    metrics, sample_metrics_df = calculate_metrics(df)
    if metrics is None:
        return
        
    # Display results
    print("\n===== METRICS SUMMARY =====")
    for model, model_metrics in metrics.items():
        print(f"\nModel: {model}")
        print(f"  WER: {model_metrics['wer']['mean']:.4f} ± {model_metrics['wer']['std']:.4f}")
        print(f"  CER: {model_metrics['cer']['mean']:.4f} ± {model_metrics['cer']['std']:.4f}")
        print(f"  Cosine Similarity: {model_metrics['cosine']['mean']:.4f} ± {model_metrics['cosine']['std']:.4f}")
    
    # Analyze medical term accuracy
    print("\nAnalyzing medical term accuracy...")
    medical_metrics = analyze_medical_terms(df)
    
    print("\n===== MEDICAL TERM ACCURACY =====")
    for model, term_metrics in medical_metrics.items():
        print(f"\nModel: {model}")
        print(f"  Overall Medical Term Accuracy: {term_metrics['overall_accuracy']:.4f}")
        
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(metrics)
    
    # Export to CSV
    print("\nExporting metrics to CSV...")
    export_to_csv(metrics, sample_metrics_df, medical_metrics)
    
    print("\nAnalysis complete. Results saved to figures/ directory and output/ directory.")
    
    # Determine the best model based on metrics
    print("\n===== BEST MODEL DETERMINATION =====")
    best_wer_model = min(metrics.items(), key=lambda x: x[1]['wer']['mean'])[0]
    best_cer_model = min(metrics.items(), key=lambda x: x[1]['cer']['mean'])[0]
    best_cosine_model = max(metrics.items(), key=lambda x: x[1]['cosine']['mean'])[0]
    best_medical_model = max(medical_metrics.items(), key=lambda x: x[1]['overall_accuracy'])[0]
    
    print(f"Best model by WER: {best_wer_model}")
    print(f"Best model by CER: {best_cer_model}")
    print(f"Best model by Cosine Similarity: {best_cosine_model}")
    print(f"Best model by Medical Term Accuracy: {best_medical_model}")

if __name__ == "__main__":
    main()