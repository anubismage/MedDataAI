import pandas as pd
import numpy as np
from jiwer import wer, cer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import csv

# Path to the CSV file
csv_path = "Sonograpy-Gold-whipser-voxtral.csv"

def load_data():
    """Load the transcription data from the CSV file"""
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data with {len(df)} rows")
        # Print column names to verify
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def calculate_metrics(df):
    """
    Calculate WER, CER, and Cosine Similarity for each model
    
    Returns:
        dict: Dictionary with metrics for each model
    """
    # Define the models to evaluate (updated for the new dataset)
    models = ["Whisper-Large-v3", "Voxtral-Mini-3b"]
    
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
    # Define common sonography/obstetric medical terms specific to the dataset
    if medical_terms is None:
        medical_terms = [
            "ultrasound", "sonography", "obstetric", "obstetrics", "graphy", "echogenic",
            "hyperechoic", "hypoechoic", "anechoic", "gestational", "sac", "fetal", "pole",
            "cardiac", "activity", "decidual", "reaction", "membrane", "separation", "crl",
            "cervix", "adnexal", "pathology", "pouch", "douglas", "conception", "edd",
            "gravid", "uterus", "endometrial", "echo", "ovary", "ovaries", "fluid",
            "adenomyotic", "adenomyosis", "bulky", "globular", "placenta", "vertex",
            "presentation", "fhr", "afi", "liquor", "vessel", "cord", "bpd", "hc", "ac",
            "fl", "hl", "gestational", "age", "birth", "weight", "artery", "pressure",
            "intrauterine", "pregnancy", "followup", "follow", "up", "lmp", "weeks",
            "days", "mm", "cm", "centimeter", "millimeter", "scar", "thickness",
            "doppler", "examination", "normal", "cyst", "follicle", "anechoic",
            "posterior", "anterior", "transverse", "measurements", "measures"
        ]
    
    models = ["Whisper-Large-v3", "Voxtral-Mini-3b"]
    
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

def analyze_specific_medical_cases(df):
    """
    Analyze specific medical case types and their accuracy
    
    Args:
        df: DataFrame with transcriptions
    
    Returns:
        dict: Dictionary with case-type-specific accuracy
    """
    models = ["Whisper-Large-v3", "Voxtral-Mini-3b"]
    
    # Define case types based on content patterns
    case_types = {
        "Early Obstetric": ["early obstetric", "early obstetrics", "lmp", "edd", "weeks", "gestational"],
        "Full-term Obstetric": ["full-time obstetric", "full term", "vertex presentation", "bpd", "hc", "ac", "fl"],
        "Pelvic Ultrasound": ["pelvic ultrasound", "uterus", "ovary", "ovaries", "endometrial"],
        "Follow-up": ["follow up", "followup", "mtp medicine", "complete abortion"],
        "Adenomyosis": ["adenomyosis", "adenomyotic", "bulky", "globular"]
    }
    
    results = {}
    
    for model in models:
        if model not in df.columns:
            continue
            
        case_metrics = {}
        
        for case_type, keywords in case_types.items():
            case_wer_scores = []
            case_cer_scores = []
            case_count = 0
            
            for _, row in df.iterrows():
                reference = str(row["Manual Transcription"]).lower()
                hypothesis = str(row[model]).lower()
                
                # Skip empty entries
                if pd.isna(reference) or pd.isna(hypothesis):
                    continue
                
                # Check if this sample belongs to the current case type
                is_case_type = any(keyword in reference for keyword in keywords)
                
                if is_case_type:
                    case_count += 1
                    try:
                        wer_score = wer(reference, hypothesis)
                        cer_score = cer(reference, hypothesis)
                        case_wer_scores.append(wer_score)
                        case_cer_scores.append(cer_score)
                    except Exception as e:
                        print(f"Error calculating metrics for {case_type}: {e}")
            
            case_metrics[case_type] = {
                "count": case_count,
                "wer_mean": np.mean(case_wer_scores) if case_wer_scores else None,
                "cer_mean": np.mean(case_cer_scores) if case_cer_scores else None
            }
        
        results[model] = case_metrics
    
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
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # WER comparison
    wer_means = [metrics[model]["wer"]["mean"] for model in models]
    wer_stds = [metrics[model]["wer"]["std"] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, wer_means, yerr=wer_stds, capsize=5, 
                   color=['#2E86AB', '#A23B72'], alpha=0.8)
    plt.title("Word Error Rate (WER) Comparison: Whisper vs Voxtral", fontsize=16, fontweight='bold')
    plt.ylabel("WER (lower is better)", fontsize=12)
    plt.xticks(rotation=0)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, wer_means, wer_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wer_comparison_whisper_voxtral.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # CER comparison
    cer_means = [metrics[model]["cer"]["mean"] for model in models]
    cer_stds = [metrics[model]["cer"]["std"] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, cer_means, yerr=cer_stds, capsize=5,
                   color=['#2E86AB', '#A23B72'], alpha=0.8)
    plt.title("Character Error Rate (CER) Comparison: Whisper vs Voxtral", fontsize=16, fontweight='bold')
    plt.ylabel("CER (lower is better)", fontsize=12)
    plt.xticks(rotation=0)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, cer_means, cer_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cer_comparison_whisper_voxtral.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cosine similarity comparison
    cosine_means = [metrics[model]["cosine"]["mean"] for model in models]
    cosine_stds = [metrics[model]["cosine"]["std"] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, cosine_means, yerr=cosine_stds, capsize=5,
                   color=['#2E86AB', '#A23B72'], alpha=0.8)
    plt.title("Semantic Similarity Comparison: Whisper vs Voxtral", fontsize=16, fontweight='bold')
    plt.ylabel("Cosine Similarity (higher is better)", fontsize=12)
    plt.xticks(rotation=0)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, cosine_means, cosine_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cosine_similarity_comparison_whisper_voxtral.png"), dpi=300, bbox_inches='tight')
    plt.close()

def export_to_csv(metrics, sample_metrics_df, medical_metrics, case_metrics, output_dir="./output"):
    """
    Export all metrics to CSV files
    
    Args:
        metrics: Dictionary with aggregated metrics for each model
        sample_metrics_df: DataFrame with per-sample metrics
        medical_metrics: Dictionary with medical term accuracy for each model
        case_metrics: Dictionary with case-type-specific metrics
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
    summary_df.to_csv(os.path.join(output_dir, "metrics_summary_whisper_voxtral.csv"), index=False)
    print(f"Exported aggregated metrics to {os.path.join(output_dir, 'metrics_summary_whisper_voxtral.csv')}")
    
    # 2. Export per-sample metrics
    sample_metrics_df.to_csv(os.path.join(output_dir, "sample_metrics_whisper_voxtral.csv"), index=False)
    print(f"Exported per-sample metrics to {os.path.join(output_dir, 'sample_metrics_whisper_voxtral.csv')}")
    
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
    medical_df.to_csv(os.path.join(output_dir, "medical_term_accuracy_whisper_voxtral.csv"), index=False)
    print(f"Exported medical term accuracy to {os.path.join(output_dir, 'medical_term_accuracy_whisper_voxtral.csv')}")
    
    # 4. Export case-type-specific metrics
    case_rows = []
    for model, cases in case_metrics.items():
        for case_type, case_metrics_data in cases.items():
            case_rows.append({
                "Model": model,
                "Case_Type": case_type,
                "Sample_Count": case_metrics_data["count"],
                "WER_Mean": case_metrics_data["wer_mean"],
                "CER_Mean": case_metrics_data["cer_mean"]
            })
    
    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(os.path.join(output_dir, "case_type_metrics_whisper_voxtral.csv"), index=False)
    print(f"Exported case-type metrics to {os.path.join(output_dir, 'case_type_metrics_whisper_voxtral.csv')}")

def main():
    """Main function to run the analysis"""
    print("Starting Whisper vs Voxtral transcription metrics analysis...")
    print("=" * 60)
    
    # Load data from CSV file
    df = load_data()
    if df is None:
        return
        
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics, sample_metrics_df = calculate_metrics(df)
    if metrics is None:
        return
        
    # Display results
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    for model, model_metrics in metrics.items():
        print(f"\n{model}:")
        print(f"  Word Error Rate (WER):     {model_metrics['wer']['mean']:.4f} ± {model_metrics['wer']['std']:.4f}")
        print(f"  Character Error Rate (CER): {model_metrics['cer']['mean']:.4f} ± {model_metrics['cer']['std']:.4f}")
        print(f"  Cosine Similarity:         {model_metrics['cosine']['mean']:.4f} ± {model_metrics['cosine']['std']:.4f}")
    
    # Analyze medical term accuracy
    print("\nAnalyzing medical term accuracy...")
    medical_metrics = analyze_medical_terms(df)
    
    print("\n" + "=" * 60)
    print("MEDICAL TERM ACCURACY")
    print("=" * 60)
    for model, term_metrics in medical_metrics.items():
        print(f"\n{model}:")
        print(f"  Overall Medical Term Accuracy: {term_metrics['overall_accuracy']:.4f}")
    
    # Analyze case-specific performance
    print("\nAnalyzing case-specific performance...")
    case_metrics = analyze_specific_medical_cases(df)
    
    print("\n" + "=" * 60)
    print("CASE-SPECIFIC PERFORMANCE")
    print("=" * 60)
    for model, cases in case_metrics.items():
        print(f"\n{model}:")
        for case_type, case_data in cases.items():
            if case_data["count"] > 0:
                print(f"  {case_type} ({case_data['count']} samples):")
                print(f"    WER: {case_data['wer_mean']:.4f}")
                print(f"    CER: {case_data['cer_mean']:.4f}")
        
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(metrics)
    
    # Export to CSV
    print("\nExporting metrics to CSV...")
    export_to_csv(metrics, sample_metrics_df, medical_metrics, case_metrics)
    
    print("\nAnalysis complete. Results saved to figures/ directory and output/ directory.")
    
    # Determine the best model based on metrics
    print("\n" + "=" * 60)
    print("BEST MODEL DETERMINATION")
    print("=" * 60)
    best_wer_model = min(metrics.items(), key=lambda x: x[1]['wer']['mean'])[0]
    best_cer_model = min(metrics.items(), key=lambda x: x[1]['cer']['mean'])[0]
    best_cosine_model = max(metrics.items(), key=lambda x: x[1]['cosine']['mean'])[0]
    best_medical_model = max(medical_metrics.items(), key=lambda x: x[1]['overall_accuracy'])[0]
    
    print(f"Best model by Word Error Rate: {best_wer_model}")
    print(f"Best model by Character Error Rate: {best_cer_model}")
    print(f"Best model by Semantic Similarity: {best_cosine_model}")
    print(f"Best model by Medical Term Accuracy: {best_medical_model}")
    
    # Overall recommendation
    print(f"\n" + "=" * 60)
    print("OVERALL RECOMMENDATION")
    print("=" * 60)
    
    whisper_wer = metrics["Whisper-Large-v3"]["wer"]["mean"]
    voxtral_wer = metrics["Voxtral-Mini-3b"]["wer"]["mean"]
    
    whisper_cosine = metrics["Whisper-Large-v3"]["cosine"]["mean"]
    voxtral_cosine = metrics["Voxtral-Mini-3b"]["cosine"]["mean"]
    
    if whisper_wer < voxtral_wer and whisper_cosine > voxtral_cosine:
        print("RECOMMENDATION: Whisper-Large-v3 performs better overall")
    elif voxtral_wer < whisper_wer and voxtral_cosine > whisper_cosine:
        print("RECOMMENDATION: Voxtral-Mini-3b performs better overall")
    else:
        print("RECOMMENDATION: Mixed results - choice depends on specific use case priorities")
        if whisper_wer < voxtral_wer:
            print("  - Whisper-Large-v3 has lower error rates")
        if voxtral_cosine > whisper_cosine:
            print("  - Voxtral-Mini-3b has better semantic similarity")

if __name__ == "__main__":
    main()
