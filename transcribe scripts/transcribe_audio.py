#!/usr/bin/env python3

import os
import argparse
import glob
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import traceback
from tqdm import tqdm
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import pipeline

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('transcription.log')
        ]
    )
    return logging.getLogger(__name__)

def find_audio_files(input_dir: str) -> List[str]:
    """Find all audio files in the specified directory."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        pattern = os.path.join(input_dir, f"**/*{ext}")
        audio_files.extend(glob.glob(pattern, recursive=True))
    
    return audio_files

def is_whisper_model(model_name: str) -> bool:
    """Check if the model is an OpenAI Whisper model."""
    return "whisper" in model_name.lower()

def create_dataset_from_files(audio_files: List[str]) -> Dataset:
    """Create a HuggingFace dataset from a list of audio files."""
    import librosa
    
    def process_audio_file(file_path):
        try:
            # For simplicity, we'll just return the path and let the model handle loading
            return {
                "file_path": file_path,
                "audio_id": os.path.splitext(os.path.basename(file_path))[0]
            }
        except Exception as e:
            logger.warning(f"Error pre-processing {file_path}: {str(e)}")
            return None
    
    # Process all files and filter out None values
    data = [process_audio_file(file) for file in audio_files]
    data = [d for d in data if d is not None]
    
    # Create the dataset
    dataset = Dataset.from_list(data)
    return dataset

def load_asr_model(model_name: str, batch_size: int = 8):
    """Load the specified ASR model with batch processing capability."""
    try:
        logger.info(f"Loading ASR model: {model_name}")
        
        # Use different loading strategies based on model type
        if is_whisper_model(model_name):
            return load_whisper_model(model_name, batch_size)
        else:
            # Use pipeline with batch processing
            device = 0 if torch.cuda.is_available() else -1
            asr = pipeline(
                "automatic-speech-recognition", 
                model=model_name, 
                device=device,
                batch_size=batch_size
            )
            return asr
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

def load_whisper_model(model_name: str, batch_size: int = 8):
    """Specifically load a Whisper model with appropriate parameters for batch processing."""
    try:
        logger.info(f"Loading Whisper model: {model_name}")
        
        # Import specific Whisper components
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        
        # Load model components separately to have more control
        processor = WhisperProcessor.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # Create a custom batch processing pipeline
        def custom_whisper_batch_pipeline(batch):
            results = []
            
            try:
                import librosa
                import numpy as np
                
                # Process each audio file in the batch
                input_features_list = []
                for file_path in batch["file_path"]:
                    audio, sr = librosa.load(file_path, sr=16000)
                    inputs = processor(
                        audio, 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    ).input_features
                    input_features_list.append(inputs)
                
                # Stack inputs for batch processing
                if input_features_list:
                    batch_input_features = torch.cat(input_features_list, dim=0).to(device)
                    
                    # Generate tokens for the entire batch
                    with torch.no_grad():
                        generated_tokens = model.generate(
                            batch_input_features,
                            task="transcribe",
                            language="en"
                        )
                    
                    # Decode each output in the batch
                    transcriptions = processor.batch_decode(
                        generated_tokens, 
                        skip_special_tokens=True
                    )
                    
                    # Create results with text and audio_id
                    for i, text in enumerate(transcriptions):
                        results.append({
                            "text": text,
                            "audio_id": batch["audio_id"][i]
                        })
            
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                logger.debug(traceback.format_exc())
            
            return results
        
        return custom_whisper_batch_pipeline
        
    except Exception as e:
        logger.error(f"Failed to load Whisper model {model_name}: {e}")
        logger.debug(traceback.format_exc())
        raise

def process_dataset_with_model(dataset: Dataset, asr_model, model_name: str, output_dir: str, batch_size: int = 8):
    """Process the entire dataset using the ASR model with batch processing."""
    successful = 0
    failed = 0
    
    # Define a processing function based on model type
    if is_whisper_model(model_name):
        # For whisper models with custom batch pipeline
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            try:
                # Process the batch with our custom batch processor
                results = asr_model(batch)
                
                # Save each result
                for result in results:
                    if "text" in result and result["text"]:
                        # Get the original file path from the dataset
                        original_file = [f for f in dataset.select(range(i, min(i + batch_size, len(dataset))))["file_path"] 
                                        if os.path.splitext(os.path.basename(f))[0] == result["audio_id"]][0]
                        
                        # Save the transcription
                        output_path = save_transcription(result["text"], original_file, output_dir, model_name)
                        logger.info(f"Saved transcription to {output_path}")
                        successful += 1
                    else:
                        failed += 1
                        logger.warning(f"No transcription generated for {result.get('audio_id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error processing batch starting at {i}: {str(e)}")
                logger.debug(traceback.format_exc())
                failed += batch_size
    else:
        # For standard pipeline models
        # Create a dataloader for batch processing
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=lambda batch: {"file_path": [item["file_path"] for item in batch],
                                     "audio_id": [item["audio_id"] for item in batch]}
        )
        
        for batch in tqdm(dataloader, desc="Processing batches"):
            try:
                # Process each file in the batch individually (pipeline handles batching internally)
                for idx, file_path in enumerate(batch["file_path"]):
                    try:
                        # The pipeline might handle batching internally
                        result = asr_model(file_path)
                        
                        if isinstance(result, dict) and "text" in result:
                            # Save the transcription
                            output_path = save_transcription(result["text"], file_path, output_dir, model_name)
                            logger.info(f"Saved transcription to {output_path}")
                            successful += 1
                        else:
                            failed += 1
                            logger.warning(f"No transcription generated for {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")
                        failed += 1
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                logger.debug(traceback.format_exc())
                failed += len(batch["file_path"])
    
    return successful, failed

def save_transcription(text: str, audio_file: str, output_dir: str, model_name: str) -> str:
    """Save the transcription to a text file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create output filename based on the input audio filename
    audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
    model_shortname = model_name.split('/')[-1] if '/' in model_name else model_name
    output_filename = f"{audio_basename}_{model_shortname}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files using ASR")
    parser.add_argument("--input", "-i", required=True, help="Input folder containing audio files")
    parser.add_argument("--output", "-o", required=True, help="Output folder for transcriptions")
    parser.add_argument("--model", "-m", default="facebook/wav2vec2-base-960h", 
                        help="ASR model name (default: facebook/wav2vec2-base-960h)")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                        help="Number of files to process in each batch (default: 8)")
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        return 1
    
    # Find all audio files
    audio_files = find_audio_files(args.input)
    if not audio_files:
        logger.warning(f"No audio files found in {args.input}")
        return 0
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Create a dataset from audio files
    logger.info("Creating dataset from audio files")
    dataset = create_dataset_from_files(audio_files)
    logger.info(f"Created dataset with {len(dataset)} entries")
    
    # Load the ASR model with batch processing capability
    try:
        logger.info(f"Loading ASR model with batch processing")
        asr_model = load_asr_model(args.model, args.batch_size)
    except Exception as e:
        logger.error(f"Failed to initialize ASR model: {str(e)}")
        return 1
    
    # Process the entire dataset
    try:
        logger.info("Starting batch processing")
        successful, failed = process_dataset_with_model(
            dataset, 
            asr_model, 
            args.model, 
            args.output, 
            args.batch_size
        )
        
        logger.info(f"Transcription complete. Successful: {successful}, Failed: {failed}")
    except Exception as e:
        logger.error(f"Error during batch processing: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return 0

if __name__ == "__main__":
    logger = setup_logging()
    exit_code = main()
    exit(exit_code)