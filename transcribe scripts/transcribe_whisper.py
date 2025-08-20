#!/usr/bin/env python3
"""
Audio Transcription Script using OpenAI Whisper Large from Hugging Face

This script automatically transcribes all audio files in a specified input folder
using the OpenAI Whisper Large model from Hugging Face.

Usage:
    python transcribe_whisper.py --input_folder /path/to/audio/files --output_folder /path/to/output --model openai/whisper-large-v3
"""

import argparse
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import List, Optional
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import librosa
import numpy as np

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*transcription using a multilingual Whisper.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    """Class to handle audio transcription using Whisper models from Hugging Face."""
    
    def __init__(self, model_name: str = "openai/whisper-large-v3", force_cpu: bool = False, language: str = "en"):
        """
        Initialize the transcriber with the specified model.
        
        Args:
            model_name: Hugging Face model identifier
            force_cpu: Force CPU usage even if CUDA is available
            language: Language code for transcription (default: "en" for English)
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.force_cpu = force_cpu
        self.language = language
        self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Transcription language: {language}")
        if force_cpu and torch.cuda.is_available():
            logger.info("CUDA available but forcing CPU usage")
        
    def load_model(self):
        """Load the Whisper model and processor from Hugging Face."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            
            # Load model with appropriate data type
            if self.device == "cuda":
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16
                ).to(self.device)
            else:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Audio data as numpy array
        """
        try:
            # Try to load with soundfile first (faster for common formats)
            try:
                audio, sample_rate = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)  # Convert stereo to mono
                logger.debug(f"Loaded audio with soundfile: {audio_path}")
            except Exception as sf_error:
                logger.debug(f"Soundfile failed for {audio_path}: {sf_error}")
                # Fallback to librosa with modern approach
                try:
                    # Use librosa's modern loading approach to avoid deprecation warnings
                    audio, sample_rate = librosa.load(
                        audio_path, 
                        sr=16000,
                        mono=True  # Ensure mono output
                    )
                    logger.debug(f"Loaded audio with librosa: {audio_path}")
                except Exception as librosa_error:
                    logger.error(f"Both soundfile and librosa failed for {audio_path}")
                    logger.error(f"Soundfile error: {sf_error}")
                    logger.error(f"Librosa error: {librosa_error}")
                    raise Exception(f"Failed to load audio file: {audio_path}")
            
            # Ensure audio is mono and correct shape
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to 16kHz if necessary (only if not already 16kHz)
            if sample_rate != 16000:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
            
            # Ensure audio is float32 and normalized
            audio = audio.astype(np.float32)
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing: {audio_path}")
            
            # Load and preprocess audio
            audio = self.load_audio(audio_path)
            
            # Process audio with Whisper
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            # Ensure input features match model's data type
            if self.device == "cuda":
                inputs.input_features = inputs.input_features.to(self.device).half()
            else:
                inputs.input_features = inputs.input_features.float()
            
            # Generate transcription with proper parameters
            predicted_ids = self.model.generate(
                inputs.input_features,
                language="en",  # Force English transcription
                task="transcribe"  # Explicitly set task
            )
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            logger.info(f"Successfully transcribed: {audio_path}")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_path}: {str(e)}")
            raise
    
    def get_audio_files(self, input_folder: str) -> List[str]:
        """
        Find all audio files in the input folder.
        
        Args:
            input_folder: Path to the input folder
            
        Returns:
            List of audio file paths
        """
        audio_extensions = {
            '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', 
            '.wma', '.aiff', '.au', '.opus', '.webm'
        }
        
        audio_files = []
        input_path = Path(input_folder)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
        
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                audio_files.append(str(file_path))
        
        logger.info(f"Found {len(audio_files)} audio files in {input_folder}")
        return audio_files
    
    def process_folder(self, input_folder: str, output_folder: str):
        """
        Process all audio files in the input folder and save transcriptions.
        
        Args:
            input_folder: Path to the input folder
            output_folder: Path to the output folder
        """
        # Create output folder if it doesn't exist
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of audio files
        audio_files = self.get_audio_files(input_folder)
        
        if not audio_files:
            logger.warning("No audio files found in the input folder")
            return
        
        # Load model
        self.load_model()
        
        # Process each audio file
        successful_transcriptions = 0
        failed_transcriptions = 0
        
        for audio_file in audio_files:
            try:
                # Generate output filename
                audio_path = Path(audio_file)
                model_name_short = self.model_name.split('/')[-1]
                output_filename = f"{audio_path.stem}_{model_name_short}.txt"
                output_file = output_path / output_filename
                
                # Skip if output file already exists
                if output_file.exists():
                    logger.info(f"Skipping {audio_file} - output already exists")
                    continue
                
                # Transcribe audio
                transcription = self.transcribe_audio(audio_file)
                
                # Save transcription
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                
                logger.info(f"Saved transcription to: {output_file}")
                successful_transcriptions += 1
                
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {str(e)}")
                failed_transcriptions += 1
                continue
        
        logger.info(f"Processing complete. Successful: {successful_transcriptions}, Failed: {failed_transcriptions}")

def main():
    """Main function to handle command-line arguments and run transcription."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python transcribe_whisper.py --input_folder ./audio --output_folder ./transcriptions
    python transcribe_whisper.py --input_folder ./audio --output_folder ./transcriptions --model openai/whisper-large-v3
    python transcribe_whisper.py --input_folder ./audio --output_folder ./transcriptions --force_cpu
        """
    )
    
    parser.add_argument(
        "--input_folder",
        required=True,
        help="Path to the folder containing audio files"
    )
    
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Path to the folder where transcriptions will be saved"
    )
    
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3",
        help="Hugging Face model identifier (default: openai/whisper-large-v3)"
    )
    
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU usage even if CUDA is available (useful for debugging GPU issues)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize transcriber
        transcriber = AudioTranscriber(model_name=args.model, force_cpu=args.force_cpu)
        
        # Process the folder
        transcriber.process_folder(args.input_folder, args.output_folder)
        
        logger.info("Transcription process completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Transcription process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
