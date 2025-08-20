#!/usr/bin/env python3
"""
Audio Transcription Script using NVIDIA NeMo Parakeet ASR Models

This script automatically transcribes all audio files in a specified input folder
using NVIDIA NeMo (Parakeet) ASR models.

Usage:
    python transcribe_parakeet.py --input_folder /path/to/audio/files --output_folder /path/to/output --model stt_en_conformer_ctc_large
"""

import argparse
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union
import torch
import numpy as np
import soundfile as sf
import librosa

# NeMo-specific imports
try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    print("Error: NVIDIA NeMo not installed. Please install it with: pip install nemo_toolkit[asr]")
    sys.exit(1)

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")

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

class NemoTranscriber:
    """Class to handle audio transcription using NVIDIA NeMo Parakeet ASR models."""
    
    def __init__(self, model_name: str = "stt_en_conformer_ctc_large", force_cpu: bool = False):
        """
        Initialize the transcriber with the specified NeMo model.
        
        Args:
            model_name: NeMo model name (e.g., stt_en_conformer_ctc_large)
            force_cpu: Force CPU usage even if CUDA is available
        """
        self.model_name = model_name
        self.model = None
        self.force_cpu = force_cpu
        self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if force_cpu and torch.cuda.is_available():
            logger.info("CUDA available but forcing CPU usage")
        
    def load_model(self):
        """Load the NeMo ASR model."""
        try:
            logger.info(f"Loading NeMo model: {self.model_name}")
            
            # Set device map based on availability
            if self.device == "cpu":
                torch.set_default_tensor_type(torch.FloatTensor)
            else:
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
            
            # Determine the appropriate model class based on model name
            if "rnnt" in self.model_name.lower() or "transducer" in self.model_name.lower():
                # RNNT models use a different class
                logger.info(f"Detected RNNT/Transducer model: {self.model_name}")
                self.model = nemo_asr.models.EncDecRNNTModel.from_pretrained(self.model_name)
            else:
                # Default to CTC-based models
                logger.info(f"Using CTC model: {self.model_name}")
                self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(self.model_name)
            
            # Move model to the appropriate device
            self.model = self.model.to(self.device)
            
            logger.info("NeMo ASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NeMo model: {str(e)}")
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
                # Fallback to librosa
                audio, sample_rate = librosa.load(
                    audio_path, 
                    sr=16000,  # NeMo models typically expect 16kHz
                    mono=True
                )
                logger.debug(f"Loaded audio with librosa: {audio_path}")
            
            # Ensure audio is mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to 16kHz if necessary (NeMo models typically expect 16kHz)
            if sample_rate != 16000:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
                sample_rate = 16000
            
            # Ensure audio is float32 and normalized
            audio = audio.astype(np.float32)
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio, sample_rate
            
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
            audio, sample_rate = self.load_audio(audio_path)
            
            # Transcribe using NeMo model
            with torch.no_grad():
                if self.device == "cuda":
                    audio_tensor = torch.tensor(audio).cuda()
                else:
                    audio_tensor = torch.tensor(audio)
                
                # Get transcription - handle different model types
                result = self.model.transcribe([audio_tensor], batch_size=1)[0]
                
                # Handle different result types from various NeMo models
                if hasattr(result, 'text'):  # Hypothesis object from RNNT models
                    transcription = result.text
                elif isinstance(result, dict) and 'text' in result:  # Dictionary with text key
                    transcription = result['text']
                elif isinstance(result, str):  # Plain string (from CTC models)
                    transcription = result
                else:
                    # If it's some other object with string representation
                    transcription = str(result)
            
            logger.info(f"Successfully transcribed: {audio_path}")
            return transcription.strip() if isinstance(transcription, str) else transcription
            
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
                model_name_short = self.model_name.replace('/', '-')
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
        description="Transcribe audio files using NVIDIA NeMo (Parakeet) ASR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python transcribe_parakeet.py --input_folder ./audio --output_folder ./transcriptions
    python transcribe_parakeet.py --input_folder ./audio --output_folder ./transcriptions --model stt_en_conformer_ctc_large
    python transcribe_parakeet.py --input_folder ./audio --output_folder ./transcriptions --force_cpu
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
        default="stt_en_conformer_ctc_large",
        help="NeMo ASR model name (default: stt_en_conformer_ctc_large)"
    )
    
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU usage even if CUDA is available"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize transcriber
        transcriber = NemoTranscriber(model_name=args.model, force_cpu=args.force_cpu)
        
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