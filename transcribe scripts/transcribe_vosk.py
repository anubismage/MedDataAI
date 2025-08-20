#!/usr/bin/env python3
import argparse
import os
import sys
import json
import wave
import logging
import tempfile
import shutil
from typing import List, Optional, Tuple
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcription.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vosk_transcribe")

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    from pydub import AudioSegment
except ImportError as e:
    logger.error(f"Required library not found: {e}. Please install it using pip.")
    sys.exit(1)

# Define default paths for models
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "vosk-model-en-in-0.5")

def find_audio_files(input_dir: str) -> List[str]:
    """
    Find all audio files in the given directory.
    
    Args:
        input_dir: Directory to search for audio files
        
    Returns:
        List of paths to audio files
    """
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    try:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(audio_files)} audio files in {input_dir}")
        return audio_files
    except Exception as e:
        logger.error(f"Error finding audio files: {e}")
        return []


def load_vosk_model(model_name: str = DEFAULT_MODEL_PATH) -> Optional[Model]:
    """
    Load the specified Vosk model.
    
    Args:
        model_name: Name/path of the model to load (defaults to large model)
        
    Returns:
        Loaded Vosk model or None if loading failed
    """
    try:
        # Set Vosk log level to avoid excessive output
        SetLogLevel(-1)
        
        if not os.path.exists(model_name):
            logger.error(f"Model {model_name} not found. Please download it.")
            return None
            
        logger.info(f"Loading Vosk model from {model_name}...")
        model = Model(model_name)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading Vosk model: {e}")
        return None


def convert_to_wav(audio_path: str) -> Tuple[str, bool]:
    """
    Convert audio file to WAV format if it's not already in WAV format.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Tuple of (wav_file_path, is_temp_file) where:
            - wav_file_path is the path to the WAV file (original path or temp file)
            - is_temp_file is True if a temp file was created that needs to be deleted
    """
    if audio_path.lower().endswith('.wav'):
        return audio_path, False
    
    try:
        logger.info(f"Converting {audio_path} to WAV format")
        
        # Determine audio format from file extension
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == '.mp3':
            audio = AudioSegment.from_mp3(audio_path)
        elif ext == '.flac':
            audio = AudioSegment.from_file(audio_path, "flac")
        elif ext == '.ogg':
            audio = AudioSegment.from_ogg(audio_path)
        elif ext == '.m4a':
            audio = AudioSegment.from_file(audio_path, "m4a")
        else:
            # Try to load with a generic method
            audio = AudioSegment.from_file(audio_path)
        
        # Create temporary file with .wav extension
        fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)  # Close file descriptor
        
        # Convert to WAV (mono, 16-bit PCM)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_sample_width(2)  # Set to 16-bit
        audio.export(temp_wav_path, format="wav")
        
        logger.info(f"Successfully converted to {temp_wav_path}")
        return temp_wav_path, True
        
    except Exception as e:
        logger.error(f"Error converting audio file {audio_path}: {e}")
        traceback.print_exc()
        return None, False


def transcribe_audio_file(audio_path: str, model: Model) -> Optional[str]:
    """
    Transcribe an audio file using the provided Vosk model.
    Handles conversion to WAV if necessary.
    
    Args:
        audio_path: Path to the audio file
        model: Loaded Vosk model
        
    Returns:
        Transcribed text or None if transcription failed
    """
    temp_wav_file = None
    is_temp_file = False
    
    try:
        # Convert to WAV if needed
        wav_path, is_temp_file = convert_to_wav(audio_path)
        if wav_path is None:
            return None
            
        temp_wav_file = wav_path if is_temp_file else None
        
        # Open the WAV file
        wf = wave.open(wav_path, "rb")
        
        # Verify format is compatible
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            logger.warning(f"Audio file {wav_path} must be mono PCM format")
            return None
            
        # Create recognizer
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        
        # Process audio file
        result = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result_dict = json.loads(rec.Result())
                if 'text' in result_dict:
                    result += result_dict['text'] + " "
                    
        # Process final result
        final_result = json.loads(rec.FinalResult())
        if 'text' in final_result:
            result += final_result['text']
            
        return result.strip()
        
    except Exception as e:
        logger.error(f"Error transcribing {audio_path}: {e}")
        traceback.print_exc()
        return None
    finally:
        # Clean up temporary files
        if is_temp_file and temp_wav_file and os.path.exists(temp_wav_file):
            try:
                os.remove(temp_wav_file)
                logger.debug(f"Removed temporary file {temp_wav_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_wav_file}: {e}")


def save_transcription(text: str, audio_path: str, output_dir: str, model_name: str) -> bool:
    """
    Save transcription text to a file.
    
    Args:
        text: Transcribed text
        audio_path: Path to the original audio file
        output_dir: Directory to save the transcription
        model_name: Name of the model used for transcription
        
    Returns:
        True if saving was successful, False otherwise
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get the base name of the audio file without extension
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Extract just the model folder name without path
        model_short_name = os.path.basename(model_name)
        
        # Create output filename
        output_file = os.path.join(output_dir, f"{audio_basename}_{model_short_name}.txt")
        
        # Write the transcription
        with open(output_file, 'w') as f:
            f.write(text)
            
        logger.info(f"Transcription saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving transcription: {e}")
        return False


def main():
    """Main function to run the transcription process."""
    parser = argparse.ArgumentParser(description="Transcribe audio files using Vosk ASR")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing audio files")
    parser.add_argument("--output", "-o", required=True, help="Output directory for transcriptions")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_PATH, help=f"Path to Vosk model directory (default: {DEFAULT_MODEL_PATH})")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input) or not os.path.isdir(args.input):
        logger.error(f"Input directory {args.input} does not exist or is not a directory")
        sys.exit(1)
        
    # Find audio files
    audio_files = find_audio_files(args.input)
    if not audio_files:
        logger.error("No audio files found in the input directory")
        sys.exit(1)
        
    # Load model
    model = load_vosk_model(args.model)
    if not model:
        logger.error("Failed to load model")
        sys.exit(1)
        
    # Process each audio file
    success_count = 0
    error_count = 0
    
    for audio_file in audio_files:
        logger.info(f"Processing {audio_file}...")
        try:
            transcription = transcribe_audio_file(audio_file, model)
            
            if transcription:
                if save_transcription(transcription, audio_file, args.output, args.model):
                    success_count += 1
                else:
                    error_count += 1
            else:
                logger.warning(f"No transcription generated for {audio_file}")
                error_count += 1
                
        except Exception as e:
            logger.error(f"Unexpected error processing {audio_file}: {e}")
            traceback.print_exc()
            error_count += 1
            
    # Summary log
    logger.info(f"Transcription completed. Successfully processed {success_count} files. {error_count} files had errors.")
    

if __name__ == "__main__":
    main()