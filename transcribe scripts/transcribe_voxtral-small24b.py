#!/usr/bin/env python3
"""
Audio Transcription Script using Mistral Voxtral-Small-24B-2507

This script automatically transcribes all audio files in a specified input folder
using the Mistral Voxtral-Small-24B-2507 model from Hugging Face.

Usage:
    python transcribe_voxtral-small24b.py --input_folder /path/to/audio/files --output_folder /path/to/output
"""

import argparse
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import List, Optional
import torch
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor
import soundfile as sf
import librosa
import numpy as np
from scipy import signal
from pydub import AudioSegment
from pydub.utils import which
import tempfile

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voxtral_transcription.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VoxtralTranscriber:
    """Class to handle audio transcription using Voxtral-Small-24B-2507 model."""
    
    def __init__(self, model_name: str = "mistralai/Voxtral-Small-24B-2507", force_cpu: bool = False, highpass_freq: Optional[float] = None):
        """
        Initialize the transcriber with the specified model.
        
        Args:
            model_name: Hugging Face model identifier
            force_cpu: Force CPU usage even if CUDA is available
            highpass_freq: High pass filter frequency in Hz (default: None, no filtering)
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.force_cpu = force_cpu
        self.highpass_freq = highpass_freq
        self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        # Check FFmpeg availability for better audio format support
        if not which("ffmpeg"):
            logger.warning("FFmpeg not found. Some audio formats (like m4a) may not be supported. Install FFmpeg for better compatibility.")
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using dtype: {self.torch_dtype}")
        if highpass_freq is not None:
            logger.info(f"High pass filter frequency: {highpass_freq} Hz")
        if force_cpu and torch.cuda.is_available():
            logger.info("CUDA available but forcing CPU usage")
    
    def apply_highpass_filter(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Apply a high pass filter to the audio signal.
        
        Args:
            audio: Audio signal as numpy array
            sample_rate: Sample rate of the audio (default: 16000)
            
        Returns:
            Filtered audio signal
        """
        if self.highpass_freq is None or self.highpass_freq <= 0:
            return audio
        
        try:
            # Ensure we don't filter above Nyquist frequency
            nyquist = sample_rate / 2
            if self.highpass_freq >= nyquist:
                logger.warning(f"High pass frequency ({self.highpass_freq} Hz) is above Nyquist frequency ({nyquist} Hz). Skipping filtering.")
                return audio
            
            # Design a high pass Butterworth filter
            # Using 4th order for good roll-off characteristics
            order = 4
            normalized_freq = self.highpass_freq / nyquist
            
            # Create the filter coefficients
            b, a = signal.butter(order, normalized_freq, btype='high', analog=False)
            
            # Apply the filter using filtfilt for zero-phase filtering
            filtered_audio = signal.filtfilt(b, a, audio)
            
            logger.debug(f"Applied high pass filter at {self.highpass_freq} Hz")
            return filtered_audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to apply high pass filter: {str(e)}")
            logger.warning("Continuing without filtering")
            return audio

    def load_model(self):
        """Load the Voxtral model and processor."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Clear GPU cache before loading
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     logger.info(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            # Load processor
            self.processor = VoxtralProcessor.from_pretrained(self.model_name)
            
            # Load model with appropriate settings and memory optimization
            if self.device == "cuda":
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    self.model_name, 
                    torch_dtype=self.torch_dtype, 
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    #max_memory={0: "30GiB"}  # Limit GPU 0 to 30GB
                )
            else:
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    self.model_name, 
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            # if torch.cuda.is_available():
            #     logger.info(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file with better format support.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Audio data as numpy array
        """
        try:
            audio = None
            sample_rate = 16000
            
            # First, try with pydub for better format support (especially m4a)
            try:
                # Use pydub to load the audio file (supports more formats including m4a)
                audio_segment = AudioSegment.from_file(audio_path)
                
                # Convert to mono if stereo
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Convert to 16kHz sample rate
                audio_segment = audio_segment.set_frame_rate(16000)
                
                # Convert to numpy array
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                
                # Normalize to [-1, 1] range
                if audio_segment.sample_width == 2:  # 16-bit
                    audio = audio / 32768.0
                elif audio_segment.sample_width == 3:  # 24-bit
                    audio = audio / 8388608.0
                elif audio_segment.sample_width == 4:  # 32-bit
                    audio = audio / 2147483648.0
                else:  # 8-bit or other
                    audio = audio / 128.0 - 1.0
                
                sample_rate = 16000
                logger.debug(f"Loaded audio with pydub: {audio_path}")
                
            except Exception as pydub_error:
                logger.debug(f"Pydub failed for {audio_path}: {pydub_error}")
                
                # Fallback to soundfile
                try:
                    audio, sample_rate = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)  # Convert stereo to mono
                    logger.debug(f"Loaded audio with soundfile: {audio_path}")
                    
                except Exception as sf_error:
                    logger.debug(f"Soundfile failed for {audio_path}: {sf_error}")
                    
                    # Final fallback to librosa
                    try:
                        # Use librosa's modern loading approach to avoid deprecation warnings
                        audio, sample_rate = librosa.load(
                            audio_path, 
                            sr=16000,
                            mono=True  # Ensure mono output
                        )
                        logger.debug(f"Loaded audio with librosa: {audio_path}")
                        
                    except Exception as librosa_error:
                        logger.error(f"All audio loading methods failed for {audio_path}")
                        logger.error(f"Pydub error: {pydub_error}")
                        logger.error(f"Soundfile error: {sf_error}")
                        logger.error(f"Librosa error: {librosa_error}")
                        raise Exception(f"Failed to load audio file: {audio_path}")
            
            # Ensure audio is mono and correct shape
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to 16kHz if necessary (only if not already 16kHz)
            if sample_rate != 16000:
                try:
                    audio = librosa.resample(
                        audio, 
                        orig_sr=sample_rate, 
                        target_sr=16000
                    )
                    sample_rate = 16000
                except Exception as resample_error:
                    logger.warning(f"Resampling failed for {audio_path}: {resample_error}")
                    # Continue with original sample rate if resampling fails
            
            # Apply high pass filter if specified
            audio = self.apply_highpass_filter(audio, sample_rate)
            
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
        Transcribe a single audio file using Voxtral model.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing: {audio_path}")
            
            # Check if we need to convert the audio format for Voxtral processor
            file_extension = Path(audio_path).suffix.lower()
            temp_file_path = None
            working_audio_path = audio_path
            
            # For M4A and other potentially problematic formats, convert to WAV using pydub directly
            if file_extension in ['.m4a', '.aac', '.ogg', '.wma', '.aiff', '.au', '.opus', '.webm']:
                try:
                    logger.debug(f"Converting {file_extension} file to WAV format")
                    
                    # Use pydub directly for conversion (more reliable for M4A)
                    audio_segment = AudioSegment.from_file(audio_path)
                    
                    # Convert to mono if stereo
                    if audio_segment.channels > 1:
                        audio_segment = audio_segment.set_channels(1)
                    
                    # Convert to 16kHz sample rate
                    audio_segment = audio_segment.set_frame_rate(16000)
                    
                    # Create temporary WAV file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_file_path = temp_file.name
                    temp_file.close()
                    
                    # Export as WAV
                    audio_segment.export(temp_file_path, format="wav")
                    working_audio_path = temp_file_path
                    logger.debug(f"Successfully converted {audio_path} to temporary WAV: {temp_file_path}")
                    
                except Exception as conversion_error:
                    logger.error(f"Failed to convert {audio_path} to WAV: {conversion_error}")
                    # Don't fall back to original path for M4A - skip this file instead
                    if file_extension == '.m4a':
                        raise Exception(f"M4A conversion failed and format not supported directly: {conversion_error}")
                    # For other formats, try the original path
                    working_audio_path = audio_path
            
            try:
                # Create conversation with audio using chat template approach
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "path": working_audio_path},
                            {"type": "text", "text": "Please transcribe this audio."}
                        ]
                    }
                ]
                
                # Apply chat template with error handling for corrupted audio
                try:
                    inputs = self.processor.apply_chat_template(
                        conversation, 
                        return_tensors="pt"
                    )
                except Exception as chat_error:
                    # If chat template fails due to audio issues, try to re-convert the file
                    if file_extension in ['.mp3', '.wav'] and temp_file_path is None:
                        logger.warning(f"Audio processing failed for {audio_path}, attempting conversion: {chat_error}")
                        try:
                            # Convert problematic MP3/WAV to clean WAV
                            audio_segment = AudioSegment.from_file(audio_path)
                            if audio_segment.channels > 1:
                                audio_segment = audio_segment.set_channels(1)
                            audio_segment = audio_segment.set_frame_rate(16000)
                            
                            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                            temp_file_path = temp_file.name
                            temp_file.close()
                            
                            audio_segment.export(temp_file_path, format="wav")
                            working_audio_path = temp_file_path
                            logger.debug(f"Re-converted {audio_path} to clean WAV: {temp_file_path}")
                            
                            # Retry with converted file
                            conversation[0]["content"][0]["path"] = working_audio_path
                            inputs = self.processor.apply_chat_template(
                                conversation, 
                                return_tensors="pt"
                            )
                        except Exception as retry_error:
                            logger.error(f"Failed to recover from audio error for {audio_path}: {retry_error}")
                            raise Exception(f"Audio file appears to be corrupted and cannot be processed: {chat_error}")
                    else:
                        raise Exception(f"Failed to process audio: {chat_error}")
                
                # Move inputs to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                
                # Generate transcription
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                # Decode the generated text (skip the input portion)
                input_length = inputs['input_ids'].shape[1]
                response_ids = generated_ids[0][input_length:]
                transcription = self.processor.tokenizer.decode(
                    response_ids, 
                    skip_special_tokens=True
                )
                
                # Clear GPU cache after generation
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
                
                logger.info(f"Successfully transcribed: {audio_path}")
                return transcription.strip()
                
            finally:
                # Clean up temporary file if created
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                        logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary file {temp_file_path}: {cleanup_error}")
            
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
                model_name_short = "voxtral-small-24b"
                
                # Include highpass info in filename if filter is applied
                if self.highpass_freq is not None:
                    output_filename = f"{audio_path.stem}_{model_name_short}_hp{int(self.highpass_freq)}.txt"
                else:
                    output_filename = f"{audio_path.stem}_{model_name_short}.txt"
                
                output_file = output_path / output_filename
                
                # Skip if output file already exists
                if output_file.exists():
                    logger.info(f"Skipping {audio_file} - output already exists")
                    successful_transcriptions += 1  # Count as successful since it's already done
                    continue
                
                # Transcribe audio with error handling
                try:
                    transcription = self.transcribe_audio(audio_file)
                    
                    # Save transcription
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(transcription)
                    
                    logger.info(f"Saved transcription to: {output_file}")
                    successful_transcriptions += 1
                    
                except Exception as transcription_error:
                    logger.error(f"Failed to transcribe {audio_file}: {str(transcription_error)}")
                    failed_transcriptions += 1
                    
                    # Create error file to track failed transcriptions
                    error_file = output_path / f"{audio_path.stem}_{model_name_short}_ERROR.txt"
                    with open(error_file, 'w', encoding='utf-8') as f:
                        f.write(f"Transcription failed: {str(transcription_error)}\n")
                        f.write(f"File: {audio_file}\n")
                        import datetime
                        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    continue  # Continue with next file
                
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {str(e)}")
                failed_transcriptions += 1
                continue
        
        logger.info(f"Processing complete. Successful: {successful_transcriptions}, Failed: {failed_transcriptions}")

def main():
    """Main function to handle command-line arguments and run transcription."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Mistral Voxtral-Small-24B-2507",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python transcribe_voxtral-small24b.py --input_folder ./golden_samples --output_folder ./out_voxtral
    python transcribe_voxtral-small24b.py --input_folder ./golden_samples --output_folder ./out_voxtral --force_cpu
    python transcribe_voxtral-small24b.py --input_folder ./golden_samples --output_folder ./out_voxtral --highpass_freq 300
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
        "--force_cpu",
        action="store_true",
        help="Force CPU usage even if CUDA is available (useful for debugging GPU issues)"
    )
    
    parser.add_argument(
        "--highpass_freq",
        type=float,
        default=None,
        help="Apply high pass filter with specified frequency in Hz (e.g., 300 for 300Hz). Default: None (no filtering)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize transcriber
        transcriber = VoxtralTranscriber(
            force_cpu=args.force_cpu,
            highpass_freq=args.highpass_freq
        )
        
        # Process the folder
        transcriber.process_folder(args.input_folder, args.output_folder)
        
        logger.info("Voxtral transcription process completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Transcription process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()