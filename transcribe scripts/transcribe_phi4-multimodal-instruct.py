#!/usr/bin/env python3
"""
Audio Transcription Script using Microsoft Phi-4 Multimodal Instruct Model

This script automatically transcribes all audio files in a specified input folder
using the Microsoft Phi-4 multimodal model from Hugging Face.

Usage:
    python transcribe_phi4-multimodal-instruct.py --input_folder /path/to/audio/files --output_folder /path/to/output
"""

import argparse
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import soundfile as sf
import librosa
import numpy as np
from scipy import signal
from pydub import AudioSegment
from pydub.utils import which
import tempfile

# Disable FlashAttention2 globally
os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["USE_FLASH_ATTENTION"] = "0"

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

class Phi4AudioTranscriber:
    """Class to handle audio transcription using Microsoft Phi-4 multimodal model."""
    
    def __init__(self, model_name: str = "microsoft/Phi-4-multimodal-instruct", force_cpu: bool = False, highpass_freq: Optional[float] = None):
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
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Check FFmpeg availability for better audio format support
        if not which("ffmpeg"):
            logger.warning("FFmpeg not found. Some audio formats (like m4a) may not be supported. Install FFmpeg for better compatibility.")
        
        logger.info(f"Using device: {self.device}")
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
        """Load the Phi-4 multimodal model and processor."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Set environment variables to disable FlashAttention2 globally
            import os
            os.environ['DISABLE_FLASH_ATTENTION_2'] = '1'
            os.environ['TRANSFORMERS_NO_FLASH_ATTENTION_2'] = '1'
            
            # Load processor first
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Phi-4 multimodal has a completely custom architecture
            # We need to import and use the custom model class directly
            try:
                # Import the custom model classes from the correct path with hash
                import sys
                hash_dir = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
                module_path = f"/home/med_data/<user>/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/{hash_dir}"
                if module_path not in sys.path:
                    sys.path.insert(0, module_path)
                
                from configuration_phi4mm import Phi4MMConfig
                from modeling_phi4mm import Phi4MMModel
                
                # Load using the specific model class
                config = Phi4MMConfig.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = Phi4MMModel.from_pretrained(
                    self.model_name,
                    config=config,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if not self.force_cpu else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                if self.force_cpu:
                    self.model.to("cpu")
                
                self.model.eval()
                logger.info(f"Phi-4 multimodal model loaded successfully: {type(self.model).__name__}")
                
                # Check available methods
                model_methods = [method for method in dir(self.model) if not method.startswith('_')]
                key_methods = [m for m in model_methods if any(keyword in m.lower() for keyword in ['forward', 'generate', 'process', 'chat', 'inference'])]
                logger.info(f"Key methods available: {key_methods}")
                
            except ImportError as import_error:
                logger.warning(f"Direct import failed: {import_error}")
                # Try alternative approach - use the downloaded files directly
                try:
                    # Load using trust_remote_code without specific model class
                    from transformers import AutoModelForCausalLM
                    
                    # Try different loading approaches
                    for attempt, approach in enumerate([
                        {"torch_dtype": self.torch_dtype, "trust_remote_code": True, "low_cpu_mem_usage": True, "attn_implementation": "eager"},
                        {"trust_remote_code": True, "attn_implementation": "eager"},
                        {"trust_remote_code": True, "revision": "main", "attn_implementation": "eager"},
                    ]):
                        try:
                            logger.info(f"Attempt {attempt + 1}: Loading with {approach}")
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_name,
                                **approach
                            )
                            if self.force_cpu:
                                self.model.to("cpu")
                            self.model.eval()
                            logger.info("Model loaded successfully with AutoModelForCausalLM")
                            break
                        except Exception as e:
                            logger.warning(f"Attempt {attempt + 1} failed: {e}")
                            if attempt == 2:  # Last attempt
                                raise
                                
                except Exception as final_error:
                    logger.error(f"All loading approaches failed: {final_error}")
                    
                    # Try one more approach with explicit FlashAttention2 disabling
                    try:
                        logger.info("Final attempt: Disabling FlashAttention2 explicitly")
                        
                        # Set environment variables to disable FlashAttention2
                        os.environ['DISABLE_FLASH_ATTENTION_2'] = '1'
                        os.environ['TRANSFORMERS_NO_FLASH_ATTENTION_2'] = '1'
                        
                        # Try loading with explicit attention implementation
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            torch_dtype=self.torch_dtype,
                            low_cpu_mem_usage=True,
                            attn_implementation="eager"
                        )
                        
                        if self.force_cpu:
                            self.model.to("cpu")
                        self.model.eval()
                        logger.info("Model loaded successfully with explicit FlashAttention2 disabled")
                        
                    except Exception as last_error:
                        logger.error(f"Final loading attempt failed: {last_error}")
                        raise
                
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
    
    def chunk_audio(self, audio: np.ndarray, chunk_length_s: int = 30, sample_rate: int = 16000) -> List[np.ndarray]:
        """
        Split audio into chunks for processing.
        
        Args:
            audio: Audio data as numpy array
            chunk_length_s: Length of each chunk in seconds
            sample_rate: Sample rate of the audio
            
        Returns:
            List of audio chunks
        """
        chunk_length_samples = chunk_length_s * sample_rate
        chunks = []
        
        for i in range(0, len(audio), chunk_length_samples):
            chunk = audio[i:i + chunk_length_samples]
            chunks.append(chunk)
        
        logger.debug(f"Split audio into {len(chunks)} chunks")
        return chunks
    
    def transcribe_audio_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribe a single audio chunk using Phi-4 multimodal model.
        
        Args:
            audio_chunk: Audio chunk as numpy array
            
        Returns:
            Transcribed text
        """
        try:
            # Create a transcription prompt for the multimodal model
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Please transcribe this audio to text. Only provide the transcription without any additional commentary."},
                        {"type": "audio", "audio": audio_chunk}
                    ]
                }
            ]
            
            # Process the input
            try:
                # Try processing with the chat template
                inputs = self.processor.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_tensors="pt",
                    return_dict=True
                )
                
                # Move inputs to device
                if hasattr(inputs, 'items'):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                elif torch.is_tensor(inputs):
                    inputs = inputs.to(self.device)
                
            except Exception as process_error:
                logger.debug(f"Chat template processing failed: {process_error}")
                # Fallback to simple text + audio processing
                try:
                    inputs = self.processor(
                        text="Transcribe this audio:",
                        audio=audio_chunk,
                        sampling_rate=16000,
                        return_tensors="pt"
                    )
                    if hasattr(inputs, 'items'):
                        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    
                except Exception as fallback_error:
                    logger.error(f"All processing methods failed: {fallback_error}")
                    return f"Processing failed: {fallback_error}"
            
            # Generate transcription
            with torch.no_grad():
                try:
                    # Use forward pass to get model outputs
                    outputs = self.model(**inputs)
                    
                    # Handle different output formats
                    if hasattr(outputs, 'logits'):
                        # Standard logits output - decode the highest probability tokens
                        logits = outputs.logits
                        if logits.dim() == 3:  # [batch, seq_len, vocab]
                            predicted_ids = torch.argmax(logits, dim=-1)
                            # Get the last sequence (after the prompt)
                            if 'input_ids' in inputs:
                                # Decode only the new tokens (after input)
                                input_length = inputs['input_ids'].shape[1] if inputs['input_ids'].dim() > 1 else len(inputs['input_ids'])
                                new_tokens = predicted_ids[0][input_length:]
                                transcription = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
                            else:
                                transcription = self.processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                        else:
                            transcription = self.processor.tokenizer.decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
                    
                    elif hasattr(outputs, 'prediction_logits'):
                        # Some models have different output names
                        predicted_ids = torch.argmax(outputs.prediction_logits, dim=-1)
                        transcription = self.processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                    
                    elif hasattr(outputs, 'last_hidden_state'):
                        # Hidden state output - this is trickier to handle
                        logger.warning("Model returned hidden states instead of logits. Cannot directly transcribe.")
                        transcription = "[Audio processed but transcription unavailable - hidden state output]"
                    
                    else:
                        # Unknown output format
                        logger.warning(f"Unknown model output format: {type(outputs)}")
                        if hasattr(outputs, '__dict__'):
                            available_attrs = [attr for attr in dir(outputs) if not attr.startswith('_')]
                            logger.info(f"Available output attributes: {available_attrs}")
                        transcription = "[Unknown model output format]"
                    
                except Exception as forward_error:
                    logger.error(f"Forward pass failed: {forward_error}")
                    transcription = f"Forward pass failed: {forward_error}"
            
            # Clean up the transcription
            if isinstance(transcription, str):
                # Remove common prompt patterns
                patterns_to_remove = [
                    "Transcribe this audio:",
                    "Please transcribe this audio to text.",
                    "Only provide the transcription without any additional commentary.",
                    "Transcription:",
                    "Text:",
                ]
                
                for pattern in patterns_to_remove:
                    if pattern in transcription:
                        transcription = transcription.replace(pattern, "").strip()
                
                # Remove extra whitespace
                transcription = " ".join(transcription.split())
            
            return transcription if transcription else "[No transcription generated]"
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio chunk: {str(e)}")
            return f"Transcription failed: {str(e)}"
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe a single audio file with support for long-form audio.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing: {audio_path}")
            
            # Load audio for processing
            audio = self.load_audio(audio_path)
            
            # Split audio into chunks if it's too long (more than 30 seconds)
            if len(audio) > 30 * 16000:  # 30 seconds at 16kHz
                audio_chunks = self.chunk_audio(audio, chunk_length_s=30)
                transcriptions = []
                
                for i, chunk in enumerate(audio_chunks):
                    logger.debug(f"Processing chunk {i+1}/{len(audio_chunks)}")
                    chunk_transcription = self.transcribe_audio_chunk(chunk)
                    transcriptions.append(chunk_transcription)
                
                # Combine all transcriptions
                full_transcription = " ".join(transcriptions)
            else:
                # Process the entire audio if it's short enough
                full_transcription = self.transcribe_audio_chunk(audio)
            
            logger.info(f"Successfully transcribed: {audio_path}")
            return full_transcription.strip()
            
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
                model_name_short = "phi4-multimodal"
                
                # Include highpass info in filename if filter is applied
                if self.highpass_freq is not None:
                    output_filename = f"{audio_path.stem}_{model_name_short}_hp{int(self.highpass_freq)}.txt"
                else:
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
        description="Transcribe audio files using Microsoft Phi-4 multimodal model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python transcribe_phi4-multimodal-instruct.py --input_folder ./audio --output_folder ./transcriptions
    python transcribe_phi4-multimodal-instruct.py --input_folder ./audio --output_folder ./transcriptions --force_cpu
    python transcribe_phi4-multimodal-instruct.py --input_folder ./audio --output_folder ./transcriptions --highpass_freq 300
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
        transcriber = Phi4AudioTranscriber(
            force_cpu=args.force_cpu,
            highpass_freq=args.highpass_freq
        )
        
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
