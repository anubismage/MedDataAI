#!/usr/bin/env python3
"""
Fixed transcription script with PEFT compatibility patches
"""

import argparse, io
import os
import sys
import logging
from pathlib import Path
from typing import List
import warnings
import torch
import soundfile as sf
import numpy as np
from scipy import signal
from urllib.parse import urlparse
from urllib.request import urlopen

# Suppress warnings that might interfere with model loading
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointImpl.*")

# PEFT compatibility patch
def patch_peft_model():
    """Patch PEFT model to add missing methods"""
    try:
        import peft
        from peft.peft_model import PeftModel
        
        # Add the missing method to PeftModel if it doesn't exist
        if not hasattr(PeftModel, 'prepare_inputs_for_generation'):
            def prepare_inputs_for_generation(self, input_ids, **kwargs):
                # If the base model has the method, use it
                if hasattr(self.base_model, 'prepare_inputs_for_generation'):
                    return self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
                # Otherwise, return a basic implementation
                return {"input_ids": input_ids, **kwargs}
            
            PeftModel.prepare_inputs_for_generation = prepare_inputs_for_generation
            print("✅ Applied PEFT compatibility patch")
        
    except Exception as e:
        print(f"⚠️ Could not apply PEFT patch: {e}")

# Apply the patch before importing transformers
patch_peft_model()

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

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

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

def apply_highpass_filter(audio: np.ndarray, sample_rate: int = 16000, cutoff_freq: float = 500.0) -> np.ndarray:
    """
    Apply a high pass filter to the audio signal.
    
    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        cutoff_freq: High pass filter frequency in Hz (default: 500Hz)
        
    Returns:
        Filtered audio signal
    """
    try:
        # Ensure we don't filter above Nyquist frequency
        nyquist = sample_rate / 2
        if cutoff_freq >= nyquist:
            logger.warning(f"High pass frequency ({cutoff_freq} Hz) is above Nyquist frequency ({nyquist} Hz). Skipping filtering.")
            return audio
        
        # Design a high pass Butterworth filter
        # Using 4th order for good roll-off characteristics
        order = 4
        normalized_freq = cutoff_freq / nyquist
        
        # Create the filter coefficients
        b, a = signal.butter(order, normalized_freq, btype='high', analog=False)
        
        # Apply the filter using filtfilt for zero-phase filtering
        filtered_audio = signal.filtfilt(b, a, audio)
        
        logger.debug(f"Applied high pass filter at {cutoff_freq} Hz")
        return filtered_audio.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Failed to apply high pass filter: {str(e)}")
        logger.warning("Continuing without filtering")
        return audio

def load_audio(path_or_url):
    """Load audio from local path or URL and apply high pass filter."""
    if urlparse(path_or_url).scheme in ("http", "https"):
        data = io.BytesIO(urlopen(path_or_url).read())
        audio, sr = sf.read(data)
    else:
        audio, sr = sf.read(path_or_url)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Apply 500Hz high pass filter
    audio = apply_highpass_filter(audio, sr, cutoff_freq=500.0)
    
    return audio, int(sr)

def get_audio_files(input_folder: str) -> List[str]:
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

def load_model_with_patches():
    """Load the model with all necessary patches and fallbacks"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} (eager attention with PEFT patches)...")
    
    # Try multiple loading strategies
    loading_configs = [
        {
            "name": "PEFT patched loading",
            "config": {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                "_attn_implementation": "eager",
                "low_cpu_mem_usage": True,
            }
        },
        {
            "name": "Basic eager loading",
            "config": {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,
                "_attn_implementation": "eager",
            }
        },
        {
            "name": "Minimal loading",
            "config": {
                "trust_remote_code": True,
                "_attn_implementation": "eager",
            }
        }
    ]
    
    model = None
    for i, loading_config in enumerate(loading_configs, 1):
        try:
            logger.info(f"Attempt {i}: {loading_config['name']}")
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **loading_config['config'])
            
            # Additional patch for the loaded model if needed
            if hasattr(model, 'model') and not hasattr(model, 'prepare_inputs_for_generation'):
                def prepare_inputs_for_generation(self, input_ids, **kwargs):
                    return {"input_ids": input_ids, **kwargs}
                model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model, model.__class__)
            
            model = model.to(device)
            logger.info(f"✅ Successfully loaded model with {loading_config['name']}")
            break
            
        except Exception as e:
            logger.error(f"❌ {loading_config['name']} failed: {str(e)}")
            if i == len(loading_configs):
                raise Exception(f"All model loading methods failed. Last error: {str(e)}")
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        logger.info("✅ Processor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load processor: {str(e)}")
        raise
    
    return model, processor

def transcribe_audio_file(model, processor, audio_path: str, prompt: str, gen_cfg, device) -> str:
    """
    Transcribe a single audio file.
    """
    try:
        logger.info(f"Transcribing: {audio_path}")
        
        # Model expects this exact chat-style speech prompt format:
        task_prompt = f"<|user|><|audio_1|>{prompt}<|end|><|assistant|>"
        
        audio, sr = load_audio(audio_path)
        
        inputs = processor(
            text=task_prompt,
            audios=[(audio, sr)],
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, generation_config=gen_cfg)
        
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>")[-1].strip()
        
        logger.info(f"Successfully transcribed: {audio_path}")
        return text
        
    except Exception as e:
        logger.error(f"Failed to transcribe {audio_path}: {str(e)}")
        raise

def process_folder(input_folder: str, output_folder: str, model, processor, prompt: str, gen_cfg, device):
    """
    Process all audio files in the input folder and save transcriptions.
    """
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of audio files
    audio_files = get_audio_files(input_folder)
    
    if not audio_files:
        logger.warning("No audio files found in the input folder")
        return
    
    # Process each audio file
    successful_transcriptions = 0
    failed_transcriptions = 0
    
    for audio_file in audio_files:
        try:
            # Generate output filename
            audio_path = Path(audio_file)
            output_filename = f"{audio_path.stem}_phi4_hp500.txt"
            output_file = output_path / output_filename
            
            # Skip if output file already exists
            if output_file.exists():
                logger.info(f"Skipping {audio_file} - output already exists")
                continue
            
            # Transcribe audio
            transcription = transcribe_audio_file(model, processor, audio_file, prompt, gen_cfg, device)
            
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
    parser = argparse.ArgumentParser(description="ASR with Phi-4-multimodal-instruct (PEFT-patched)")
    
    # Add mutually exclusive group for single file or folder processing
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", help="Path or URL to a single audio file")
    input_group.add_argument("--input_folder", help="Path to folder containing audio files")
    
    parser.add_argument("--output_folder", help="Path to output folder (required when using --input_folder)")
    parser.add_argument("--prompt", default="Transcribe the audio clip into text.",
                        help="Instruction after the <|audio_1|> token")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_folder and not args.output_folder:
        parser.error("--output_folder is required when using --input_folder")

    try:
        # Load model and processor with patches
        model, processor = load_model_with_patches()
        
        device = next(model.parameters()).device
        
        gen_cfg = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=False if args.temperature == 0 else True,
        )

        if args.audio:
            # Process single audio file
            try:
                transcription = transcribe_audio_file(model, processor, args.audio, args.prompt, gen_cfg, device)
                print(transcription)
            except Exception as e:
                logger.error(f"Failed to process audio file: {str(e)}")
                sys.exit(1)
        
        elif args.input_folder:
            # Process folder of audio files
            try:
                process_folder(args.input_folder, args.output_folder, model, processor, args.prompt, gen_cfg, device)
                logger.info("Transcription process completed successfully")
            except Exception as e:
                logger.error(f"Failed to process folder: {str(e)}")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
