#!/usr/bin/env python3
"""
Alternative transcription script that avoids PEFT issues entirely
"""

import argparse
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
import librosa
from pydub import AudioSegment
from pydub.utils import which

# Completely disable PEFT to avoid compatibility issues
os.environ["DISABLE_PEFT"] = "1"

# Mock PEFT modules to prevent loading
import sys
class MockPEFT:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['peft'] = MockPEFT()
sys.modules['peft.peft_model'] = MockPEFT()
sys.modules['peft.mapping_func'] = MockPEFT()

# Suppress warnings
warnings.filterwarnings("ignore")

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

# Check FFmpeg availability for better audio format support
if not which("ffmpeg"):
    logger.warning("FFmpeg not found. Some audio formats (like m4a) may not be supported.")
else:
    logger.info("FFmpeg found - all audio formats should be supported")

def load_audio_advanced(path_or_url):
    """Advanced audio loading with multiple fallbacks"""
    try:
        audio = None
        sample_rate = 16000
        
        # Handle URLs
        if urlparse(path_or_url).scheme in ("http", "https"):
            data = io.BytesIO(urlopen(path_or_url).read())
            audio, sr = sf.read(data)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            sample_rate = sr
        else:
            # Try pydub first for better format support
            try:
                audio_segment = AudioSegment.from_file(path_or_url)
                
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                audio_segment = audio_segment.set_frame_rate(16000)
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                
                if audio_segment.sample_width == 2:  # 16-bit
                    audio = audio / 32768.0
                elif audio_segment.sample_width == 3:  # 24-bit
                    audio = audio / 8388608.0
                elif audio_segment.sample_width == 4:  # 32-bit
                    audio = audio / 2147483648.0
                else:  # 8-bit or other
                    audio = audio / 128.0 - 1.0
                
                sample_rate = 16000
                logger.debug(f"Loaded audio with pydub: {path_or_url}")
                
            except Exception as pydub_error:
                logger.debug(f"Pydub failed, trying soundfile: {pydub_error}")
                
                try:
                    audio, sample_rate = sf.read(path_or_url)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    logger.debug(f"Loaded audio with soundfile: {path_or_url}")
                    
                except Exception as sf_error:
                    logger.debug(f"Soundfile failed, trying librosa: {sf_error}")
                    
                    audio, sample_rate = librosa.load(path_or_url, sr=16000, mono=True)
                    logger.debug(f"Loaded audio with librosa: {path_or_url}")
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sample_rate != 16000:
            try:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except Exception:
                logger.warning(f"Resampling failed for {path_or_url}")
        
        # Apply 500Hz high pass filter
        try:
            nyquist = sample_rate / 2
            if 500.0 < nyquist:
                b, a = signal.butter(4, 500.0 / nyquist, btype='high', analog=False)
                audio = signal.filtfilt(b, a, audio).astype(np.float32)
                logger.debug("Applied 500Hz high pass filter")
        except Exception:
            logger.warning("High pass filter failed")
        
        # Normalize
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio, int(sample_rate)
        
    except Exception as e:
        logger.error(f"Failed to load audio file {path_or_url}: {str(e)}")
        raise

def transcribe_with_manual_forward(model, processor, audio_path, prompt, device):
    """Use manual forward pass instead of generate() to avoid PEFT issues"""
    try:
        logger.info(f"Transcribing with manual forward: {audio_path}")
        
        # Load audio
        audio, sr = load_audio_advanced(audio_path)
        
        # Prepare inputs
        task_prompt = f"<|user|><|audio_1|>{prompt}<|end|><|assistant|>"
        
        inputs = processor(
            text=task_prompt,
            audios=[(audio, sr)],
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Use a very simple generation approach
        with torch.inference_mode():
            # Try to use the model's forward method directly
            try:
                # Get input_ids from the processor
                input_ids = inputs["input_ids"]
                
                # Simple greedy decoding
                max_new_tokens = 256
                generated_ids = input_ids.clone()
                
                for _ in range(max_new_tokens):
                    # Forward pass
                    with torch.no_grad():
                        outputs = model(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'audios']})
                        logits = outputs.logits[:, -1, :]  # Get last token logits
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        
                        # Check for end token
                        if next_token.item() == processor.tokenizer.eos_token_id:
                            break
                        
                        # Append token
                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                        
                        # Update inputs for next iteration
                        inputs["input_ids"] = generated_ids
                
                # Decode the result
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                if "<|assistant|>" in text:
                    text = text.split("<|assistant|>")[-1].strip()
                
                logger.info(f"Successfully transcribed with manual forward: {audio_path}")
                return text
                
            except Exception as forward_error:
                logger.error(f"Manual forward pass failed: {forward_error}")
                # Fallback to simple processing
                return "Transcription failed - please try with a different model"
        
    except Exception as e:
        logger.error(f"Failed to transcribe {audio_path}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="ASR with Phi-4-multimodal-instruct (PEFT-free version)")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--prompt", default="Transcribe the audio clip into text.", help="Transcription prompt")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {MODEL_ID} on {device} (PEFT-free mode)...")
    
    try:
        # Load model without PEFT
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            _attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        logger.info("Model and processor loaded successfully")
        
        # Transcribe
        result = transcribe_with_manual_forward(model, processor, args.audio, args.prompt, device)
        print(result)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
