#!/usr/bin/env python3
"""
Debug script to test Voxtral model functionality
"""

import librosa
import numpy as np
from transformers import VoxtralProcessor, VoxtralForConditionalGeneration
from transformers.audio_utils import load_audio_as
import torch
import base64
import io
import soundfile as sf

def test_voxtral_methods():
    print("Loading Voxtral processor and model...")
    processor = VoxtralProcessor.from_pretrained('mistralai/Voxtral-Small-24B-2507')
    
    # Load a test audio file
    audio_file = 'golden_samples/00000048-AUDIO-2024-11-27-08-10-48.mp3'
    
    # Test 1: Chat template with file path
    print("\n=== Testing Chat Template with File Path ===")
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_file},
                    {"type": "text", "text": "Transcribe this audio."},
                ],
            },
        ]
        
        result = processor.apply_chat_template(
            conversation,
            return_tensors='pt'
        )
        print("Chat template with path success!")
        print(f"Result keys: {result.keys()}")
        for k, v in result.items():
            print(f"  {k}: {type(v)}, shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")
            
    except Exception as e:
        print(f"Chat template with path error: {e}")
    
    # Test 2: Chat template with base64 encoded audio
    print("\n=== Testing Chat Template with Base64 Audio ===")
    try:
        # Load audio and convert to base64
        audio_base64 = load_audio_as(audio_file, return_format="base64", force_mono=True)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "base64": audio_base64},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            },
        ]
        
        result = processor.apply_chat_template(
            conversation,
            return_tensors='pt'
        )
        print("Chat template with base64 success!")
        print(f"Result keys: {result.keys()}")
        for k, v in result.items():
            print(f"  {k}: {type(v)}, shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")
            
    except Exception as e:
        print(f"Chat template with base64 error: {e}")
    
    # Test 3: Apply transcription request with file path
    print("\n=== Testing Apply Transcription Request with Path ===")
    try:
        result = processor.apply_transcription_request(
            language='en',
            audio=audio_file,  # Use file path instead of numpy array
            model_id='mistralai/Voxtral-Small-24B-2507'
        )
        print("Transcription request with path success!" if result else "Transcription request returned None")
        if result:
            print(f"Result keys: {result.keys()}")
            for k, v in result.items():
                print(f"  {k}: {type(v)}, shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")
                
    except Exception as e:
        print(f"Transcription request with path error: {e}")

if __name__ == "__main__":
    test_voxtral_methods()
