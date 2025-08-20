#!/usr/bin/env python3
"""
Test Voxtral with proper mistral_common TranscriptionRequest
"""
import torch
from transformers import VoxtralProcessor, VoxtralForConditionalGeneration
import librosa
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer, TranscriptionRequest
import numpy as np

def try_transcription_request():
    """Try using TranscriptionRequest from mistral_common"""
    print("Loading model and processor...")
    processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Small-24B-2507")
    model = VoxtralForConditionalGeneration.from_pretrained(
        "mistralai/Voxtral-Small-24B-2507",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load audio file
    audio_path = "./golden_samples/00000048-AUDIO-2024-11-27-08-10-48.mp3"
    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sr}")
    
    try:
        # Create transcription request
        request = TranscriptionRequest(audio=audio.tolist())  # Convert numpy to list
        print(f"TranscriptionRequest created: {type(request)}")
        
        # Try to use the processor with the request
        inputs = processor.apply_transcription_request(request, return_tensors="pt")
        print(f"Transcription request inputs keys: {inputs.keys()}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        # Move to device
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        
        # Decode
        transcription = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Transcription: {transcription}")
        
    except Exception as e:
        print(f"TranscriptionRequest approach failed: {e}")
        import traceback
        traceback.print_exc()

def try_audio_only_simple():
    """Try the simplest possible approach - just audio"""
    print("\n=== Testing simplest audio-only approach ===")
    processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Small-24B-2507")
    model = VoxtralForConditionalGeneration.from_pretrained(
        "mistralai/Voxtral-Small-24B-2507",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load audio file
    audio_path = "./golden_samples/00000048-AUDIO-2024-11-27-08-10-48.mp3"
    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    
    try:
        # Try without any text, just audio
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        print(f"Simple audio inputs keys: {inputs.keys()}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        # Move to device
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        # Decode
        transcription = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Transcription: {transcription}")
        
    except Exception as e:
        print(f"Simple audio approach failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try_transcription_request()
    try_audio_only_simple()
