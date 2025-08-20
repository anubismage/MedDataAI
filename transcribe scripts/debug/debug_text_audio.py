#!/usr/bin/env python3
"""
Test Voxtral with proper text + audio combination
"""
import torch
from transformers import VoxtralProcessor, VoxtralForConditionalGeneration
import librosa

def try_text_audio_combination():
    """Try providing both text and audio properly"""
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
    print(f"Audio token: '{processor.audio_token}'")
    
    try:
        # Text with audio token placeholder
        text = f"Transcribe: {processor.audio_token}"
        print(f"Using text: '{text}'")
        
        # Process together
        inputs = processor(text=text, audio=audio, sampling_rate=sr, return_tensors="pt")
        print(f"Combined inputs keys: {inputs.keys()}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
        
        # Move to device
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        # Decode
        transcription = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Transcription: {transcription}")
        
    except Exception as e:
        print(f"Text+audio approach failed: {e}")
        import traceback
        traceback.print_exc()

def try_just_audio_token():
    """Try with just the audio token"""
    print("\n=== Testing with just audio token ===")
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
        # Just the audio token
        text = processor.audio_token
        print(f"Using text: '{text}'")
        
        # Process together
        inputs = processor(text=text, audio=audio, sampling_rate=sr, return_tensors="pt")
        print(f"Audio token inputs keys: {inputs.keys()}")
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
        print(f"Audio token approach failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try_text_audio_combination()
    try_just_audio_token()
