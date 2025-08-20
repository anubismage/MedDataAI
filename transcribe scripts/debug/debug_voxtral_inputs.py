#!/usr/bin/env python3
"""
Debug script to understand Voxtral model input format
"""
import torch
from transformers import VoxtralProcessor, VoxtralForConditionalGeneration
import librosa

def debug_voxtral_inputs():
    """Debug how Voxtral expects inputs to be formatted"""
    print("Loading model and processor...")
    processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Small-24B-2507")
    model = VoxtralForConditionalGeneration.from_pretrained(
        "mistralai/Voxtral-Small-24B-2507",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load a test audio file
    audio_path = "./golden_samples/00000048-AUDIO-2024-11-27-08-10-48.mp3"
    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sr}")
    
    # Try the standard processor approach
    print("\n=== Testing standard processor approach ===")
    try:
        # Text prompt for transcription
        text = "Transcribe this audio:"
        
        # Process inputs together
        inputs = processor(audio=audio, text=text, sampling_rate=sr, return_tensors="pt")
        print(f"Processor inputs keys: {inputs.keys()}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
        
        # Move to device
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Try generation
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        
        # Decode
        transcription = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Transcription: {transcription}")
        
    except Exception as e:
        print(f"Standard processor failed: {e}")
    
    # Try separate processing
    print("\n=== Testing separate processing ===")
    try:
        # Process audio separately
        audio_inputs = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        print(f"Audio inputs keys: {audio_inputs.keys()}")
        for key, value in audio_inputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        # Process text separately
        text = "Transcribe this audio:"
        text_inputs = processor.tokenizer(text, return_tensors="pt")
        print(f"Text inputs keys: {text_inputs.keys()}")
        for key, value in text_inputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        # Try combining manually
        combined_inputs = {}
        combined_inputs.update(audio_inputs)
        combined_inputs.update(text_inputs)
        
        print(f"Combined inputs keys: {combined_inputs.keys()}")
        
        # Move to device
        combined_inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in combined_inputs.items()}
        
        # Try generation
        with torch.no_grad():
            outputs = model.generate(**combined_inputs, max_new_tokens=100)
        
        # Decode
        transcription = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Transcription: {transcription}")
        
    except Exception as e:
        print(f"Separate processing failed: {e}")
    
    # Try looking at model config
    print("\n=== Model config info ===")
    print(f"Model config: {model.config}")
    print(f"Model input names: {model.get_input_embeddings()}")

if __name__ == "__main__":
    debug_voxtral_inputs()
