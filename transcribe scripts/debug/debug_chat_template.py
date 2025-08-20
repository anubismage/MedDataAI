#!/usr/bin/env python3
"""
Test Voxtral with proper apply_chat_template method
"""
import torch
from transformers import VoxtralProcessor, VoxtralForConditionalGeneration
import librosa

def try_chat_template_approach():
    """Try using apply_chat_template with proper message format"""
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
        # Format messages for chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Please transcribe this audio."}
                ]
            }
        ]
        
        print(f"Messages format: {[{k: v if k != 'content' else 'content with audio+text' for k, v in msg.items()} for msg in messages]}")
        
        # Apply chat template
        inputs = processor.apply_chat_template(
            messages, 
            sampling_rate=sr,
            return_tensors="pt"
        )
        
        print(f"Chat template inputs keys: {inputs.keys()}")
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
        print(f"Chat template approach failed: {e}")
        import traceback
        traceback.print_exc()

def try_simple_transcription_message():
    """Try with just audio and no text instruction"""
    print("\n=== Testing simple transcription message ===")
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
        # Just audio content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio}
                ]
            }
        ]
        
        print(f"Simple messages format: audio only")
        
        # Apply chat template
        inputs = processor.apply_chat_template(
            messages, 
            sampling_rate=sr,
            return_tensors="pt"
        )
        
        print(f"Simple chat template inputs keys: {inputs.keys()}")
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
        print(f"Simple transcription message failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try_chat_template_approach()
    try_simple_transcription_message()
