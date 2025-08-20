#!/usr/bin/env python3
"""
Search for proper Voxtral usage examples from transformers documentation
"""
import torch
from transformers import VoxtralProcessor, VoxtralForConditionalGeneration
import librosa
from mistral_common import AudioChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

def try_mistral_common_approach():
    """Try using mistral_common for proper audio handling"""
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
        # Create audio chunk
        audio_chunk = AudioChunk.from_numpy(audio)
        
        # Create user message with audio
        user_message = UserMessage(content=[audio_chunk])
        
        # Create chat completion request
        request = ChatCompletionRequest(messages=[user_message])
        
        print(f"Request created: {request}")
        
        # Try to use the processor with the request
        inputs = processor.apply_chat_template(request, return_tensors="pt")
        print(f"Chat template inputs keys: {inputs.keys()}")
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
        print(f"Mistral common approach failed: {e}")
        import traceback
        traceback.print_exc()

def try_simple_audio_text_format():
    """Try simple audio + text format without mistral_common"""
    print("\n=== Testing simple audio + text format ===")
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
        # Just process audio without text
        inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
        print(f"Audio-only inputs keys: {inputs.keys()}")
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
        print(f"Audio-only approach failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try_mistral_common_approach()
    try_simple_audio_text_format()
