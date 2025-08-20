#!/usr/bin/env python3
"""
Debug script to understand the Phi-4-multimodal model structure
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

def inspect_model_structure():
    """Inspect the model structure to understand the PEFT issue"""
    print("Loading model for inspection...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            _attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        
        print(f"Model type: {type(model)}")
        print(f"Model class: {model.__class__}")
        
        # Check attributes
        print("\nModel attributes:")
        attrs = [attr for attr in dir(model) if not attr.startswith('_')]
        for attr in sorted(attrs):
            if 'generation' in attr.lower() or 'prepare' in attr.lower():
                print(f"  {attr}: {hasattr(model, attr)}")
        
        # Check if model has a base model
        if hasattr(model, 'model'):
            print(f"\nBase model type: {type(model.model)}")
            print(f"Base model class: {model.model.__class__}")
            
            print("\nBase model attributes:")
            base_attrs = [attr for attr in dir(model.model) if not attr.startswith('_')]
            for attr in sorted(base_attrs):
                if 'generation' in attr.lower() or 'prepare' in attr.lower():
                    print(f"  {attr}: {hasattr(model.model, attr)}")
        
        # Check for PEFT wrapper
        if hasattr(model, 'peft_config'):
            print(f"\nPEFT config found: {model.peft_config}")
        
        if hasattr(model, 'base_model'):
            print(f"\nBase model found: {type(model.base_model)}")
        
        # Try to understand the generation chain
        print("\nMethod Resolution Order (MRO):")
        for i, cls in enumerate(model.__class__.__mro__):
            print(f"  {i}: {cls}")
            if hasattr(cls, 'prepare_inputs_for_generation'):
                print(f"    ✅ Has prepare_inputs_for_generation")
        
        print("\nTrying to call prepare_inputs_for_generation...")
        try:
            dummy_ids = torch.tensor([[1, 2, 3]])
            result = model.prepare_inputs_for_generation(dummy_ids)
            print(f"✅ prepare_inputs_for_generation works: {type(result)}")
        except Exception as e:
            print(f"❌ prepare_inputs_for_generation failed: {e}")
        
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Model loading failed: {e}")

def main():
    print("Phi-4-multimodal Model Structure Inspector")
    print("=" * 50)
    inspect_model_structure()

if __name__ == "__main__":
    main()
