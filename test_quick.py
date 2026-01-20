#!/usr/bin/env python3
"""
Quick test script to verify model loading and generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_model():
    print("="*60)
    print("Testing Phi-3.5-mini Model")
    print("="*60)
    
    model_name = "microsoft/Phi-3.5-mini-instruct"
    
    try:
        print("\n1. Loading tokenizer...")
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print(f"   Tokenizer loaded in {time.time()-start:.2f}s")
        
        print("\n2. Loading model...")
        start = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True
        )
        if device == "cuda":
            model = model.to(device)
        model.eval()
        print(f"   Model loaded in {time.time()-start:.2f}s")
        
        print("\n3. Testing simple generation...")
        test_prompt = """You are an ad classifier. Classify if this ad is relevant to Coffee.

Ad text: I love drinking fresh coffee every morning from my new coffee maker.

Output ONLY valid JSON:
{"is_relevant": true, "theme": "Product Promotion"}"""
        
        messages = [{"role": "user", "content": test_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print(f"\n   Prompt length: {len(formatted_prompt)} chars")
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=256)
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"   Input tokens: {inputs['input_ids'].shape[1]}")
        print("   Generating response...")
        
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
        
        generation_time = time.time() - start
        generated_tokens = int(outputs.shape[1] - inputs["input_ids"].shape[1])
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n   Generation completed in {generation_time:.2f}s")
        print(f"\n   Response preview (first 300 chars):")
        print(f"   {response[:300]}...")
        
        print("\n" + "="*60)
        print("SUCCESS! Model is working")
        print("="*60)
        print(f"\nPerformance:")
        print(f"  - Generation time: {generation_time:.2f}s")
        if generation_time > 0:
            print(f"  - Tokens/second: {generated_tokens/generation_time:.1f}")
        
        return True
        
    except Exception as e:
        print(f"\n" + "="*60)
        print("ERROR!")
        print("="*60)
        print(f"\nError: {e}")
        print(f"Type: {type(e).__name__}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    test_model()
