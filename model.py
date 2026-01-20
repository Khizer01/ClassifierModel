import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict
import json
import re
import os
from pathlib import Path
from config import get_settings

settings = get_settings()

class AdClassifier:
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.tokenizer = None
        self.device = self.settings.device

    def _configure_gpu_performance(self):
        if not torch.cuda.is_available() or not str(self.device).startswith("cuda"):
            return

        if self.settings.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    
    def _check_model_cached(self) -> bool:
        """Check if model is already cached locally"""
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_id = self.settings.model_name.replace("/", "--")
        model_dir = cache_dir / f"models--{model_id}"
        
        if model_dir.exists():
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                return True
        return False
        
    def load_model(self):
        print(f"Loading model: {self.settings.model_name}")
        
        # Check if model is cached
        if not self._check_model_cached():
            print("\n" + "="*60)
            print("[WARNING] MODEL NOT FOUND IN CACHE")
            print("="*60)
            print("\nThe AI model needs to be downloaded first (~7GB)")
            print("This will take 10-30 minutes on first run")
            print("\nRecommended: Stop the server and run:")
            print("  python test_quick.py")
            print("\nThis will trigger a download and verify the model loads")
            print("="*60 + "\n")
            print("Attempting to download now (may appear stuck)...")
        else:
            print("[OK] Model found in cache, loading...")
        
        use_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")

        if use_cuda:
            self._configure_gpu_performance()

        use_4bit = bool(self.settings.use_4bit_quantization and use_cuda)
        if use_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except Exception as e:
                print(f"[WARNING] bitsandbytes not available, disabling 4-bit quantization: {e}")
                use_4bit = False

        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    attn_implementation="sdpa"
                )
            except Exception as e:
                print(f"[WARNING] Falling back to eager attention due to: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    attn_implementation="eager"
                )
        else:
            torch_dtype = torch.float16 if use_cuda else torch.float32

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa" if use_cuda else "eager"
                )
            except Exception as e:
                if use_cuda:
                    print(f"[WARNING] Falling back to eager attention due to: {e}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.settings.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager"
                    )
                else:
                    raise
            if use_cuda:
                self.model = self.model.to(self.device)

            if use_cuda and self.settings.enable_torch_compile and hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model, mode=self.settings.torch_compile_mode)
                except Exception as e:
                    print(f"[WARNING] torch.compile failed, continuing without compile: {e}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_name,
            trust_remote_code=True
        )

        self.tokenizer.padding_side = "left"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print("Model loaded successfully")
    
    def _create_prompt(self, keyword: str, ad_text: str) -> str:
        prompt = f"""You are an ad classification engine used by a brand intelligence platform.

Your task is to analyze advertisements and social media posts and classify them based on INTENT, not keywords.

IMPORTANT RULES:
- Mentioning a product or keyword alone does NOT make it an ad.
- Personal opinions, stories, or fan posts are NOT promotional ads.
- An ad is promotional ONLY if it is created to market, sell, or promote a product, service, or brand.

You must suggest relevance only if the content appears to be a brand-driven or business-driven advertisement.

Brand Context:
Target Brand: {keyword}

Classification Rules:
1. Mark is_relevant = true ONLY if the content promotes, markets, or advertises a product or service.
2. Mark is_relevant = false for:
   - Personal stories
   - Fan pages
   - Organic mentions
   - News articles
   - Novels
   - Stories
3. Identify the main theme of the ad.

Ad Text:
{ad_text}

Return STRICT JSON ONLY in this exact format:
{{"is_relevant": true, "theme": "Product Promotion"}}"""
        
        return prompt
    
    def _parse_response(self, response: str, ad_id: str) -> Dict:
        try:
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                
                return {
                    "ad_id": str(ad_id),
                    "is_relevant": bool(result.get("is_relevant", False)),
                    "theme": str(result.get("theme", "Unrelated"))
                }
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"Parse error for ad {ad_id}: {e}, Response: {response}")
            return {
                "ad_id": str(ad_id),
                "is_relevant": False,
                "theme": "Unrelated"
            }
    
    def classify_batch(self, keyword: str, ads: List[Dict]) -> List[Dict]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        batch_size = self.settings.batch_size
        
        for i in range(0, len(ads), batch_size):
            batch = ads[i:i + batch_size]
            prompts = []
            ad_ids = []
            
            for ad in batch:
                ad_text = ad.get('body_text', '')
                if not ad_text or len(ad_text.strip()) == 0:
                    results.append({
                        "ad_id": str(ad['id']),
                        "is_relevant": False,
                        "theme": "Unrelated"
                    })
                    continue
                
                # Truncate to ~300 chars (4-5 lines) for faster processing
                ad_text_truncated = ad_text[:300].strip()
                prompt = self._create_prompt(keyword, ad_text_truncated)
                
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                prompts.append(formatted_prompt)
                ad_ids.append(ad['id'])
            
            if not prompts:
                continue
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.settings.max_length
            )

            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.settings.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            for idx, output in enumerate(outputs):
                gen_tokens = output[int(input_lengths[idx]):]
                response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                result = self._parse_response(response, ad_ids[idx])
                results.append(result)
        
        return results
    
    def classify_single(self, keyword: str, ad: Dict) -> Dict:
        return self.classify_batch(keyword, [ad])[0]
