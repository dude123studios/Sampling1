import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel

class HFModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.device = config.get("device_map", "auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=self.device,
            torch_dtype=getattr(torch, config.torch_dtype, torch.float16)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def generate(self, prompt: str, **kwargs):
        """Generate text using manual token-by-token generation to ensure temperature/top_p work correctly."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_ids = inputs['input_ids']
        current_ids = prompt_ids.clone()
        generated_tokens = []
        
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 1.0)
        top_k = kwargs.get('top_k', 0)  # 0 means no top_k filtering
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(current_ids)
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                
                # Handle greedy decoding (temperature = 0)
                if temperature == 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    token_id = next_token.item()
                    generated_tokens.append(token_id)
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    if token_id == self.tokenizer.eos_token_id:
                        break
                    continue
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top_k filtering if specified
                if top_k > 0 and top_k < logits.shape[-1]:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    top_k_mask = torch.zeros_like(logits, dtype=torch.bool)
                    top_k_mask.scatter_(-1, top_k_indices, True)
                    logits = logits.masked_fill(~top_k_mask, float('-inf'))
                
                # Apply top_p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 0] = False  # Keep at least one token
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()
                
                generated_tokens.append(token_id)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Stop if EOS
                if token_id == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated tokens
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded 
