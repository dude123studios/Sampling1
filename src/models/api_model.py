import requests
import json
import os
import time
from .base import BaseModel

class APIModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        # Support both direct API key and environment variable lookup
        if hasattr(config, 'api_key') and config.api_key:
            self.api_key = config.api_key
        elif hasattr(config, 'api_key_env') and config.api_key_env:
            self.api_key = os.getenv(config.api_key_env)
            if not self.api_key:
                raise ValueError(f"API Key not found in environment variable: {config.api_key_env}")
        else:
            raise ValueError("Either 'api_key' or 'api_key_env' must be specified in config")

        self.base_url = config.base_url
        self.model_name = config.model_name
        
    def generate(self, prompt: str, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://neurips-experiment.com", # Required by OpenRouter
            "X-Title": "Sampling Limits NeurIPS",
            "Content-Type": "application/json"
        }
        
        # Helper to get the token prefix arguments
        prefix = kwargs.get('prefix', None)

        messages = [{"role": "user", "content": prompt}]
        if prefix:
             messages.append({"role": "assistant", "content": prefix})

        # Default params
        n = kwargs.get('n', 1)
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_new_tokens', 4096),
            "top_p": kwargs.get('top_p', 1.0),
            "n": n
        }

        # Add top_k if specified
        if 'top_k' in kwargs:
            data['top_k'] = kwargs['top_k']
        
        retries = 3
        last_error = None
        
        current_timeout = kwargs.get('timeout', 300)

        for i in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=current_timeout
                )
                
                if response.status_code != 200:
                    # Log the full error body for debugging
                    print(f"API Error (Attempt {i+1}): {response.status_code} - {response.text}")
                    response.raise_for_status()
                
                resp_json = response.json()
                if 'choices' not in resp_json:
                    print(f"API Error (Unexpected Format): {resp_json}")
                    raise KeyError("'choices' not found in response")
                
                extracted_contents = []
                for choice in resp_json['choices']:
                    message = choice['message']
                    content = message.get('content', '')
                    reasoning = message.get('reasoning', '')
                    
                    if reasoning:
                        combined_output = f"{reasoning}\n\n{content}"
                        extracted_contents.append(combined_output)
                    else:
                        extracted_contents.append(content if content is not None else "")
                
                if n == 1:
                    return extracted_contents[0]
                return extracted_contents
                
            except Exception as e:
                print(f"Request failed (Attempt {i+1}): {e}")
                last_error = e
                if i < retries - 1:
                    time.sleep(2 ** i)
                    
        raise last_error or Exception("Unknown API error")

    def embed(self, text: str, timeout: int = 30):
        """Get embeddings for text using OpenRouter embeddings API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://neurips-experiment.com",
            "X-Title": "Sampling Limits NeurIPS",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "input": text
        }

        retries = 3
        last_error = None

        current_timeout = timeout

        for i in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=data,
                    timeout=current_timeout
                )

                if response.status_code != 200:
                    print(f"Embedding API Error (Attempt {i+1}): {response.status_code} - {response.text}")
                    response.raise_for_status()

                resp_json = response.json()
                if 'data' not in resp_json or not resp_json['data']:
                    print(f"Embedding API Error (Unexpected Format): {resp_json}")
                    raise KeyError("'data' not found in response")

                # Return the embedding vector
                embedding = resp_json['data'][0]['embedding']
                return embedding

            except Exception as e:
                print(f"Embedding request failed (Attempt {i+1}): {e}")
                last_error = e
                if i < retries - 1:
                    time.sleep(2 ** i)

        raise last_error or Exception("Unknown embedding API error")
