
import os
import sys
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.api_model import APIModel
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()

def test_n_parameter():
    config = DictConfig({
        "name": "qwen3-8b",
        "model_name": "qwen/qwen3-8b",
        "type": "api",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY"
    })

    model = APIModel(config)
    
    print("Testing qwen/qwen3-8b with n=5...")
    try:
        outputs = model.generate("Print hello world in python", n=5, max_new_tokens=10)
        if isinstance(outputs, list):
            print(f"Received list of length: {len(outputs)}")
        else:
            print(f"Received single string: {outputs}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_n_parameter()
