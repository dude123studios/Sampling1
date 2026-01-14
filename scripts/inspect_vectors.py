import torch
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_vectors.py <path_to_vector.pt>")
        sys.exit(1)
        
    path = sys.argv[1]
    try:
        vec = torch.load(path)
        print(f"Vector loaded from {path}")
        print(f"Shape: {vec.shape}")
        print(f"Stats: Mean={vec.mean()}, Std={vec.std()}")
    except Exception as e:
        print(f"Error loading vector: {e}")

if __name__ == "__main__":
    main()
