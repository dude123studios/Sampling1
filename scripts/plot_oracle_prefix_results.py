import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_results(results_file, output_file):
    with open(results_file, 'r') as f:
        data = json.load(f)

    plt.figure(figsize=(10, 6))
    
    # Iterate through each model
    for model_name, model_data in data.get('results_per_model', {}).items():
        prefix_data = []
        
        # Extract prefix length and accuracy
        for prefix_len, stats in model_data.get('results_per_prefix', {}).items():
            # Ensure prefix_len is treated as int for sorting
            # JSON keys are always strings
            try:
                p_len = int(prefix_len)
                acc = stats.get('accuracy', 0)
                prefix_data.append((p_len, acc))
            except ValueError:
                continue
        
        # Sort by prefix length
        prefix_data.sort(key=lambda x: x[0])
        
        if not prefix_data:
            print(f"No valid data found for model {model_name}")
            continue
            
        x = [p for p, a in prefix_data]
        y = [a for p, a in prefix_data]
        
        plt.plot(x, y, marker='o', label=model_name)
        
        # Label points
        for i, val in enumerate(y):
            plt.annotate(f"{val:.2f}", (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.xlabel('Prefix Length (Tokens)')
    plt.ylabel('Pass@1 Accuracy')
    plt.title('Effect of Oracle Prefix Length on Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Handle output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Oracle Prefix Experiment Results")
    parser.add_argument("--input", type=str, 
                        default="results/oracle_prefix/oracle_prefix_results_2026-01-14_06-05-21.json",
                        help="Path to results JSON file")
    parser.add_argument("--output", type=str, 
                        default="results/plots/oracle_prefix_accuracy.png",
                        help="Path to output plot image")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found.")
    else:
        plot_results(args.input, args.output)
