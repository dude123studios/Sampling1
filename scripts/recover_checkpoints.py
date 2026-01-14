import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def recover_checkpoints(results_file):
    results_path = Path(results_file)
    if not results_path.exists():
        log.error(f"File not found: {results_path}")
        return

    output_dir = results_path.parent
    
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    if 'results_per_model' not in data:
        log.error("Invalid results file format")
        return

    for model_name, model_data in data['results_per_model'].items():
        checkpoint_file = output_dir / f"checkpoint_{model_name}.json"
        
        # We need to save it in exactly the format the script expects
        # The script expects 'model_id' and 'results_per_prefix'
        
        # model_data from final file should match this structure
        
        if checkpoint_file.exists():
            log.info(f"Checkpoint for {model_name} already exists. Skipping.")
            continue
            
        with open(checkpoint_file, 'w') as f:
            json.dump(model_data, f, indent=2)
            
        log.info(f"Recovered checkpoint for {model_name} to {checkpoint_file}")

if __name__ == "__main__":
    # Hardcoded to the file we found
    results_file = "results/oracle_prefix/oracle_prefix_results_2026-01-13_13-01-10.json"
    recover_checkpoints(results_file)
