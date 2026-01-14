
import sys
import os
import json
import logging
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.math_grader import grade_math

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def regrade_oracle_solutions(input_path="data/oracle_solutions.json"):
    log.info(f"Loading solutions from {input_path}")
    
    with open(input_path, 'r') as f:
        solutions = json.load(f)
        
    log.info(f"Regrading {len(solutions)} solutions...")
    
    changed_count = 0
    new_correct_count = 0
    
    for item in tqdm(solutions):
        original_correct = item.get('is_correct', False)
        
        # Use existing oracle_solution and gold_answer
        prediction = item.get('oracle_solution', '')
        gold = item.get('gold_answer', '')
        
        new_is_correct = grade_math(prediction, gold)
        
        if new_is_correct != original_correct:
            changed_count += 1
            item['is_correct'] = new_is_correct
            item['grade_changed'] = True
            log.debug(f"ID {item['id']}: Changed from {original_correct} to {new_is_correct}")
        else:
            item['grade_changed'] = False
            
        if new_is_correct:
            new_correct_count += 1
            
    # Save updated solutions
    with open(input_path, 'w') as f:
        json.dump(solutions, f, indent=2)
        
    accuracy = new_correct_count / len(solutions) if solutions else 0
    
    log.info(f"Regrading complete.")
    log.info(f"Total entries changed: {changed_count}")
    log.info(f"New accuracy: {accuracy:.2%} ({new_correct_count}/{len(solutions)})")
    log.info(f"Updated file saved to {input_path}")

if __name__ == "__main__":
    regrade_oracle_solutions()
