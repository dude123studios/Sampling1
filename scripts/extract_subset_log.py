import json
import argparse
from datetime import datetime

def extract(input_file, output_file, start_time_str, end_time_str):
    print(f"Reading from {input_file}")
    
    # Simple isoformat parsing
    start_dt = datetime.fromisoformat(start_time_str)
    end_dt = datetime.fromisoformat(end_time_str)
    
    kept = 0
    total = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                total += 1
                
                ts_str = data.get("timestamp")
                if not ts_str: continue
                
                # Handle potentially missing milliseconds or different formats?
                # The log showed "2025-12-28T15:32:37.739076"
                try:
                    dt = datetime.fromisoformat(ts_str)
                    
                    if start_dt <= dt <= end_dt:
                        fout.write(line)
                        kept += 1
                except ValueError:
                    continue
                    
            except json.JSONDecodeError:
                pass
                
    print(f"Processed {total} lines.")
    print(f"Extracted {kept} lines to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--start", required=True, help="ISO format start time")
    parser.add_argument("--end", required=True, help="ISO format end time")
    args = parser.parse_args()
    
    extract(args.input_file, args.output_file, args.start, args.end)
