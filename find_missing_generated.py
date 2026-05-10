#!/usr/bin/env python3
"""Find indexes with missing generated images."""

import os
from pathlib import Path

def find_missing_generated(directory):
    """Find item indexes that don't have generated images."""
    directory = Path(directory)

    # Get all JSON files
    json_files = sorted(directory.glob("item_*.json"))

    missing = []
    for json_file in json_files:
        base_name = json_file.stem  # e.g., item_00000
        generated_file = directory / f"{base_name}_generated_0.png"

        if not generated_file.exists():
            missing.append(base_name)

    return missing

if __name__ == "__main__":
    valid_dir = "/Users/log/Github/sketchvlm/results/mix_eval/maze_v2/nano_banana/nanob_maze_valid"
    invalid_dir = "/Users/log/Github/sketchvlm/results/mix_eval/maze_v2/nano_banana/nanob_maze_invalid"

    print("=== nanob_maze_valid ===")
    valid_missing = find_missing_generated(valid_dir)
    if valid_missing:
        # Extract numbers from item names (e.g., "item_00012" -> "12")
        numbers = [item.replace("item_", "").lstrip("0") or "0" for item in valid_missing]
        print(",".join(numbers))
    else:
        print("All items have generated images")

    print(f"\n=== nanob_maze_invalid ===")
    invalid_missing = find_missing_generated(invalid_dir)
    if invalid_missing:
        # Extract numbers from item names (e.g., "item_00012" -> "12")
        numbers = [item.replace("item_", "").lstrip("0") or "0" for item in invalid_missing]
        print(",".join(numbers))
    else:
        print("All items have generated images")
