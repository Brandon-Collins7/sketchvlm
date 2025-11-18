#!/usr/bin/env python3
"""
Compile all JSON response files from a directory into a CSV file.
"""

import os
import re
import csv
import json
from pathlib import Path
from typing import Optional

# Same regex patterns from gpt_baseline_ball_drop.py
_BOXED_RE = re.compile(r"\$\\boxed\{s*(1|2|3|4|none)\s*\}\$", re.IGNORECASE)
_LOOSE_RE = re.compile(r"\\boxed\{\s*(1|2|3|4|none)\s*\}", re.IGNORECASE)
_FALLBACK_RE = re.compile(r"\b(1|2|3|4|none)\b", re.IGNORECASE)

def parse_bucket(text: Optional[str]) -> Optional[str]:
    """Parse bucket number from text."""
    if not text:
        return None

    # Try strict $\boxed{} format first
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1)

    # Try without $ signs
    m = _LOOSE_RE.search(text)
    if m:
        return m.group(1)

    # Fallback to any digit
    m = _FALLBACK_RE.search(text)
    if m:
        return m.group(1)

    return None

def extract_text_from_response(response_data: dict) -> str:
    """Extract the output text from the response JSON."""
    try:
        output = response_data.get("output", [])
        for item in output:
            if item.get("type") == "message":
                content = item.get("content", [])
                for content_item in content:
                    if content_item.get("type") == "output_text":
                        return content_item.get("text", "")
    except Exception:
        pass
    return ""

def get_model_name(response_data: dict, reasoning_effort: str = "medium") -> str:
    """Extract model name and reasoning effort."""
    model = response_data.get("model", "gpt-5")
    if "gpt-5" in model:
        # Extract reasoning effort if available
        reasoning = response_data.get("reasoning", {})
        effort = reasoning.get("effort", reasoning_effort)
        return f"openai::gpt-5-{effort}"
    return f"openai::{model}"

def compile_jsons_to_csv(json_dir: str, output_csv: str, labels_dict: Optional[dict] = None):
    """
    Compile all .response.json files from json_dir into a CSV file.

    Args:
        json_dir: Directory containing the JSON response files
        output_csv: Output CSV file path
        labels_dict: Optional dictionary mapping image names to gold labels
    """
    json_path = Path(json_dir)

    # Find all response.json files
    response_files = sorted(json_path.glob("*.response.json"))

    if not response_files:
        print(f"No .response.json files found in {json_dir}")
        return

    print(f"Found {len(response_files)} response files")

    # Prepare data rows
    rows = []

    for response_file in response_files:
        # Extract image name from filename (remove .response.json)
        image_name = response_file.stem.replace(".response", "") + ".png"

        try:
            # Read JSON file
            with open(response_file, "r", encoding="utf-8") as f:
                response_data = json.load(f)

            # Extract raw text
            raw_text = extract_text_from_response(response_data)

            # Parse bucket number
            parsed_label = parse_bucket(raw_text)
            parsed_int = int(parsed_label) if parsed_label and parsed_label.isdigit() else None

            # Get model name
            model = get_model_name(response_data)

            # Get gold labels if available
            gold = ""
            gold_int = ""
            correct = ""

            if labels_dict and image_name in labels_dict:
                gold = labels_dict[image_name]
                gold_int = int(gold) if gold.isdigit() else ""
                if parsed_int is not None and gold_int != "":
                    correct = int(parsed_int == gold_int)

            rows.append({
                "image": image_name,
                "model": model,
                "raw_text": raw_text,
                "parsed_label": parsed_label or "",
                "parsed_int": parsed_int or "",
                "gold": gold,
                "gold_int": gold_int,
                "correct": correct
            })

        except Exception as e:
            print(f"Error processing {response_file}: {e}")
            continue

    # Write to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "model", "raw_text", "parsed_label", "parsed_int",
            "gold", "gold_int", "correct"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile JSON response files to CSV")
    parser.add_argument("--json-dir", required=True, help="Directory containing .response.json files")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--labels", help="Optional labels CSV file (image,label columns)")

    args = parser.parse_args()

    # Load labels if provided
    labels_dict = None
    if args.labels:
        labels_dict = {}
        with open(args.labels, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = row.get("image", "")
                label = row.get("label", "") or row.get("gold", "")
                if image and label:
                    labels_dict[image] = label

    compile_jsons_to_csv(args.json_dir, args.output, labels_dict)
