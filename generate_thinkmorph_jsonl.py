#!/usr/bin/env python3
"""
Generate a JSONL file from thinkmorph/vilasr directory structure.
"""
import json
import re
from pathlib import Path


def generate_jsonl(results_dir: Path, output_jsonl: Path):
    """Generate JSONL from text_data.json files in subdirectories."""
    results = []

    # Find all subdirectories with pattern sample_*_sim_*_initial
    for subdir in sorted(results_dir.glob("sample_*_sim_*_initial*")):
        if not subdir.is_dir():
            continue

        # Extract sim number from directory name
        match = re.search(r"sim_(\d+)_initial", subdir.name, re.I)
        if not match:
            continue

        sim_num = match.group(1)
        image_path = f"sim_{sim_num}_initial.png"

        # Read text_data.json
        text_data_path = subdir / "text_data.json"
        if not text_data_path.exists():
            continue

        try:
            j = json.loads(text_data_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Extract raw text and answer
        raw_text = ""
        answer = None

        # For thinkmorph: text_outputs is a list
        if "text_outputs" in j and isinstance(j["text_outputs"], list):
            raw_text = "\n".join(str(t) for t in j["text_outputs"])
        # For vilasr: response is a string
        elif "response" in j:
            raw_text = str(j.get("response", ""))

        # Extract <answer>X</answer>
        if raw_text:
            answer_match = re.search(r"<answer>(\d+)</answer>", raw_text, re.I)
            if answer_match:
                answer = int(answer_match.group(1))

        # Create result entry
        result = {
            "image_path": image_path,
            "model_output": raw_text,
            "answer": answer
        }
        results.append(result)

    # Write JSONL
    with output_jsonl.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Generated {len(results)} entries in {output_jsonl}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <results_dir> <output_jsonl>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    output_jsonl = Path(sys.argv[2])

    generate_jsonl(results_dir, output_jsonl)
