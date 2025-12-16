#!/usr/bin/env python3
"""
Combined Analysis for All Models in batch1
Processes both sketch (ball_paths) and direct_vqa results for all models.
"""

import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_boxed_answer(text: str) -> Optional[int]:
    r"""Extract answer from $\boxed{...}$ or \boxed{...} or \(\boxed{...}\) or <answer> format."""
    if not text:
        return None

    # Try <answer> tags first (for ViLaSR)
    answer_match = re.search(r'<answer>\s*(\d+|none)\s*</answer>', text, re.IGNORECASE)
    if answer_match:
        content = answer_match.group(1).strip().lower()
        if content == 'none':
            return 0
        try:
            return int(content)
        except ValueError:
            return None

    # Look for various boxed patterns
    # Try $\boxed{...}$ first
    match = re.search(r'\$\\boxed\{([^}]+)\}\$', text)
    if not match:
        # Try \boxed{...}
        match = re.search(r'\\boxed\{([^}]+)\}', text)
    if not match:
        # Try \(\boxed{...}\)
        match = re.search(r'\\\(\\boxed\{([^}]+)\}\\\)', text)

    if match:
        content = match.group(1).strip().lower()
        if content == 'none':
            return 0  # Use 0 for "none"
        try:
            return int(content)
        except ValueError:
            return None
    return None


def extract_boxed_text(text: str) -> str:
    r"""Extract the full boxed answer text including the boxed notation."""
    if not text:
        return ""

    # Look for various boxed patterns and return the full match
    # Try $\boxed{...}$ first
    match = re.search(r'\$\\boxed\{[^}]+\}\$', text)
    if match:
        return match.group(0)

    # Try \boxed{...}
    match = re.search(r'\\boxed\{[^}]+\}', text)
    if match:
        return match.group(0)

    # Try \(\boxed{...}\)
    match = re.search(r'\\\(\\boxed\{[^}]+\}\\\)', text)
    if match:
        return match.group(0)

    return ""


def load_ground_truth() -> Dict[str, int]:
    """Load ground truth from random_scene_metadata.json files."""
    ground_truth = {}

    # Path to the dataset metadata
    metadata_base = Path('/Users/log/Github/sketchvlm/datasets/large_run_split')

    if not metadata_base.exists():
        print(f"Warning: Metadata directory not found: {metadata_base}")
        return ground_truth

    # Iterate through all run directories
    for run_dir in sorted(metadata_base.glob('run_*')):
        metadata_file = run_dir / 'random_scene_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)

                # Extract bucket_hit from simulation data
                bucket_hit = data.get('simulation', {}).get('bucket_hit')
                if bucket_hit is not None:
                    image_name = run_dir.name + '.png'
                    ground_truth[image_name] = int(bucket_hit)
            except Exception as e:
                print(f"Error loading ground truth from {metadata_file}: {e}")

    return ground_truth


def process_sketch_results(model_dir: Path, model_name: str, ground_truth: Dict[str, int]) -> List[Dict]:
    """Process sketch/ball_paths results from individual JSON files."""
    results = []

    # Process all item_*.json files
    for json_file in sorted(model_dir.glob("item_*.json")):
        if json_file.name == "summary.json":
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract image name from source_image (original filename)
            source_image = data.get('source_image', '')
            if source_image:
                image_name = Path(source_image).name
            else:
                # Fallback to raw_image or grid_image path
                image_path = data.get('raw_image', data.get('grid_image', ''))
                image_name = Path(image_path).stem.replace('_orig', '').replace('_grid', '') + '.png'

            # Parse the answer
            answer = data.get('answer')
            if answer is None:
                # Try to parse from model_output if it contains boxed format
                model_output = data.get('model_output', '')
                answer = parse_boxed_answer(model_output)

            if answer is not None:
                gold = ground_truth.get(image_name)
                if gold is not None:
                    # Get full model output (prefer model_output_full, fallback to model_output)
                    full_output = data.get('model_output_full', data.get('model_output', ''))
                    boxed_answer = extract_boxed_text(full_output)
                    results.append({
                        'image': image_name,
                        'model': model_name,
                        'type': 'paths',
                        'prediction': int(answer),
                        'gold': gold,
                        'correct': int(answer) == gold,
                        'boxed_answer': boxed_answer,
                        'model_output': full_output
                    })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return results


def process_direct_vqa_json_array(json_file: Path, model_name: str, ground_truth: Dict[str, int]) -> List[Dict]:
    """Process direct_vqa results from JSON array format (Gemini models)."""
    results = []

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        for item in data:
            # Get run_dir and extract image name
            run_dir = item.get('run_dir', '')
            image_name = Path(run_dir).name + '.png'

            # Parse answer from response_text
            response_text = item.get('response_text', '')
            answer = parse_boxed_answer(response_text)

            if answer is not None:
                gold = ground_truth.get(image_name)
                if gold is not None:
                    boxed_answer = extract_boxed_text(response_text)
                    results.append({
                        'image': image_name,
                        'model': model_name,
                        'type': 'direct_vqa',
                        'prediction': answer,
                        'gold': gold,
                        'correct': answer == gold,
                        'boxed_answer': boxed_answer,
                        'model_output': response_text
                    })
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

    return results


def process_direct_vqa_csv(csv_file: Path, model_name: str, ground_truth: Dict[str, int]) -> List[Dict]:
    """Process direct_vqa results from CSV format (Qwen models)."""
    results = []

    try:
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            image_name = row['image']
            parsed_int = row.get('parsed_int')

            if pd.notna(parsed_int):
                gold = ground_truth.get(image_name)
                if gold is not None:
                    raw_text = row.get('raw_text', '')
                    boxed_answer = extract_boxed_text(raw_text)
                    results.append({
                        'image': image_name,
                        'model': model_name,
                        'type': 'direct_vqa',
                        'prediction': int(parsed_int),
                        'gold': gold,
                        'correct': int(parsed_int) == gold,
                        'boxed_answer': boxed_answer,
                        'model_output': raw_text
                    })
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

    return results


def process_jsonl_results(jsonl_file: Path, model_name: str, result_type: str, ground_truth: Dict[str, int]) -> List[Dict]:
    """Process ViLaSR results from a JSONL file."""
    results = []

    if not jsonl_file.exists():
        return results

    try:
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Extract image name from image_path
                    image_path = data.get('image_path', [])
                    if isinstance(image_path, list) and len(image_path) > 0:
                        image_name = Path(image_path[0]).name
                    else:
                        image_name = Path(image_path).name

                    # Parse the answer from model_output
                    model_output = data.get('model_output', '')
                    answer = parse_boxed_answer(model_output)

                    if answer is not None:
                        gold = ground_truth.get(image_name)
                        if gold is not None:
                            boxed_answer = extract_boxed_text(model_output)
                            results.append({
                                'image': image_name,
                                'model': model_name,
                                'type': result_type,
                                'prediction': int(answer),
                                'gold': gold,
                                'correct': int(answer) == gold,
                                'boxed_answer': boxed_answer,
                                'model_output': model_output
                            })
                except json.JSONDecodeError as e:
                    print(f"    Warning: Error decoding JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"    Warning: Error processing line {line_num}: {e}")
                    continue

    except Exception as e:
        print(f"  Error reading {jsonl_file}: {e}")

    return results


def process_thinkmorph_results(model_dir: Path, model_name: str, ground_truth: Dict[str, int]) -> List[Dict]:
    """Process ThinkMorph results from directories with text_data.json files.

    ThinkMorph has a unique structure where each result is in a directory named
    like 'sample_YYYYMMDD_HHMMSS_run_XXX' and contains a text_data.json file.
    The run name (e.g., run_XXX or run_XXX_Y) must be extracted from the directory name.
    """
    results = []

    if not model_dir.exists():
        return results

    # Process all sample directories
    for sample_dir in sorted(model_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        # Extract run name from directory name
        # Pattern: sample_YYYYMMDD_HHMMSS_<run_name>
        # e.g., sample_20251203_080208_run_001_1 -> run_001_1
        dir_name = sample_dir.name
        parts = dir_name.split('_')

        # Find the index where 'run' starts
        run_idx = None
        for i, part in enumerate(parts):
            if part == 'run':
                run_idx = i
                break

        if run_idx is None:
            continue

        # Extract run name (everything from 'run' onwards)
        run_name = '_'.join(parts[run_idx:])
        image_name = run_name + '.png'

        # Look for text_data.json
        json_file = sample_dir / 'text_data.json'
        if not json_file.exists():
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract answer from text_outputs
            text_outputs = data.get('text_outputs', [])
            model_output = '\n'.join(text_outputs) if text_outputs else ''

            # Parse the boxed answer
            answer = parse_boxed_answer(model_output)

            if answer is not None:
                gold = ground_truth.get(image_name)
                if gold is not None:
                    boxed_answer = extract_boxed_text(model_output)
                    results.append({
                        'image': image_name,
                        'model': model_name,
                        'type': 'paths',
                        'prediction': int(answer),
                        'gold': gold,
                        'correct': int(answer) == gold,
                        'boxed_answer': boxed_answer,
                        'model_output': model_output
                    })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return results


def process_direct_vqa_individual_json(model_dir: Path, model_name: str, ground_truth: Dict[str, int]) -> List[Dict]:
    """Process direct_vqa results from individual JSON files (GPT5)."""
    results = []

    # Process all item_*.json files
    for json_file in sorted(model_dir.glob("item_*.json")):
        if json_file.name == "summary.json":
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract image name from source_image (original filename)
            source_image = data.get('source_image', '')
            if source_image:
                image_name = Path(source_image).name
            else:
                # Fallback to raw_image or grid_image path
                image_path = data.get('raw_image', data.get('grid_image', ''))
                image_name = Path(image_path).stem.replace('_orig', '').replace('_grid', '') + '.png'

            # Parse the answer from model_output_full or model_output
            model_output = data.get('model_output_full', data.get('model_output', ''))
            answer = parse_boxed_answer(model_output)

            if answer is not None:
                gold = ground_truth.get(image_name)
                if gold is not None:
                    # Get full model output (prefer model_output_full, fallback to model_output)
                    full_output = data.get('model_output_full', data.get('model_output', ''))
                    boxed_answer = extract_boxed_text(full_output)
                    results.append({
                        'image': image_name,
                        'model': model_name,
                        'type': 'direct_vqa',
                        'prediction': answer,
                        'gold': gold,
                        'correct': answer == gold,
                        'boxed_answer': boxed_answer,
                        'model_output': full_output
                    })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return results


def process_all_models():
    """Process all models in batch1 directory."""
    base_dir = Path(__file__).parent.parent.parent
    batch1_dir = base_dir / "results" / "mix_eval" / "ball_paths" / "batch1"
    output_dir = base_dir / "analysis" / "ball_physics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth from metadata files
    print("Loading ground truth from random_scene_metadata.json files...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground truth labels")

    all_results = []

    # Define models to process
    models_config = [
        # Sketch (ball_paths) models
        ('gemini_25_flash_ball_paths', 'Gemini-2.5-Flash', 'paths', False),
        ('gemini_25_pro_ball_paths', 'Gemini-2.5-Pro', 'paths', False),
        ('gpt5_low_ball_paths', 'GPT-5-low', 'paths', False),
        ('gpt5_med_ball_paths', 'GPT-5-med', 'paths', False),
        ('qwen3_235b_thinking_ball_paths', 'Qwen-235B', 'paths', False),
        ('qwen3_8b_thinking_ball_paths', 'Qwen-8B', 'paths', False),
        ('qwen25_7b_ball_paths', 'Qwen2.5-7B', 'paths', False),
        ('vilasr_ball_paths', 'ViLaSR', 'paths', True),  # JSONL format
    ]

    # Process sketch/ball_paths models
    print("\nProcessing sketch (ball_paths) results...")
    for dir_name, model_name, result_type, is_jsonl in models_config:
        model_dir = batch1_dir / dir_name
        if model_dir.exists() and model_dir.is_dir():
            print(f"  Processing {model_name}...")
            if is_jsonl:
                # Process JSONL file
                jsonl_file = model_dir / 'results.jsonl'
                results = process_jsonl_results(jsonl_file, model_name, result_type, ground_truth)
            else:
                # Process individual JSON files
                results = process_sketch_results(model_dir, model_name, ground_truth)
            all_results.extend(results)
            print(f"    Found {len(results)} results")

    # Process ThinkMorph (special handling due to different directory structure)
    thinkmorph_dir = batch1_dir / "thinkmorph_ball_paths"
    if thinkmorph_dir.exists():
        print(f"  Processing ThinkMorph...")
        results = process_thinkmorph_results(thinkmorph_dir, "ThinkMorph", ground_truth)
        all_results.extend(results)
        print(f"    Found {len(results)} results")

    # Process direct_vqa models
    print("\nProcessing direct_vqa results...")
    direct_vqa_dir = batch1_dir / "direct_vqa"

    # Gemini JSON arrays
    for json_file in direct_vqa_dir.glob("gemini-*.json"):
        if "2.5-flash" in json_file.name:
            model_name = "Gemini-2.5-Flash"
        elif "2.5-pro" in json_file.name:
            model_name = "Gemini-2.5-Pro"
        else:
            continue

        print(f"  Processing {model_name} (direct_vqa)...")
        results = process_direct_vqa_json_array(json_file, model_name, ground_truth)
        all_results.extend(results)
        print(f"    Found {len(results)} results")

    # Qwen and GPT-5-med CSVs
    csv_files = [
        ('qwen235b_results.csv', 'Qwen-235B'),
        ('qwen8b_results.csv', 'Qwen-8B'),
        ('gpt5_med_no_sketch.csv', 'GPT-5-med'),
    ]

    for csv_file, model_name in csv_files:
        csv_path = direct_vqa_dir / csv_file
        if csv_path.exists():
            print(f"  Processing {model_name} (direct_vqa)...")
            results = process_direct_vqa_csv(csv_path, model_name, ground_truth)
            all_results.extend(results)
            print(f"    Found {len(results)} results")

    # GPT5-low and Qwen2.5-7B individual JSONs
    individual_json_dirs = [
        ('gpt5_low_no_sketch', 'GPT-5-low'),
        ('qwen25_7b_no_sketch', 'Qwen2.5-7B'),
    ]

    for dir_name, model_name in individual_json_dirs:
        model_dir = direct_vqa_dir / dir_name
        if model_dir.exists():
            print(f"  Processing {model_name} (direct_vqa)...")
            results = process_direct_vqa_individual_json(model_dir, model_name, ground_truth)
            all_results.extend(results)
            print(f"    Found {len(results)} results")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save to CSV
    output_file = output_dir / "batch1_all_models_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved all results to: {output_file}")
    print(f"  Total results: {len(df)}")

    # Compute and display accuracy
    print("\n" + "="*80)
    print("ACCURACY SUMMARY")
    print("="*80)

    # Overall accuracy by model and type
    summary = df.groupby(['model', 'type']).agg({
        'correct': ['sum', 'count', 'mean']
    }).round(4)

    print("\nAccuracy by Model and Type:")
    print("-" * 80)
    for (model, type_), row in summary.iterrows():
        correct = int(row['correct']['sum'])
        total = int(row['correct']['count'])
        accuracy = row['correct']['mean'] * 100
        type_label = 'SketchVLM' if type_ == 'paths' else 'Direct VQA'
        print(f"{model:20s} | {type_label:12s} | {correct:3d}/{total:3d} = {accuracy:5.1f}%")

    # Create pivot table for easier comparison
    pivot = df.groupby(['model', 'type'])['correct'].mean().unstack(fill_value=0) * 100

    # Save pivot table
    pivot_file = output_dir / "batch1_accuracy_pivot.csv"
    pivot.to_csv(pivot_file)
    print(f"\n✓ Saved accuracy pivot table to: {pivot_file}")

    # Display pivot table
    print("\n" + "="*80)
    print("ACCURACY PIVOT TABLE (%)")
    print("="*80)
    print(pivot.to_string())

    # Model-level summary (best of both approaches)
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    model_summary = df.groupby('model')['correct'].agg(['sum', 'count', 'mean'])
    model_summary['accuracy'] = (model_summary['mean'] * 100).round(1)
    model_summary = model_summary.sort_values('accuracy', ascending=False)

    print("\nOverall Accuracy (combining both direct_vqa and paths):")
    for model, row in model_summary.iterrows():
        print(f"  {model:20s}: {int(row['sum']):3d}/{int(row['count']):3d} = {row['accuracy']:5.1f}%")

    return df, pivot


if __name__ == "__main__":
    df, pivot = process_all_models()
