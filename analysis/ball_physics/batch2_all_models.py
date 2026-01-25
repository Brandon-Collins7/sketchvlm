#!/usr/bin/env python3
"""
Combined Analysis for All Models in batch2
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


def load_ground_truth_from_metadata() -> Dict[str, int]:
    """Load ground truth from random_scene_metadata.json files."""
    ground_truth = {}

    # Path to the dataset metadata
    metadata_base = Path('/Users/log/Github/sketchvlm/datasets/large_second_batch')

    if not metadata_base.exists():
        print(f"Warning: Metadata directory not found: {metadata_base}")
        return ground_truth

    # Iterate through all run directories
    for run_dir in sorted(metadata_base.glob('run_b2_*')):
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
                # Handle both forward and backward slashes
                source_image = source_image.replace('\\', '/')
                image_name = Path(source_image).name
            else:
                # Fallback to raw_image or grid_image path
                image_path = data.get('raw_image', data.get('grid_image', ''))
                image_path = image_path.replace('\\', '/')
                image_name = Path(image_path).stem.replace('_orig', '').replace('_grid', '') + '.png'

            # Parse the answer
            answer = data.get('answer')
            # If answer is a string (e.g., "$\boxed{3}$"), parse it
            if isinstance(answer, str):
                answer = parse_boxed_answer(answer)
            # If answer is still None, try other fields
            if answer is None:
                # Try model_output_full first (for gemini3 image results)
                model_output_full = data.get('model_output_full', '')
                answer = parse_boxed_answer(model_output_full)
                if answer is None:
                    # Fallback to model_output
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
    like 'sample_YYYYMMDD_HHMMSS_run_b2_XXX' and contains a text_data.json file.
    The run name (e.g., run_b2_XXX) must be extracted from the directory name.
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
        # e.g., sample_20251203_182246_run_b2_001 -> run_b2_001
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
    """Process direct_vqa results from individual JSON files."""
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
                # Handle both forward and backward slashes
                source_image = source_image.replace('\\', '/')
                image_name = Path(source_image).name
            else:
                # Fallback to raw_image or grid_image path
                image_path = data.get('raw_image', data.get('grid_image', ''))
                image_path = image_path.replace('\\', '/')
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


def load_gemini3_nb_results_with_mapping(json_file: Path, gemini3_dir: Path,
                                         ground_truth: Dict[str, int]) -> List[Dict]:
    """
    Load Gemini-3-Pro consistency check results and map them to original images.

    Args:
        json_file: Path to batch2_gemini3_pro_nb_results.json
        gemini3_dir: Path to gemini3_image_two_turn_batch2 directory with original results
        ground_truth: Ground truth dictionary

    Returns:
        List of result dictionaries
    """
    if not json_file.exists():
        print(f"Warning: {json_file} not found")
        return []

    try:
        with open(json_file, 'r') as f:
            nb_data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return []

    # Build mapping from item index to original image name
    index_to_image = {}

    if gemini3_dir.exists():
        for item_file in sorted(gemini3_dir.glob("item_*.json")):
            try:
                with open(item_file, 'r') as f:
                    item_data = json.load(f)

                # Extract item index from filename
                match = re.search(r'item_(\d+)\.json', item_file.name)
                if not match:
                    continue

                item_idx = int(match.group(1))

                # Get source image name
                source_image = item_data.get('source_image', '')
                if source_image:
                    image_name = Path(source_image).name
                    index_to_image[item_idx] = image_name
            except Exception as e:
                print(f"Error processing {item_file}: {e}")
                continue

    # Now process the consistency check results
    results = []

    for item in nb_data:
        # Skip failed items
        if not item.get('success', False):
            continue

        # Get the index
        item_idx = item.get('index')
        if item_idx is None:
            continue

        # Get the response
        response = item.get('consistency_check_response', '')
        if not response:
            continue

        # Parse the answer
        answer = parse_boxed_answer(response)
        if answer is None:
            continue

        # Map to original image
        image_name = index_to_image.get(item_idx)
        if not image_name:
            continue

        # Get ground truth
        gold = ground_truth.get(image_name)
        if gold is None:
            continue

        boxed_answer = extract_boxed_text(response)
        results.append({
            'image': image_name,
            'model': 'Gemini-3-Pro-NB',
            'type': 'paths',
            'prediction': int(answer),
            'gold': gold,
            'correct': int(answer) == gold,
            'boxed_answer': boxed_answer,
            'model_output': response
        })

    return results


def process_all_models():
    """Process all models in batch2 directory."""
    base_dir = Path(__file__).parent.parent.parent
    batch2_dir = base_dir / "results" / "mix_eval" / "ball_paths" / "batch2"
    output_dir = base_dir / "analysis" / "ball_physics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth from metadata files
    print("Loading ground truth from random_scene_metadata.json files...")
    ground_truth = load_ground_truth_from_metadata()
    print(f"Loaded {len(ground_truth)} ground truth labels")

    all_results = []

    # Define models to process
    models_config = [
        # Sketch (ball_paths) models
        ('gemini_25_flash_ball_paths_batch2', 'Gemini-2.5-Flash', 'paths', False),
        ('gemini_25_pro_ball_paths_batch2', 'Gemini-2.5-Pro', 'paths', False),
        ('gpt5_low_ball_paths_batch2', 'GPT-5-low', 'paths', False),
        ('20260121_205501_gpt5low_ball_batch2', 'GPT-5-low-NEW', 'paths', False),
        ('gpt5_med_ball_paths_batch2', 'GPT-5-med', 'paths', False),
        ('qwen25_7b_ball_paths_batch2', 'Qwen2.5-7B', 'paths', False),
        ('vilasr_ball_paths_batch2', 'ViLaSR', 'paths', True),  # JSONL format
        ('gemini3_image_two_turn_batch2', 'nano_banana_pro', 'paths', False),
        ('gemini3_ball_paths_batch2', 'Gemini-3-Pro', 'paths', False),
        ('gemini3pro_batch2_ball_paths_no_grid_0_to_1000', 'Gemini-3-Pro-0-1000', 'paths', False),
        ('20260114_111423_gem3pro_meta_second_batch_copy', 'Gemini-3-Pro-Meta', 'paths', False),
        ('20260123_232913_gpt_meta_batch2_nogrid_sketch', 'GPT-Meta-NoGrid', 'paths', False),
        ('20260124_183213_gpt_meta_grid_vqa_2', 'GPT-Meta-Grid-VQA', 'paths', False),

    ]

    # Process sketch/ball_paths models
    print("\nProcessing sketch (ball_paths) results...")
    for dir_name, model_name, result_type, is_jsonl in models_config:
        model_dir = batch2_dir / dir_name
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
    thinkmorph_dir = batch2_dir / "thinkmorph_ball_paths_batch2"
    if thinkmorph_dir.exists():
        print(f"  Processing ThinkMorph...")
        results = process_thinkmorph_results(thinkmorph_dir, "ThinkMorph", ground_truth)
        all_results.extend(results)
        print(f"    Found {len(results)} results")

    # Process Gemini-3-Pro consistency check results
    print(f"  Processing Gemini-3-Pro-NB (consistency check)...")
    gemini3_nb_json = batch2_dir / "batch2_gemini3_pro_nb_results.json"
    gemini3_orig_dir = batch2_dir / "gemini3_image_two_turn_batch2"
    results = load_gemini3_nb_results_with_mapping(gemini3_nb_json, gemini3_orig_dir, ground_truth)
    all_results.extend(results)
    print(f"    Found {len(results)} results")

    # Process direct_vqa models
    print("\nProcessing direct_vqa results...")
    direct_vqa_dir = batch2_dir / "direct_vqa"

    # Direct VQA individual JSONs
    vqa_dirs = [
        ('gemini_25_flash_no_sketch_batch2', 'Gemini-2.5-Flash'),
        ('gemini_25_pro_no_sketch_batch_2', 'Gemini-2.5-Pro'),
        ('gpt5_low_no_sketch_batch2', 'GPT-5-low'),
        ('gpt5_med_no_sketch_batch2', 'GPT-5-med'),
        ('qwen25_7b_no_sketch_batch2', 'Qwen2.5-7B'),
        ('gemini3_no_sketch_batch2', 'Gemini-3-Pro'),
    ]

    for dir_name, model_name in vqa_dirs:
        vqa_model_dir = direct_vqa_dir / dir_name
        if vqa_model_dir.exists():
            print(f"  Processing {model_name} (direct_vqa)...")
            results = process_direct_vqa_individual_json(vqa_model_dir, model_name, ground_truth)
            all_results.extend(results)
            print(f"    Found {len(results)} results")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save to CSV
    output_file = output_dir / "batch2_all_models_results.csv"
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
    pivot_file = output_dir / "batch2_accuracy_pivot.csv"
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
