#!/usr/bin/env python3
"""
Create a comprehensive CSV file with all maze results.
Each row represents a unique maze + validity combination.
Columns include ground truth and all model responses.
"""

import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path('/Users/log/Github/sketchvlm')
OUTPUT_TAIL_CHARS = 50


def normalize_path(path_value: Optional[str]) -> str:
    """Return a repository-relative POSIX path if possible."""
    if not path_value:
        return ''

    path_str = str(path_value).replace('\\', '/')
    path_obj = Path(path_str)

    if path_obj.is_absolute():
        try:
            return str(path_obj.relative_to(PROJECT_ROOT)).replace('\\', '/')
        except ValueError:
            return path_str

    return path_str


def get_output_tail(text: Optional[str], limit: int = OUTPUT_TAIL_CHARS) -> str:
    """Return the trailing portion of a model output for quick inspection."""
    if not text:
        return ''

    text = str(text)
    if len(text) <= limit:
        return text
    return text[-limit:]


def build_result_entry(answer: str,
                       model_output: Optional[str] = None,
                       source_image: Optional[str] = None,
                       annotated_image: Optional[str] = None) -> Dict[str, str]:
    """Package grading information and lightweight metadata for CSV export."""
    return {
        'answer': answer,
        'output_tail': get_output_tail(model_output),
        'source_image': normalize_path(source_image),
        'annotated_image': normalize_path(annotated_image)
    }


def extract_answer_from_response(response_text: str) -> str:
    """Extract answer from model response."""
    if not response_text or response_text.strip() == '':
        return 'unknown'

    # Try <final_answer> tags first
    final_answer_match = re.search(r'<final_answer>\s*(.*?)\s*</final_answer>',
                                   response_text, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        answer_text = final_answer_match.group(1).strip()
        if 'valid' in answer_text.lower():
            if 'invalid' in answer_text.lower():
                return 'invalid'
            else:
                return 'valid'
        return 'unknown'

    # Try <answer> tags (for ViLaSR)
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>',
                            response_text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        if 'valid' in answer_text.lower():
            if 'invalid' in answer_text.lower():
                return 'invalid'
            else:
                return 'valid'
        return 'unknown'

    # Try \boxed{} format (for ViLaSR)
    boxed_match = re.search(r'\$?\\boxed\{(.*?)\}\$?',
                           response_text, re.IGNORECASE | re.DOTALL)
    if boxed_match:
        answer_text = boxed_match.group(1).strip()
        if 'valid' in answer_text.lower():
            if 'invalid' in answer_text.lower():
                return 'invalid'
            else:
                return 'valid'
        return 'unknown'

    # Fallback: check last 30 characters
    last_chars = response_text[-30:].lower()
    if 'invalid' in last_chars:
        return 'invalid'
    elif 'valid' in last_chars:
        return 'valid'

    return 'unknown'


def extract_maze_id(source_image_path: str) -> str:
    """Extract maze ID from source image path."""
    if not source_image_path:
        return None
    # Normalize path separators to handle Windows paths on Unix systems
    normalized_path = str(source_image_path).replace('\\', '/')
    filename = Path(normalized_path).stem
    return filename


def get_path_length_from_maze_id(maze_id: str, maze_to_path_length: Dict[str, int]) -> Optional[int]:
    """Get path length for a maze ID."""
    return maze_to_path_length.get(maze_id)


def build_maze_to_path_length_mapping() -> Dict[str, int]:
    """Build mapping of maze_id to path_length."""
    maze_to_path_length = {}

    for path_length in range(1, 10):  # Check up to path_length_9
        path_dir = Path(f'/Users/log/Github/sketchvlm/datasets/maze_v2/path_length_{path_length}')
        if not path_dir.exists():
            continue
        for maze_dir in path_dir.iterdir():
            if maze_dir.is_dir() and maze_dir.name.startswith('maze_'):
                maze_to_path_length[maze_dir.name] = path_length

    return maze_to_path_length


def load_model_results(base_path: Path, model_prefix: str, validity: str) -> Dict[str, Dict[str, str]]:
    """Load answers plus lightweight metadata for a directory of JSON results."""
    results: Dict[str, Dict[str, str]] = {}

    results_dir = base_path / f'{model_prefix}_{validity}'
    if not results_dir.exists():
        return results

    for json_file in results_dir.glob('item_*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            source_image = data.get('source_image', '')
            maze_id = extract_maze_id(source_image)
            model_output = data.get('model_output_full') or data.get('model_output', '')

            if not maze_id:
                continue

            annotated_path = json_file.with_name(f'{json_file.stem}_annotated.png')
            entry = build_result_entry(
                extract_answer_from_response(model_output),
                model_output=model_output,
                source_image=source_image,
                annotated_image=str(annotated_path) if annotated_path.exists() else None,
            )

            results[maze_id] = entry

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return results


def load_vilasr_jsonl_results(jsonl_path: Path) -> Dict[str, Dict[str, str]]:
    """Load ViLaSR answers plus snippets from JSONL files."""
    results: Dict[str, Dict[str, str]] = {}

    if not jsonl_path.exists():
        return results

    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    image_path = data.get('image_path', [])
                    if isinstance(image_path, list) and image_path:
                        source_image = image_path[0]
                    else:
                        source_image = image_path

                    maze_id = extract_maze_id(source_image)
                    if not maze_id:
                        continue

                    model_output = data.get('model_output', '')
                    results[maze_id] = build_result_entry(
                        extract_answer_from_response(model_output),
                        model_output=model_output,
                        source_image=source_image,
                    )

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {e}")
                    continue

    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")

    return results


def load_consistency_check_json(json_path: Path, results_base_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load Gemini consistency-check responses plus metadata.

    This function handles two JSON formats:
    1. Files with 'consistency_check_response' field (from limitedmaze_nanob_gemini3_pro_*.json)
    2. Files with 'model_answer' field (from nanob_gemini3_pro_*.json)

    For the limited maze format, we use consistency_check_response to get the answer.
    """
    results: Dict[str, Dict[str, str]] = {}

    if not json_path.exists():
        return results

    # Build mapping from item index to maze metadata
    item_meta: Dict[int, Dict[str, str]] = {}
    if results_base_dir and results_base_dir.exists():
        for jf in sorted(results_base_dir.glob('item_*.json')):
            try:
                j = json.loads(jf.read_text(encoding='utf-8'))
                source_img = str(j.get('source_image', '')).replace('\\', '/')
                maze_id = extract_maze_id(source_img)
                if not maze_id:
                    continue
                idxm = re.match(r'item_(\d+)\.json$', jf.name, re.I)
                if idxm:
                    item_idx = int(idxm.group(1))
                    item_meta[item_idx] = {
                        'maze_id': maze_id,
                        'source_image': source_img,
                        'item_idx': item_idx
                    }
            except Exception as e:
                continue

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            return results

        # Track seen image paths to handle duplicates (keep first occurrence)
        seen_image_paths = set()

        for entry in data:
            # Determine which format we're using
            has_consistency_response = 'consistency_check_response' in entry
            has_model_answer = 'model_answer' in entry

            image_path = entry.get('image_path', '')

            # Skip duplicates (keep first occurrence only)
            if image_path in seen_image_paths:
                continue
            seen_image_paths.add(image_path)

            maze_id = None
            meta = None

            # Preferred: map via item_XXX index using results_base_dir
            try:
                item_match = re.search(r'item_(\d+)', Path(image_path).name, re.I)
            except Exception:
                item_match = None

            annotated_image = None
            if item_match:
                idx = int(item_match.group(1))
                meta = item_meta.get(idx)
                if meta:
                    maze_id = meta.get('maze_id')
                # For Nano Banana, prefer the generated sketch image; fallback to annotated
                try:
                    gen_candidate = results_base_dir / f"item_{idx:05d}_generated_0.png"
                    ann_candidate = results_base_dir / f"item_{idx:05d}_annotated.png"
                    if gen_candidate.exists():
                        annotated_image = str(gen_candidate)
                    elif ann_candidate.exists():
                        annotated_image = str(ann_candidate)
                except Exception:
                    pass

            # Fallback: derive maze_id directly from the image_path filename
            if not maze_id:
                try:
                    maze_id = extract_maze_id(str(image_path))
                except Exception:
                    maze_id = None

            if not maze_id:
                continue

            # Extract answer based on format
            if has_consistency_response:
                # Use consistency_check_response for limited maze format
                response = entry.get('consistency_check_response', '')
                answer = extract_answer_from_response(response) if response else 'unknown'
            elif has_model_answer:
                # Use model_answer for direct format
                response = entry.get('model_answer', '')
                answer = extract_answer_from_response(response) if response else 'unknown'
            else:
                # Fallback
                response = ''
                answer = 'unknown'

            results[maze_id] = build_result_entry(
                answer,
                model_output=response,
                source_image=(meta.get('source_image') if meta else image_path),
                annotated_image=annotated_image,
            )

    except Exception as e:
        print(f"Error reading {json_path}: {e}")

    return results


def load_nanob_direct_json(json_path: Path, results_base_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load NanoBanana + Gemini direct VQA results from a single JSON array file.

    This file has entries with keys like: image_path, model_answer, extracted_answer, model.
    We map item_XXXXX to maze_id using the provided results_base_dir's item_XXXXX.json to
    recover the canonical source_image/maze_id. We also attach the generated image path
    (item_XXXXX_generated_0.png) as the annotated image for visualization.
    """
    results: Dict[str, Dict[str, str]] = {}

    if not json_path.exists():
        return results

    # Build quick mapping item index -> (maze_id, source_image)
    item_meta: Dict[int, Dict[str, str]] = {}
    if results_base_dir and results_base_dir.exists():
        for jf in sorted(results_base_dir.glob('item_*.json')):
            try:
                j = json.loads(jf.read_text(encoding='utf-8'))
                source_img = str(j.get('source_image', '')).replace('\\', '/')
                maze_id = extract_maze_id(source_img)
                if not maze_id:
                    continue
                idxm = re.match(r'item_(\d+)\.json$', jf.name, re.I)
                if idxm:
                    item_idx = int(idxm.group(1))
                    item_meta[item_idx] = {
                        'maze_id': maze_id,
                        'source_image': source_img
                    }
            except Exception:
                continue

    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
        if not isinstance(data, list):
            return results

        for entry in data:
            image_path = str(entry.get('image_path', ''))
            # Extract item index from the filename
            try:
                item_match = re.search(r'item_(\d+)', Path(image_path).name, re.I)
            except Exception:
                item_match = None

            meta = None
            maze_id = None
            annotated_image = None
            if item_match:
                idx = int(item_match.group(1))
                meta = item_meta.get(idx)
                if meta:
                    maze_id = meta.get('maze_id')
                # Prefer generated image for NanoBanana
                gen_candidate = results_base_dir / f"item_{idx:05d}_generated_0.png"
                ann_candidate = results_base_dir / f"item_{idx:05d}_annotated.png"
                if gen_candidate.exists():
                    annotated_image = str(gen_candidate)
                elif ann_candidate.exists():
                    annotated_image = str(ann_candidate)

            # Fallback: derive maze_id from image_path if mapping missing
            if not maze_id:
                maze_id = extract_maze_id(image_path)

            if not maze_id:
                continue

            # Choose the answer: prefer extracted_answer, else parse model_answer
            extracted = (entry.get('extracted_answer') or '').strip()
            if extracted:
                final_answer = extracted.lower() if extracted.lower() in ('valid', 'invalid') else extract_answer_from_response(extracted)
                tail_text = entry.get('extracted_answer', '')
            else:
                model_answer = entry.get('model_answer', '')
                final_answer = extract_answer_from_response(model_answer)
                tail_text = model_answer

            results[maze_id] = build_result_entry(
                final_answer,
                model_output=tail_text,
                source_image=(meta.get('source_image') if meta else image_path),
                annotated_image=annotated_image,
            )

    except Exception as e:
        print(f"Error reading {json_path}: {e}")

    return results

def load_nanob_combined_json(json_path: Path) -> Dict[str, Dict[str, str]]:
    """Load unified NanoB + Gemini combined JSON produced by build_nanob_combined.py."""
    results: Dict[str, Dict[str, str]] = {}
    if not json_path.exists():
        return results
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
        if not isinstance(data, list):
            return results
        for e in data:
            maze_id = e.get('maze_id')
            if not maze_id:
                continue
            results[maze_id] = build_result_entry(
                e.get('extracted_answer', 'unknown'),
                model_output=e.get('model_output', ''),
                source_image=e.get('source_image', ''),
                annotated_image=e.get('image_path', ''),
            )
    except Exception as ex:
        print(f'Error reading combined JSON {json_path}: {ex}')
    return results


def load_thinkmorph_results(thinkmorph_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load ThinkMorph answers from per-sample directories."""
    results: Dict[str, Dict[str, str]] = {}

    if not thinkmorph_dir.exists():
        return results

    for sample_dir in thinkmorph_dir.iterdir():
        if not sample_dir.is_dir() or not sample_dir.name.startswith('sample_'):
            continue

        parts = sample_dir.name.split('_')
        maze_idx = None
        for i, part in enumerate(parts):
            if part == 'maze':
                maze_idx = i
                break

        if maze_idx is None:
            continue

        maze_id = '_'.join(parts[maze_idx:])
        json_file = sample_dir / 'text_data.json'
        if not json_file.exists():
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            text_outputs = data.get('text_outputs', [])
            model_output = '\n'.join(text_outputs) if text_outputs else ''
            results[maze_id] = build_result_entry(
                extract_answer_from_response(model_output),
                model_output=model_output,
            )

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return results


def main():
    print("Building maze to path length mapping...")
    maze_to_path_length = build_maze_to_path_length_mapping()
    print(f"Mapped {len(maze_to_path_length)} mazes from dataset")

    base_path = Path('/Users/log/Github/sketchvlm/results/mix_eval/maze_v2')

    # Define all models
    # Format: (model_name, model_base_path, model_prefix, format_type)
    # format_type: 'json' for standard, 'jsonl' for ViLaSR, 'thinkmorph' for ThinkMorph, 'consistency' for consistency check
    models = [
        ('gemini_flash_sketch', base_path / 'gemini', 'gemini25_flash', 'json'),
        ('gemini_flash_vqa', base_path / 'gemini' / 'direct_vqa', 'gemini25_flash', 'json'),
        ('gemini_flash_two_turn', base_path / 'gemini' / 'two_turn', 'gemini25_flash', 'json'),
        ('gemini_pro_sketch', base_path / 'gemini', 'gemini25_pro', 'json'),
        ('gemini_pro_vqa', base_path / 'gemini' / 'direct_vqa', 'gemini25_pro', 'json'),
        ('gemini_pro_two_turn', base_path / 'gemini' / 'two_turn', 'gemini25_pro', 'json'),
        ('gemini3_pro_sketch', base_path / 'gemini', 'gemini3_pro', 'json'),
        ('gemini3_pro_vqa', base_path / 'gemini' / 'direct_vqa', 'gemini3_pro', 'json'),
        ('gemini3_pro_0_1000_sketch', base_path / 'gemini', 'gemini3pro_gridworld_paths_0_to_1000', 'json'),
        ('gpt5_med_sketch', base_path / 'gpt5', 'gpt5_med', 'json'),
        ('gpt5_med_vqa', base_path / 'gpt5' / 'direct_vqa', 'gpt5_med', 'json'),
        ('gpt5_low_sketch', base_path / 'gpt5', 'gpt5_low', 'json'),
        ('gpt5_low_vqa', base_path / 'gpt5' / 'direct_vqa', 'gpt5_low', 'json'),
        ('gpt5_low_two_turn', base_path / 'gpt5' / 'two_turn', 'gpt5_low', 'json'),
        ('gpt5_low_1000_sketch', base_path / 'gpt5', 'gpt5_low_1000', 'json'),
        ('qwen3_235b_sketch', base_path / 'qwen3', 'qwen3_235b', 'json'),
        ('qwen3_235b_vqa', base_path / 'qwen3' / 'direct_vqa', 'qwen3_235b', 'json'),
        ('qwen25_7b_sketch', base_path / 'qwen25_7b', 'qwen25_7b', 'json'),
        ('qwen25_7b_vqa', base_path / 'qwen25_7b' / 'direct_vqa', 'qwen25_7b', 'json'),
        ('vilasr_sketch', base_path / 'vilasr', 'vilasr', 'jsonl'),
        ('thinkmorph_sketch', base_path / 'thinkmorph', 'thinkmorph', 'thinkmorph'),
        # Use consistency check format for NanoB + Gemini (from merged files)
        ('nanob_gemini3', base_path / 'gemini' / 'normal_gemini3', 'maze_nanob_gemini3pro_merged', 'consistency'),
    ]

    # Collect all maze IDs from results files (not just dataset)
    print("\nCollecting maze IDs from results...")
    all_maze_ids = set()

    # Scan one model's results to get all maze IDs
    ref_model_base = base_path / 'gemini'
    ref_model_prefix = 'gemini25_flash'

    for validity in ['invalid', 'valid']:
        results_dir = ref_model_base / f'{ref_model_prefix}_{validity}'
        if results_dir.exists():
            for json_file in results_dir.glob('item_*.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    source_image = data.get('source_image', '')
                    maze_id = extract_maze_id(source_image)
                    if maze_id:
                        all_maze_ids.add(maze_id)
                except Exception as e:
                    print(f"  Error reading {json_file}: {e}")

    print(f"Found {len(all_maze_ids)} unique mazes in results")

    # Load all model results
    print("\nLoading model results...")
    all_results = {}

    for model_name, model_base, model_prefix, format_type in models:
        print(f"  Loading {model_name}...")
        if format_type == 'jsonl':
            # Load from JSONL files
            invalid_jsonl = model_base / f'{model_prefix}_invalid' / 'results.jsonl'
            valid_jsonl = model_base / f'{model_prefix}_valid' / 'results.jsonl'
            all_results[model_name] = {
                'invalid': load_vilasr_jsonl_results(invalid_jsonl),
                'valid': load_vilasr_jsonl_results(valid_jsonl)
            }
        elif format_type == 'thinkmorph':
            # Load from ThinkMorph directories
            invalid_dir = model_base / f'{model_prefix}_invalid'
            valid_dir = model_base / f'{model_prefix}_valid'
            all_results[model_name] = {
                'invalid': load_thinkmorph_results(invalid_dir),
                'valid': load_thinkmorph_results(valid_dir)
            }
        elif format_type == 'consistency':
            # Load from consistency check JSON files
            invalid_json = model_base / f'{model_prefix}_invalid.json'
            valid_json = model_base / f'{model_prefix}_valid.json'
            # Get the original results directory for mapping
            # maze_nanob_gemini3pro_merged -> nanob_maze
            if 'nanob' in model_prefix:
                results_base_invalid = base_path / 'nano_banana' / 'nanob_maze_invalid'
                results_base_valid = base_path / 'nano_banana' / 'nanob_maze_valid'
            else:
                # Legacy handling: maze_nanob_gemini3_pro -> nanob_maze
                original_model = model_prefix.replace('_gemini3_pro', '')
                if original_model.startswith('maze_'):
                    original_model = original_model.replace('maze_', '', 1) + '_maze'
                results_base_invalid = base_path / 'nano_banana' / f'{original_model}_invalid'
                results_base_valid = base_path / 'nano_banana' / f'{original_model}_valid'
            all_results[model_name] = {
                'invalid': load_consistency_check_json(invalid_json, results_base_invalid),
                'valid': load_consistency_check_json(valid_json, results_base_valid)
            }
        elif format_type == 'nanob_combined':
            # Load from pre-built combined JSON arrays
            invalid_json = model_base / f'{model_prefix}_invalid.json'
            valid_json = model_base / f'{model_prefix}_valid.json'
            all_results[model_name] = {
                'invalid': load_nanob_combined_json(invalid_json),
                'valid': load_nanob_combined_json(valid_json)
            }
        else:
            # Load from individual JSON files
            all_results[model_name] = {
                'invalid': load_model_results(model_base, model_prefix, 'invalid'),
                'valid': load_model_results(model_base, model_prefix, 'valid')
            }

    # No in-memory combine; rely on prebuilt combined JSON

    # Expand maze set to include any IDs present in any loaded model results
    expanded_ids = set(all_maze_ids)
    for model_name, buckets in all_results.items():
        for validity in ('invalid', 'valid'):
            expanded_ids.update(buckets.get(validity, {}).keys())
    if len(expanded_ids) != len(all_maze_ids):
        print(f"\nExpanding maze set using loaded models: {len(all_maze_ids)} -> {len(expanded_ids)}")
    all_maze_ids = sorted(expanded_ids)

    # Build CSV rows
    print("\nBuilding CSV data...")
    rows = []

    # Check for mazes without path length info
    mazes_without_path_length = [m for m in all_maze_ids if m not in maze_to_path_length]
    if mazes_without_path_length:
        print(f"  Warning: {len(mazes_without_path_length)} mazes not found in dataset directories")
        print(f"  These will be marked with path_length='unknown'")

    model_names = [name for name, _, _, _ in models]

    def populate_row(row: Dict[str, str], validity: str, maze_id: str):
        row['source_image'] = ''
        for model_name in model_names:
            entry = all_results.get(model_name, {}).get(validity, {}).get(maze_id)
            if entry:
                row[model_name] = entry.get('answer', 'missing')
                row[f'{model_name}__output_tail'] = entry.get('output_tail', '')
                row[f'{model_name}__annotated_image'] = entry.get('annotated_image', '')
                if not row['source_image'] and entry.get('source_image'):
                    row['source_image'] = entry['source_image']
            else:
                row[model_name] = 'missing'
                row[f'{model_name}__output_tail'] = ''
                row[f'{model_name}__annotated_image'] = ''

    for maze_id in sorted(all_maze_ids):
        path_length = maze_to_path_length.get(maze_id, 'unknown')

        invalid_row = {
            'maze_id': maze_id,
            'path_length': path_length,
            'validity': 'invalid',
            'ground_truth': 'invalid'
        }
        populate_row(invalid_row, 'invalid', maze_id)
        rows.append(invalid_row)

        valid_row = {
            'maze_id': maze_id,
            'path_length': path_length,
            'validity': 'valid',
            'ground_truth': 'valid'
        }
        populate_row(valid_row, 'valid', maze_id)
        rows.append(valid_row)

    # Write CSV
    output_path = Path('/Users/log/Github/sketchvlm/analysis/maze/maze_v2_combined_results.csv')

    print(f"\nWriting CSV to {output_path}...")

    fieldnames = ['maze_id', 'path_length', 'validity', 'ground_truth', 'source_image']
    for model_name, _, _, _ in models:
        fieldnames.append(model_name)
    for model_name, _, _, _ in models:
        fieldnames.append(f'{model_name}__output_tail')
        fieldnames.append(f'{model_name}__annotated_image')

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSuccessfully wrote {len(rows)} rows to {output_path}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total unique mazes: {len(all_maze_ids)}")
    print(f"  Total rows (maze × validity): {len(rows)}")
    print(f"  Models included: {len(models)}")
    print(f"  Columns: {len(fieldnames)}")

    # Check for missing data
    print("\nData completeness check:")
    for model_name, _, _, _ in models:
        invalid_count = sum(1 for row in rows if row['validity'] == 'invalid' and row[model_name] != 'missing')
        valid_count = sum(1 for row in rows if row['validity'] == 'valid' and row[model_name] != 'missing')
        total_invalid = sum(1 for row in rows if row['validity'] == 'invalid')
        total_valid = sum(1 for row in rows if row['validity'] == 'valid')

        print(f"  {model_name}:")
        print(f"    Invalid: {invalid_count}/{total_invalid}")
        print(f"    Valid: {valid_count}/{total_valid}")


if __name__ == '__main__':
    main()
