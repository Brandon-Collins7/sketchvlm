import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict


# Standard prompts for consistency checking
GENERAL_PROMPT = "You are given an image that another AI model has annotated. Your task is to analyze the annotation and determine what the final answer should be.\n\n"
BALL_DROP_PROMPT = "The image is a physics simulation of a ball being dropped. There are 4 different buckets called bucket 1, bucket 2, bucket 3, and bucket 4. Please respond with what bucket the **annotation shows** the ball will fall into. Please note that it is not your job to determine if the annotation is correct or not, just to extract the answer from the annotation. Your final answer must be formatted as \"$\\boxed{bucket number}$\". For example, if the ball will fall into bucket 2, respond with \"$\\boxed{2}$\". If the annotation does not show the ball falling into any bucket, respond with \"$\\boxed{none}$\". If the the annotation shows that the ball will fall into multiple different buckets, answer with \"$\\boxed{multiple}$\"."


BALL_QUALITY = """# Overall Instructions
You will be shown two images: the original image and an AI-annotated version. Your job is to grade the quality of the sketch using a rubric. You should grade only based on the following attributes that are given.

**Logic consistency**
Do the drawn lines make some sort of logical sense? For example, does drawn ball path clip through any of the static environment or does the ball path follow _extremely_ unrealistic physics? If there is no sketch provided, then the score should be a 1. The score should be based on the scale of 1 - 5

1) The sketch makes absolutely no logical sense.
2) The sketch has some critical flaws that breaks the logic of the sketch.
3) The sketch contains multiple logical errors.
4) The sketch contains a minor logical error.
5) The sketch contains zero logical errors.

# Output Format
You should follow this output format EXACTLY with no other output:

{reasoning for logical consistency score}
Logical Consistency Score: {integer from 1 - 5}

# Example Output

<example_1>
The drawing contains multiple errors. The ball path clips through one of the platforms, but it is a minor issue as the path of the ball still makes sense. Additionally, the ball path defies gravity and begins to momentarily float straight upwards for no logical reason. The minor error combined with the critical error results in a logical score of 2/5
Logical Consistency Score: 5
</example_1>
"""

MAZE_QUALITY = """# Overall Instructions
You will be shown two images: the original image and an AI-annotated version. Your job is to grade the quality of the sketch using a rubric. You should grade only based on the following attributes that are given.

**Quality Checks**
Do the drawn lines make some sort of logical sense? For example, does drawn path clip through any of the static environment? Does the sketch contain additional moves that are not in the path? Does the drawn sketch contradict the given text path? If there is no sketch provided, then the score should be a 1. The score should be based on the scale of 1 - 5

1) The sketch makes absolutely no logical sense.
2) The sketch has some critical flaws that breaks the logic of the sketch.
3) The sketch contains multiple logical errors.
4) The sketch contains a minor error.
5) The sketch contains zero errors.

# Output Format
You should follow this output format EXACTLY with no other output:

{reasoning for quality score}
Quality Score: {integer from 1 - 5}

# Example Output

<example_1>
The drawing contains multiple errors. The drawn path goes up, up, left instead of up, up, right. This contradicts the given text path. Additionally, the end of the drawn path slightly clips through the solid black wall. The minor error combined with the critical error results in a logical score of 2/5.

Quality Score: 5
</example_1>
"""

GEMINI_3_SECOND_TURN = """You are given an image that another AI model has annotated. Your task is to analyze the annotation and determine what the final answer should be.

You are given the start frame of a physics simulation. A ball is dropped from the top of the screen and falls due to gravity. The ball can roll off the lines or the walls in the image. The bouncing of the ball is relatively minor and realistic for normal gravity. Nothing in the image will move besides the ball. Predict which bucket will eventually catch the ball. There are 4 different buckets called bucket 1, bucket 2, bucket 3, and bucket 4. Draw the path that the ball will take. Please also respond with what bucket the ball will fall into. Your final answer must be formatted as "$\boxed{bucket number}$". For example, if the ball will fall into bucket 2, respond with "$\boxed{2}$"."""


def extract_answer_from_response(response_text: str) -> str:
    """
    Extract answer from model response.
    Tries multiple patterns: <answer> tags, <final_answer> tags, or last number.

    Args:
        response_text: The full model output

    Returns:
        Extracted answer as string, or empty string if not found
    """
    if not response_text or response_text.strip() == '':
        return ''

    # Try to extract from <answer> tags first
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # Try <final_answer> tags
    final_answer_match = re.search(r'<final_answer>\s*(.*?)\s*</final_answer>', response_text, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        return final_answer_match.group(1).strip()

    # Try to find "The answer is: X" pattern
    answer_is_match = re.search(r'(?:the answer is|answer:)\s*[:]*\s*(\d+|[A-Za-z]+)', response_text, re.IGNORECASE)
    if answer_is_match:
        return answer_is_match.group(1).strip()

    # Fallback: try to extract last number in the text
    numbers = re.findall(r'\b\d+\b', response_text)
    if numbers:
        return numbers[-1]

    return ''


def gather_sketchvlm_results(base_dir: str, model: str, use_generated: bool = False) -> List[Dict[str, str]]:
    """
    Gather SketchVLM results from a directory with item_*.json files.

    Args:
        base_dir: Directory containing item_*.json files
        model: Model name
        use_generated: If True, use generated image instead of annotated (for nano_banana format)

    Returns:
        List of dictionaries containing image_path, prompt, model_answer, and extracted_answer fields
    """
    entries = []
    base_path = Path(base_dir)

    # Find all item JSON files
    json_files = sorted(base_path.glob('item_*.json'))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Get the image path
            item_name = json_file.stem  # e.g., 'item_00000'

            if use_generated:
                # nano_banana format: use generated image
                # Try item_XXXXX_generated_0.jpeg first, then .png
                image_path = base_path / f"{item_name}_generated_0.jpeg"
                if not image_path.exists():
                    image_path = base_path / f"{item_name}_generated_0.png"

                # Also get original image path
                original_image_path = base_path / f"{item_name}_orig.jpg"
                if not original_image_path.exists():
                    original_image_path = base_path / f"{item_name}_orig.png"
            else:
                # Regular SketchVLM format: use annotated image
                image_path = base_path / f"{item_name}_annotated.png"
                # Also get original image path for quality evaluation
                original_image_path = base_path / f"{item_name}_orig.jpg"
                if not original_image_path.exists():
                    original_image_path = base_path / f"{item_name}_orig.png"

            # Get model output and answer
            model_answer = data.get('model_output_full', data.get('model_output', ''))
            extracted_answer = data.get('answer', '')

            # If no extracted answer in JSON, try to extract it
            if not extracted_answer:
                extracted_answer = extract_answer_from_response(model_answer)

            entry = {
                'image_path': str(image_path),
                # 'prompt': GENERAL_PROMPT + BALL_DROP_PROMPT,
                'prompt': BALL_QUALITY,
                # 'prompt': GEMINI_3_SECOND_TURN,
                'model_answer': model_answer,
                'extracted_answer': extracted_answer,
                'model': model,
                'item_index': data.get('index', None)
            }

            # Add original image path if it exists
            if original_image_path and original_image_path.exists():
                entry['original_image_path'] = str(original_image_path)

            entries.append(entry)

        except Exception as e:
            print(f"Warning: Could not process {json_file}: {e}")
            continue

    return entries


def gather_image_paths(base_dir: str, model: str = 'thinkmorph', last_image_only: bool = False) -> List[Dict[str, str]]:
    """
    Gather all image file paths from sample directories and extract model answers.
    Automatically detects SketchVLM format (item_*.json) or ThinkMorph/ViLaSR format (sample_*/images/).

    Args:
        base_dir: Base directory containing sample folders
        model: Model name (e.g., 'thinkmorph', 'gpt4', etc.)
        last_image_only: If True, only use the last image in each sample

    Returns:
        List of dictionaries containing image_path, prompt, model_answer, and extracted_answer fields
    """
    base_path = Path(base_dir)

    # Detect directory structure
    # SketchVLM format: has item_*.json files in the base directory
    if list(base_path.glob('item_*.json')):
        # Check if it's nano_banana format (has generated images)
        has_generated = bool(list(base_path.glob('*_generated_*.jpeg')) or list(base_path.glob('*_generated_*.png')))
        if has_generated:
            print("Detected nano_banana format (item_*.json with generated images)")
            return gather_sketchvlm_results(base_dir, model, use_generated=True)
            # return gather_sketchvlm_results(base_dir, model, use_generated=False)
        else:
            print("Detected SketchVLM format (item_*.json files)")
            return gather_sketchvlm_results(base_dir, model, use_generated=False)

    # ThinkMorph/ViLaSR format: has sample_* subdirectories
    entries = []

    # Find all sample directories
    sample_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('sample_')])

    for sample_dir in sample_dirs:
        images_dir = sample_dir / 'images'

        if not images_dir.exists():
            continue

        # Extract model answer from text_data.json
        text_data_file = sample_dir / 'text_data.json'
        model_answer = ''

        if text_data_file.exists():
            try:
                with open(text_data_file, 'r') as f:
                    data = json.load(f)

                # Extract answer - try text_outputs first (ThinkMorph format), then response (ViLaSR format)
                text_outputs = data.get('text_outputs', [])
                if text_outputs:
                    model_answer = '\n'.join(text_outputs)
                elif 'response' in data:
                    model_answer = data.get('response', '')
                else:
                    model_answer = ''
            except Exception as e:
                print(f"Warning: Could not read {text_data_file}: {e}")
                model_answer = ''

        # Find all image files (png, jpg, jpeg)
        image_files = sorted(images_dir.glob('*.png')) + \
                     sorted(images_dir.glob('*.jpg')) + \
                     sorted(images_dir.glob('*.jpeg'))

        # If last_image_only is True, only keep the last image
        if last_image_only and image_files:
            image_files = [image_files[-1]]

        # Extract the actual answer from model output
        extracted_answer = extract_answer_from_response(model_answer)

        # For ThinkMorph and ViLaSR, extract run ID to get original image
        original_image_path = None
        if 'thinkmorph' in str(base_path).lower():
            sample_name = sample_dir.name
            if 'run_b' in sample_name:
                # Batch2 format: run_b2_001
                parts = sample_name.split('_')
                for i, part in enumerate(parts):
                    if part.startswith('run'):
                        run_id = '_'.join(parts[i:i+3])  # run_b2_001
                        dataset_path = Path('/Users/log/Github/sketchvlm/datasets/second_batch_ball_path') / f'{run_id}.png'
                        if dataset_path.exists():
                            original_image_path = dataset_path
                        break
            elif 'run_' in sample_name:
                # Batch1 format: run_001 or run_001_1
                parts = sample_name.split('_')
                for i, part in enumerate(parts):
                    if part.startswith('run'):
                        # Try run_001_1 format (3 parts) first, then run_001 (2 parts)
                        if i+2 < len(parts) and parts[i+2].isdigit():
                            run_id = '_'.join(parts[i:i+3])  # run_001_1
                        else:
                            run_id = '_'.join(parts[i:i+2])  # run_001
                        dataset_path = Path('/Users/log/Github/sketchvlm/datasets/ball_path') / f'{run_id}.png'
                        if dataset_path.exists():
                            original_image_path = dataset_path
                        break
        elif 'vilasr' in str(base_path).lower():
            # For ViLaSR, read results.jsonl to map sample_dir to run_id
            results_file = base_path / 'results.jsonl'
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        for line in f:
                            data = json.loads(line)
                            if data.get('sample_dir', '').endswith(sample_dir.name):
                                image_path_str = data.get('image_path', [''])[0]
                                run_id = Path(image_path_str).stem
                                # Check batch2 path first, then batch1
                                if 'run_b' in image_path_str:
                                    # Batch2 format: run_b2_001
                                    dataset_path = Path('/Users/log/Github/sketchvlm/datasets/second_batch_ball_path') / f'{run_id}.png'
                                else:
                                    # Batch1 format: run_001
                                    dataset_path = Path('/Users/log/Github/sketchvlm/datasets/ball_path') / f'{run_id}.png'
                                if dataset_path.exists():
                                    original_image_path = dataset_path
                                break
                except Exception as e:
                    print(f"Warning: Could not read results.jsonl: {e}")

        for image_path in image_files:
            entry = {
                'image_path': str(image_path),
                # 'prompt': GENERAL_PROMPT + BALL_DROP_PROMPT,
                'prompt': BALL_QUALITY,
                'model_answer': model_answer,
                'extracted_answer': extracted_answer,
                'model': model
            }

            # Add original image path if it exists
            if original_image_path:
                entry['original_image_path'] = str(original_image_path)

            entries.append(entry)

    return entries


def main():
    parser = argparse.ArgumentParser(description='Generate questions JSON from model results')
    parser.add_argument('--model', type=str, default='thinkmorph',
                       help='Model name (default: thinkmorph)')
    parser.add_argument('--base-dir', type=str,
                       default='/Users/log/Github/sketchvlm/results/mix_eval/vpct/vpct_thinkmorph',
                       help='Base directory containing sample folders')
    parser.add_argument('--output-dir', type=str,
                       default='/Users/log/Github/sketchvlm/consistency',
                       help='Base output directory (will create source_data/ subdirectory)')
    parser.add_argument('--last-image-only', action='store_true',
                       help='Only use the last image in each sample')

    args = parser.parse_args()

    # Create source_data directory
    source_data_dir = os.path.join(args.output_dir, 'source_data')
    os.makedirs(source_data_dir, exist_ok=True)

    # Output JSON file path
    output_file = os.path.join(source_data_dir, f'image_questions_{args.model}.json')

    print(f"Model: {args.model}")
    print(f"Scanning directory: {args.base_dir}")
    print(f"Last image only: {args.last_image_only}")
    entries = gather_image_paths(args.base_dir, args.model, args.last_image_only)
    print(f"Found {len(entries)} images")

    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(entries, f, indent=2)

    print(f"Saved to: {output_file}")
    print(f"\nSample entry:")
    if entries:
        print(json.dumps(entries[0], indent=2))


if __name__ == '__main__':
    main()
