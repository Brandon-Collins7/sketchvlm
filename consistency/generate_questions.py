import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict


# Standard prompts for consistency checking
GENERAL_PROMPT = "You are given an image that another AI model has annotated. Your task is to analyze the annotation and determine what the final answer should be.\n\n"
BALL_DROP_PROMPT = "The image is a physics simulation of a ball being dropped. There are 4 different buckets called bucket 1, bucket 2, bucket 3, and bucket 4. Please respond with what bucket the **annotation shows** the ball will fall into. Please note that it is not your job to determine if the annotation is correct or not, just to extract the answer from the annotation. Your final answer must be formatted as \"$\\boxed{bucket number}$\". For example, if the ball will fall into bucket 2, respond with \"$\\boxed{2}$\". If the annotation does not show the ball falling into any bucket, respond with \"$\\boxed{none}$\". If the the annotation shows that the ball will fall into multiple different buckets, answer with \"$\\boxed{multiple}$\"."
VPCT_BALL_DROP_PROMPT = "The ball is released from rest, the only force it is subject to is gravity. The black lines are walls and platforms and the ball cannot pass through them. Put your answer of which container it will land in, 1, 2, or 3 (numbered left to right)."
MAZE_PROMPT = "You are given an image of a maze where the green square marks the START cell and the red square marks the END cell of the maze. The walls of the maze are solid black lines. Dashed gray lines mark cell boundaries that can be crossed. You are given a proposed sequence of moves to reach the end of the maze starting from the green square and ending at the red square. Each move will move exactly one cell length in that direction. For example, \"right\" means move one cell in the maze to the right. A valid path must NOT cross any solid black walls and must end up in the red square cell. A valid path can also move through any of the dashed gray cell lines. Respond with $\\boxed{valid}$ if the path is valid or respond with $\\boxed{invalid}$ if the path is invalid. Determine if the following proposed path is valid.\n\n"

DONT_HALLUCINATE_PROMPT = "Keep in mind that your final answer should only be based off the annotated image. Only look at the path that is taken when making your decision. Be careful to make sure that you do not hallucinate moves that do not exist in the image."

BALL_QUALITY = """# Overall Instructions
You will be shown two images: the original image and an AI-annotated version. The annotation is supposed to show how the path the ball will take when it is dropped and where it will eventually end up. Your job is to grade the quality of the sketch using a rubric. You should grade only based on the following attributes that are given.

A minor issue is a violation that has little effect on the final answer of what bucket the ball will fall into. For example, if the ball path slightly clips through the walls, but the trajectory of the path still makes sense, then this is only a minor issue and not a major issue. A major issue is a violation that has a significant effect on the final answer of what bucket the ball will fall into. For example, if the ball path clips through the walls and the trajectory of the path is not logical, then this is a major issue. If there is no sketch provided, then the score should be a 1. The score should be based on the scale of 1 - 5

# Quality Checks

Below are more details:

## Things that are bad
1) The drawn path clips through any of the black walls.
2) The drawn path has very unrealistic physics. For example, the ball path momentarily disobeys gravity by moving upwards or moving in a direction that is not logical for gravity.
3) The drawn path substantially alters the original image (like adding or removing walls)
4) The drawn path contains multiple different paths instead of a single path.

## Things that are good
1) The drawn path is a single path that clearly shows where the final resting position of the ball will be.
2) The drawn path does not cut through any of the black walls.

## Things that are okay
1) If the path slightly clips through the walls, but the trajectory of the path still makes sense, then this is only a minor issue and not a major issue. We want to heavily penalize the ball going through walls and following a totally illogical path.
2) It's okay if the drawn path does not start exactly at the origin of the ball. As long as it is close, then this is not a minor or major issue.



# Scoring breakdown

1) The sketch has several critical flaws
2) The sketch has a critical flaw
3) The sketch contains some errors, overall direction of the path is valid and makes sense.
4) The sketch contains one minor logical error.
5) The sketch contains zero errors.

# Output Format
You should follow this output format EXACTLY with no other output:

{reasoning for logical consistency score}
Quality Score: {integer from 1 - 5}

# Example Output

<example_1>
The drawing contains multiple errors. The ball path barely clips through one of the platforms, but it is a minor issue as the path of the ball still makes sense. Additionally, the ball path defies gravity and begins to momentarily float straight upwards for no logical reason. The minor error combined with the critical error results in a logical score of 2/5.
Quality Score: 2
</example_1>

<example_2>
Overall, the path the ball takes is logical. It properly reflects off the first wall and then rolls to the right. Then, it falls down to the left and lands in bucket 2. The ball path slightly clips through one of the black edges, but this is a minor issue because the big picture trajectory of the ball is still logical. Given the overall path makes sense and there was just one minor issue, the quality score is 4/5.
Quality Score: 4
</example_2>
"""

MAZE_QUALITY = """# Overall Instructions
You will be shown two images: the original image and an AI-annotated version. Your job is to grade the quality of the sketch using a rubric. You should grade only based on the following attributes that are given.

# Sketch Instructions
The original sketch was drawn according to the following prompt:

"You are given an image of a maze where the green square marks the START cell and the red square marks the END cell of the maze. The walls of the maze are solid black lines. Dashed gray lines mark cell boundaries that can be crossed. You are given a proposed sequence of moves to reach the end of the maze starting from the green square and ending at the red square. Each move will move exactly one cell length in that direction. For example, "right" means move one cell in the maze to the right. A valid path must NOT cross any solid black walls and must end up in the red square cell. A valid path can also move through any of the dashed gray cell lines. The grid on the outside of the image is only there to help provide a reference for you. Moving one right means go one big cell in the maze right."

# Quality Checks

The main things that you should be looking for are:
1) Clipping through walls when it is not required to.
2) The drawn path does not go to the CENTER of each cell that it goes through.
3) The drawn path contradicts the given text path.

Below are more details:

## Things that are bad
1) The drawn path clips through any of the black walls when it is not required to. For example, even if the directions of the drawn path are correct, if the path touches or goes through a wall, then it is a bad sketch. That means that if the path goes through a wall even when it is not absolutely required to, then it is a bad sketch.
2) Each move in the drawn path should go to the **center** of the next cell in the path. If the drawn path is a curved path, then this does not apply. This is important! Look at each step in the path and make sure that the drawn path goes to the center of the next cell.
3) The sketch contains additional moves that are not in the path
4) The drawn sketch contradicts the given text path.
5) Even if the directions of the drawn path are correct, the end of the path does not end up touching the red square.
6) The drawn path does not start by touching the green square

## Things that are not an issue
1) If the proposed path is not valid, the drawn sketch shows exactly what the path should look like (even if it has to clip through walls or double back on itself).
2) If the proposed path is not valid, and the drawing ends as soon as there is an invalid move that is taken (such as requiring to go through a wall), then that is not an issue. It's okay for the drawing to not show all the steps of the path here since the sketch is emphasizing that the path is invalid.

## Score breakdown

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
The drawing contains multiple errors. The drawn path goes up, up, left instead of up, up, right. This contradicts the given text path. Additionally, the end of the drawn path slightly clips through the solid black wall. Additionally, the draw path does NOT go to the center of each cell that it goes through. Instead, it only moves about 60 percent of the way across each cell and doesn't therefore doesn't follow the grid logic of moving from one center of each cell to the center of the next cell. The minor error combined with the multiple critical errors results in a score of 1/5.

Quality Score: 1
</example_1>

The original proposed path that the model should have followed is:

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

            # Extract proposed path from original prompt if it exists
            proposed_path = ""
            original_prompt = data.get('prompt', '')
            if 'Proposed path:' in original_prompt:
                # Extract everything after "Proposed path:"
                path_match = re.search(r'Proposed path:\s*(.+?)(?:\n|$)', original_prompt)
                if path_match:
                    proposed_path = "\n\nProposed path: " + path_match.group(1).strip()

                    # Add valid/invalid instruction based on directory name
                    # Check for "invalid" first since "valid" is a substring of "invalid"
                    if 'invalid' in str(base_dir).lower():
                        proposed_path += "\n\nThe draw path here should be INVALID. that means that the path should NOT be able to start from the green square and end up at the red square."
                    elif 'valid' in str(base_dir).lower():
                        proposed_path += "\n\nThe draw path here should be VALID. that means that the path should be able to start from the green square and end up at the red square. It should not clip through any of the walls."

            entry = {
                'image_path': str(image_path),
                # 'prompt': GENERAL_PROMPT + BALL_DROP_PROMPT,
                # 'prompt': GENERAL_PROMPT + VPCT_BALL_DROP_PROMPT,
                # 'prompt': GENERAL_PROMPT + MAZE_PROMPT + proposed_path,
                # 'prompt': GENERAL_PROMPT + MAZE_PROMPT,
                # 'prompt': GENERAL_PROMPT + MAZE_PROMPT + DONT_HALLUCINATE_PROMPT,
                # 'prompt': BALL_QUALITY,
                # 'prompt': GEMINI_3_SECOND_TURN,
                'prompt': MAZE_QUALITY + proposed_path,
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
        proposed_path = ""

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

                # Extract proposed path from prompt if it exists (for maze tasks)
                original_prompt = data.get('prompt', '')
                if 'Proposed path:' in original_prompt:
                    path_match = re.search(r'Proposed path:\s*(.+?)(?:\n|$)', original_prompt)
                    if path_match:
                        proposed_path = "\n\nProposed path: " + path_match.group(1).strip()

                        # Add valid/invalid instruction based on directory name
                        # Check for "invalid" first since "valid" is a substring of "invalid"
                        if 'invalid' in str(base_path).lower():
                            proposed_path += "\n\nThe draw path here should be INVALID. that means that the path should NOT be able to start from the green square and end up at the red square."
                        elif 'valid' in str(base_path).lower():
                            proposed_path += "\n\nThe draw path here should be VALID. that means that the path should be able to start from the green square and end up at the red square. It should not clip through any of the walls."
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

        # For ThinkMorph and ViLaSR, extract ID to get original image
        original_image_path = None
        sample_name = sample_dir.name

        # For ViLaSR, always read results.jsonl to get original image path (works for both maze and ball tasks)
        if 'vilasr' in str(base_path).lower():
            results_file = base_path / 'results.jsonl'
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        for line in f:
                            data_line = json.loads(line)
                            if data_line.get('sample_dir', '').endswith(sample_dir.name):
                                image_path_str = data_line.get('image_path', [''])[0]
                                file_id = Path(image_path_str).stem

                                # Check if it's a maze task (contains 'maze_')
                                if 'maze_' in image_path_str:
                                    # Determine if invalid or valid based on path
                                    if 'invalid' in str(base_path).lower():
                                        dataset_path = Path('/Users/log/Github/sketchvlm/datasets/maze_v2/invalid_flattened') / f'{file_id}.png'
                                    else:
                                        dataset_path = Path('/Users/log/Github/sketchvlm/datasets/maze_v2/valid_flattened') / f'{file_id}.png'
                                # Otherwise it's a ball path task
                                elif 'run_b' in image_path_str:
                                    # Batch2 format: run_b2_001
                                    dataset_path = Path('/Users/log/Github/sketchvlm/datasets/second_batch_ball_path') / f'{file_id}.png'
                                else:
                                    # Batch1 format: run_001
                                    dataset_path = Path('/Users/log/Github/sketchvlm/datasets/ball_path') / f'{file_id}.png'

                                if dataset_path.exists():
                                    original_image_path = dataset_path
                                break
                except Exception as e:
                    print(f"Warning: Could not read results.jsonl: {e}")

        # For ThinkMorph
        elif 'thinkmorph' in str(base_path).lower():
            # Check if this is a maze task (contains 'maze_' in sample name)
            if 'maze_' in sample_name:
                # Maze format: extract maze_ID from sample name
                # e.g., sample_20251203_185642_maze_100_fbcdb0b4 -> maze_100_fbcdb0b4
                maze_match = re.search(r'(maze_\d+_[a-f0-9]+)', sample_name)
                if maze_match:
                    maze_id = maze_match.group(1)
                    # Determine if invalid or valid based on path
                    if 'invalid' in str(base_path).lower():
                        dataset_path = Path('/Users/log/Github/sketchvlm/datasets/maze_v2/invalid_flattened') / f'{maze_id}.png'
                    else:
                        dataset_path = Path('/Users/log/Github/sketchvlm/datasets/maze_v2/valid_flattened') / f'{maze_id}.png'
                    if dataset_path.exists():
                        original_image_path = dataset_path
            # Ball path format: extract run_ID from sample name
            elif 'run_' in sample_name:
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
                else:
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

        for image_path in image_files:
            # Use consistency checking prompt (same as SketchVLM format)
            prompt = GENERAL_PROMPT + BALL_DROP_PROMPT

            entry = {
                'image_path': str(image_path),
                'prompt': prompt,
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Output JSON file path
    output_file = os.path.join(args.output_dir, f'{args.model}.json')

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
