"""
Process images with OpenRouter API for consistency checking.

This script reads a JSON file containing image paths and prompts,
sends each image to the OpenRouter API with the specified model,
and saves the results incrementally (after each request).

Usage:
    export OPENROUTER_API_KEY="your-api-key-here"

    python process_with_openrouter.py \\
        --input consistency/image_questions_thinkmorph.json \\
        --output consistency/consistency_results_thinkmorph.json

    # Resume from a specific index:
    python process_with_openrouter.py \\
        --input consistency/image_questions_thinkmorph.json \\
        --output consistency/consistency_results_thinkmorph.json \\
        --start-index 50

Features:
    - Incremental saving: Results are saved after each API call
    - Resume capability: Can resume from any index if interrupted
    - Error handling: Continues processing even if individual requests fail
    - Rate limiting: Automatic delay between requests
"""

import os
import json
import base64
import argparse
import time
from pathlib import Path
from typing import Dict, List
import requests


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded image string
    """
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_openrouter_api(image_path: str, prompt: str, api_key: str, model: str, original_image_path: str = None) -> Dict:
    """
    Call OpenRouter API with image and prompt.

    Args:
        image_path: Path to the image file
        prompt: Text prompt for the model
        api_key: OpenRouter API key
        model: Model identifier

    Returns:
        Dictionary containing the response and metadata
    """
    # Check if image exists
    if not os.path.exists(image_path):
        return {
            'success': False,
            'response': None,
            'error': f'Image file not found: {image_path}'
        }

    # Encode image to base64
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        return {
            'success': False,
            'response': None,
            'error': f'Failed to read image: {str(e)}'
        }

    # Determine image format
    image_ext = Path(image_path).suffix.lower()
    mime_type = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg'
    }.get(image_ext, 'image/png')

    # Prepare the request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Build content array
    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    # Add original image if provided
    if original_image_path and os.path.exists(original_image_path):
        base64_original = encode_image_to_base64(original_image_path)
        original_ext = Path(original_image_path).suffix.lower()
        original_mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(original_ext, 'image/png')
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{original_mime};base64,{base64_original}"
            }
        })

    # Add main image
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{base64_image}"
        }
    })

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "provider": {
            "order": ["google-ai-studio"],
            "allow_fallbacks": False
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        response_data = response.json()

        # Extract the response text
        if 'choices' in response_data and len(response_data['choices']) > 0:
            response_text = response_data['choices'][0]['message']['content']
            return {
                'success': True,
                'response': response_text,
                'error': None
            }
        else:
            return {
                'success': False,
                'response': None,
                'error': 'No response from model'
            }

    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'response': None,
            'error': str(e)
        }


def process_entries(input_file: str, output_file: str, api_key: str, model: str, start_index: int = 0):
    """
    Process entries from input JSON and save results incrementally.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        api_key: OpenRouter API key
        model: Model identifier
        start_index: Index to start processing from (for resuming)
    """
    # Load input data
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    # Load existing output data if it exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            output_data = json.load(f)
        print(f"Loaded existing output with {len(output_data)} entries")
    else:
        output_data = []

    # Process each entry
    total_entries = len(input_data)
    for i in range(start_index, total_entries):
        entry = input_data[i]

        print(f"\n[{i+1}/{total_entries}] Processing: {entry['image_path']}")

        # Call API
        result = call_openrouter_api(
            image_path=entry['image_path'],
            prompt=entry['prompt'],
            api_key=api_key,
            model=model,
            original_image_path=entry.get('original_image_path')
        )

        # Create output entry
        output_entry = {
            'index': i,
            'image_path': entry['image_path'],
            'prompt': entry['prompt'],
            'original_model': entry.get('model', ''),
            'original_model_answer': entry.get('model_answer', ''),
            'original_extracted_answer': entry.get('extracted_answer', ''),
            'consistency_check_model': model,
            'consistency_check_response': result['response'],
            'success': result['success'],
            'error': result['error']
        }

        # Add or update entry in output data
        if i < len(output_data):
            output_data[i] = output_entry
        else:
            output_data.append(output_entry)

        # Save immediately after each request
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        if result['success']:
            print(f"✓ Success - Response length: {len(result['response'])} chars")
        else:
            error_msg = result['error']
            if 'Image file not found' in error_msg:
                print(f"⚠ Warning - Image not found, skipping")
            else:
                print(f"✗ Failed - Error: {error_msg}")

        # Rate limiting - wait between requests
        if i < total_entries - 1:  # Don't wait after the last request
            time.sleep(1)  # Adjust based on rate limits

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total entries: {total_entries}")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Process images with OpenRouter API')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file from generate_questions.py')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results (default: auto-generated in judge_output/)')
    parser.add_argument('--model', type=str, default='google/gemini-3-flash-preview',
                       help='Model identifier (default)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Index to start processing from (for resuming)')

    args = parser.parse_args()

    # Get API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable must be set")
        return

    # Auto-generate output path if not provided
    if args.output is None:
        # Extract model name from input file path
        # e.g., consistency/source_data/image_questions_thinkmorph.json -> thinkmorph
        input_path = Path(args.input)
        input_filename = input_path.stem  # e.g., image_questions_thinkmorph
        model_name = input_filename.replace('image_questions_', '')

        # Create judge_output directory
        base_dir = input_path.parent.parent  # Go up from source_data to consistency
        judge_output_dir = base_dir / 'judge_output'
        judge_output_dir.mkdir(exist_ok=True)

        output_file = str(judge_output_dir / f'consistency_results_{model_name}.json')
    else:
        output_file = args.output

    print(f"Input file: {args.input}")
    print(f"Output file: {output_file}")
    print(f"Model: {args.model}")
    print(f"Starting from index: {args.start_index}")

    process_entries(
        input_file=args.input,
        output_file=output_file,
        api_key=api_key,
        model=args.model,
        start_index=args.start_index
    )


if __name__ == '__main__':
    main()
