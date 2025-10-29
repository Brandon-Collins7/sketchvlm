#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import sys

'''
PROMPT_DRAW = (
    "Draw the path that the ball will take. It will either land in container 1, 2, 3, or 4, "
    "or it will get stuck before then. The ball is released from rest, the only force it is subject to is gravity."
    "The ball cannot be compressed so it must not be too big to fit between gaps. The black lines are walls and platforms. "
    "The path cannot pass through the black lines, and the ball cannot pass through the black lines."
    
    "After emitting the strokes, put your answer of which container it will land in, 1, 2, 3, or 4, or 0 if it gets stuck before reaching any."
    "Put your answer in an answer tag at the end like <answer>2</answer>."
    
'''

#PROMPT_DRAW = """You are given the start frame of a physics simulation. A ball is dropped from the top of the screen and falls due to gravity. The ball can roll off the lines or the walls in the image. The bouncing of the ball is relatively minor and realistic for normal gravity. Nothing in the image will move besides the ball. Predict which bucket will eventually catch the ball. There are 4 different buckets called bucket 1, bucket 2, bucket 3, and bucket 4. Please respond with what bucket the ball will fall into. Your final answer must be formatted as "$\\boxed{bucket number or none}$". For example, if the ball will fall into bucket 2, respond with "$\\boxed{2}$"."""


#straight lines only
#PROMPT_DRAW = """You are given the start frame of a physics simulation. A ball is dropped from the top of the screen and falls due to gravity. The ball can roll off the lines or the walls in the image. The bouncing of the ball is relatively minor and realistic for normal gravity. Nothing in the image will move besides the ball. Predict which bucket will eventually catch the ball. There are 4 different buckets called bucket 1, bucket 2, bucket 3, and bucket 4. Draw the path only using straight lines. Please also respond with what bucket the ball will fall into. Your final answer must be formatted as "$\\boxed{bucket number or none}$". For example, if the ball will fall into bucket 2, respond with "$\\boxed{2}$"."""

PROMPT_DRAW = """You are given the start frame of a physics simulation. A ball is dropped from the top of the screen and falls due to gravity. The ball can roll off the lines or the walls in the image. The bouncing of the ball is relatively minor and realistic for normal gravity. Nothing in the image will move besides the ball. Predict which bucket will eventually catch the ball. There are 4 different buckets called bucket 1, bucket 2, bucket 3, and bucket 4. Draw the path that the ball will take. Please also respond with what bucket the ball will fall into. Your final answer must be formatted as "$\\boxed{bucket number}$". For example, if the ball will fall into bucket 2, respond with "$\\boxed{2}$"."""


PROMPT_NUMBER = (
    """You are given the start frame of a physics simulation. A ball is dropped from the top of the screen and falls due to gravity. The ball can roll off the lines or the walls in the image. The bouncing of the ball is relatively minor and realistic for normal gravity. Nothing in the image will move besides the ball. Predict which bucket will eventually catch the ball. There are 4 different buckets called bucket 1, bucket 2, bucket 3, and bucket 4. Please respond with what bucket the ball will fall into. Your final answer must be formatted as "$\\boxed{bucket number}$". For example, if the ball will fall into bucket 2, respond with "$\\boxed{2}$"."""
)

def collect_runs(src_root: Path, image_name: str):
    runs = []
    for p in sorted(src_root.glob("run_*")):
        img = p / image_name
        if img.is_file():
            runs.append((p.name, img))
    return runs

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def copy_with_prompt(run_name: str, src_img: Path, dest_dir: Path, prompt: str):
    ensure_dir(dest_dir)
    stem = run_name  # e.g., "run_001"
    dest_png = dest_dir / f"{stem}.png"
    dest_txt = dest_dir / f"{stem}.txt"

    shutil.copy2(src_img, dest_png)
    dest_txt.write_text(prompt + "\n", encoding="utf-8")
    return dest_png, dest_txt

def main():
    parser = argparse.ArgumentParser(description="Collect run_* images and create paired prompt files.")
    parser.add_argument("--src", default="datasets/large_run_split", help="Source root folder containing run_* subfolders")
    parser.add_argument("--image-name", default="random_scene_start.png", help="Image file name inside each run folder")
    parser.add_argument("--out-draw", default="datasets/ball_path", help="Output folder for 'draw the path' prompt")
    parser.add_argument("--out-number", default="datasets/ball_number", help="Output folder for 'number only' prompt")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_draw = Path(args.out_draw)
    out_number = Path(args.out_number)

    if not src_root.exists():
        print(f"Source root not found: {src_root}", file=sys.stderr)
        sys.exit(1)

    runs = collect_runs(src_root, args.image_name)
    if not runs:
        print(f"No runs found with image '{args.image_name}' under {src_root}", file=sys.stderr)
        sys.exit(2)

    created = []
    for run_name, img_path in runs:
        p1 = copy_with_prompt(run_name, img_path, out_draw, PROMPT_DRAW)
        p2 = copy_with_prompt(run_name, img_path, out_number, PROMPT_NUMBER)
        created.append((run_name, p1, p2))

    print(f"Processed {len(created)} run(s).")
    print(f"- Wrote images & prompts to: {out_draw.resolve()}")
    print(f"- Wrote images & prompts to: {out_number.resolve()}")

if __name__ == "__main__":
    main()
