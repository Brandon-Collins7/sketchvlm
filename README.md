
# Architecture

  - llm_adapters.py - LLM provider integrations
  - grid_manager.py - Dynamic grid and image placement
  - collab_sketch_with_label.py - Core sketch application logic
  - utils.py - Utility functions
  - prompts.py - LLM prompts and templates

# render_strokes_postprocess.py args

--results-dir path/to/results - results folder to read from

--base grid - will overlay on image with grid

--base orig - will overlay on image without grid

--origin bottom-left - overlay assuming bottom left origin

--origin top-left - overlay assuming top left origin

--res-x 1000 --res-y 1000 - use if results were generated with the --no-grid --res-x 1000 --res-y 1000 approach

--only "0,1,2,3" - will only render strokes for each index in the comma separated indices (0-based indexing)

Examples:

Render onto the grid image (easy path, uses cell_pixel_map directly):

python render_strokes_postprocess.py \
  --results-dir results/mix_eval/20251221_210419 \
  --base grid \
  --origin bottom-left \
  --only "1,2,3"

Render onto the orig image (grid→orig mapping happens automatically if cell_pixel_map exists):

python render_strokes_postprocess.py \
  --results-dir results/mix_eval/20251221_210419 \
  --base orig \
  --origin bottom-left \

If your no-grid run used res_x=res_y=1000 coords:

python render_strokes_postprocess.py \
  --results-dir results/mix_eval/20251221_210419 \
  --base orig \
  --origin top-left \
  --res-x 1000 --res-y 1000

# collab_sketch_with_label.py args (to be finished)

--save-annotated-no-grid - save an annotated image without the grid background (in addition to the normal annotated image)

--no-grid - removes grid from the image that is sent to model

--no-system-prompt - removes sketch system prompt from being sent to model

--adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 - better grid scaling for different resolutions

--max-tokens {number} - set max tokens per call

--only "0,1,2,3" - will only run for each index in the comma separated indices (0-based indexing)

--llm --model - see in a below section for model choices


# Better Setup for Gemini-3-Pro

--no-grid --res-x 1000 --res-y 1000

--res-x determines number of cells on x-axis (columns)
--res-y determines number of cells on y-axis (rows)

Ex:

python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-preview --mixed-dir datasets/vpct_ball_drop --max-tokens 20000 --no-grid --res-x 1000 --res-y 1000 --prompt-origin top_left

# Run with Specific Model 

## GPT-5

### GPT-5 (low)
python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir path/to/dataset --max-tokens 20000 --reasoning-effort low --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20   

### GPT-5 (medium)
python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir path/to/dataset --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20   

### GPT-5 (high)
python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir path/to/dataset --max-tokens 20000 --reasoning-effort high --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20   

## Gemini 

### Gemini 3.0 Pro
python collab_sketch_with_label.py --llm gemini --model gemini-3-pro-preview --mixed-dir path/to/dataset --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

### Gemini 2.5 Pro
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --mixed-dir path/to/dataset --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

### Gemini 2.5 Flash
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-flash --mixed-dir path/to/dataset --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

## Claude
python collab_sketch_with_label.py --llm claude --model claude-3-5-sonnet-20240620 --mixed-dir path/to/dataset --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20   


**Inference:**

To get the ball_path dataset (so it's formatted in same way as other datasets) from the large_run_split, run gather_ball_prompts.py. This will create almost two equivalent folders, where one just adds a small bit to the prompt that requires the model to sketch the path.


-----


**Comparisons / Accuracy:**

\***Html compare + accuracy - Physics ball drop; VQA (no sketch) vs Sketch**

python ball_drop_compare_report.py --gt-root large_run_split --raw-jsonl results/mix_eval/ball_number_gpt5_medium.jsonl --grid-dir results/mix_eval/gpt_ball_drop_curves --out gpt_ball_drop_report.html --thumb-width 480

\***Html compare + accuracy - Physics ball drop; Sketch1 vs Sketch2**

python ball_drop_compare_report.py --gt-root large_run_split --grid-dir results/mix_eval/gpt_ball_drop_curves --grid-dir-b results/mix_eval/gpt_ball_drop_lines --grid-a-name "Curves + lines" --grid-b-name "Straight lines only" --no-raw --out meta_ball_better.html --thumb-width 480



*Grid directory results are assumed to be in the file layout that would be saved after running inference

The raw jsonl needed for html compare (where raw result is when doing VQA i.e. no sketch) is in format:

{"index": 0, "file": "datasets\\ball_number\\run_001_1.png", "prompt": "Bucket Physics (boxed)", "model": "openai::gpt-5-medium", "raw_text": "\\boxed{1}", "parsed_label": "1", "parsed_int": 1, "gold": null, "gold_int": null, "correct": null}

{"index": 1, "file": "datasets\\ball_number\\run_004.png", "prompt": "Bucket Physics (boxed)", "model": "openai::gpt-5-medium", "raw_text": "$\\boxed{3}$", "parsed_label": "3", "parsed_int": 3, "gold": null, "gold_int": null, "correct": null}

... etc. new line for each sample

Where really just the "file" variable path (to match the result with correct gt) and the "parsed_label" is needed for each row.
This is shoddy and should probably be made better.



## Task Types
sketch_app.py is now collab_sketch_all.py


## Labeling

### Gemini 2.5 Pro, label everything in datasets/labeling, concept from filenames:
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --api-delay 0.0 --label-dir datasets/labeling

### …or force a constant concept for every image + custom hints:
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --api-delay 0.0 \
  --label-dir datasets/labeling \
  --concept-mode constant --concept "human body" \
  --labels-hint "head, torso, left_arm, right_arm, left_leg, right_leg, hand, foot"

---- Eval one stroke at a time ----

python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --count-stepwise-dir datasets/biased  --count-only-text --count-stepwise-max-turns 30 --api-delay 0.2          

---- Mixed testing -----

#all strokes in one call
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --mixed-dir datasets/mix --count-only-text

#one stroke per turn
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --mixed-dir datasets/mix --mixed-stepwise --mixed-max-turns 40 --count-only-text --api-delay 0.2
