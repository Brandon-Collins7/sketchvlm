
# Architecture

  - llm_adapters.py - LLM provider integrations
  - grid_manager.py - Dynamic grid and image placement
  - collab_sketch_with_label.py - Core sketch application logic
  - utils.py - Utility functions
  - prompts.py - LLM prompts and templates


# **Frequently Used**

**Inference:**

**GPT-5 (medium) - Connect Dots**

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir {dataset_directory} --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20   

Ex:

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir connecting_dots_dataset/worksheets_source --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20   


**GPT-5 (medium) - Physics ball drop (exact same but change dataset)**

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir {dataset_directory} --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20   

Ex:

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir datasets/ball_path --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20  

To get the ball_path dataset (so it's formatted in same way as other datasets) from the large_run_split, run gather_ball_prompts.py. This will create almost two equivalent folders, where one just adds a small bit to the prompt that requires the model to sketch the path.


**Gemini-2.5-Pro - Connect Dots (just changed model)**

python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --mixed-dir {dataset_directory} --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 

**Gemini-2.5-Pro - Physics ball drop**

python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --mixed-dir {dataset_directory} --max-tokens 10000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20


-----

**Comparisons / Accuracy:**

Sometimes will take a minute for html to finish generating. Htmls include accuracy

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




# Code Execution Examples

## Model Types
### Claude
python sketch_app.py --llm claude --model claude-3-5-sonnet-20240620

### OpenAI GPT
python sketch_app.py --llm gpt --model o3

### Gemini 2.5 Pro
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro


## New Grid Fix (for higher res images)

example:

python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --mixed-dir datasets/ball_path --count-only-text --max-tokens 10000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20        


## Task Types
sketch_app.py is now collab_sketch_all.py

### Counting

#### Claude example:
python collab_sketch_all.py --llm claude --model claude-3-5-sonnet-20240620 --eval-dataset vikhyatk/CountBenchQA --eval-split test --count-only-text

#### OpenAI example:
python collab_sketch_all.py --llm gpt --model o3 --eval-dataset vikhyatk/CountBenchQA --eval-split test --count-only-text

####
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --eval-dataset vikhyatk/CountBenchQA --eval-split test --count-only-text --api-delay 4


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
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --eval-stepwise --eval-dataset vikhyatk/CountBenchQA --count-only-text --max-examples 100 --api-delay 0.2


---- Mixed testing -----

#all strokes in one call
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --mixed-dir datasets/mix --count-only-text

#one stroke per turn
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --mixed-dir datasets/mix --mixed-stepwise --mixed-max-turns 40 --count-only-text --api-delay 0.2


--- TallyQA


python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --tallyqa-json TallyQA_dataset/test_sample_500.json --vg-root data --tallyqa-outdir results/tallyqa_eval --max-examples 200 --count-only-text --api-delay 5

#multi-turn
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --tallyqa-json TallyQA_dataset/test_sample_500.json --vg-root data --tallyqa-outdir results/tallyqa_eval --tallyqa-stepwise --tallyqa-max-turns 40 --max-examples 200 --count-only-text --api-delay 4
