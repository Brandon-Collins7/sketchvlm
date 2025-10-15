
# Architecture

  - llm_adapters.py - LLM provider integrations
  - grid_manager.py - Dynamic grid and image placement
  - collab_sketch_with_label.py - Core sketch application logic
  - utils.py - Utility functions
  - prompts.py - LLM prompts and templates


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
