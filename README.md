
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
python collab_sketch_all.py --llm gemini --model gemini-2.5-pro


## Task Types
sketch_app.py is now collab_sketch_all.py

### Counting

#### Claude example:
python collab_sketch_all.py --llm claude --model claude-3-5-sonnet-20240620 --eval-dataset vikhyatk/CountBenchQA --eval-split test --count-only-text

#### OpenAI example:
python collab_sketch_all.py --llm gpt --model o3 --eval-dataset vikhyatk/CountBenchQA --eval-split test --count-only-text

####
python collab_sketch_all.py --llm gemini --model gemini-2.5-pro --eval-dataset vikhyatk/CountBenchQA --eval-split test --count-only-text


## Labeling

### Gemini 2.5 Pro, label everything in datasets/labeling, concept from filenames:
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --api-delay 0.0 --label-dir datasets/labeling
python collab_sketch_label_line.py --llm gemini --model gemini-2.5-pro --api-delay 0.0 --label-dir datasets/labeling

### …or force a constant concept for every image + custom hints:
python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --api-delay 0.0 \
  --label-dir datasets/labeling \
  --concept-mode constant --concept "human body" \
  --labels-hint "head, torso, left_arm, right_arm, left_leg, right_leg, hand, foot"

