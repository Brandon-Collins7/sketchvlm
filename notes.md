
# Architecture

  - llm_adapters.py - LLM provider integrations
  - grid_manager.py - Dynamic grid and image placement
  - collab_sketch_with_label.py - Core sketch application logic
  - utils.py - Utility functions
  - prompts.py - LLM prompts and templates


# **Frequently Used**


**Baseline:**

--no-grid

removes grid from the image that is sent to model


--no-system-prompt

removes system prompt from being sent to model



**Inference:**

**GPT-5 (medium) - Connect Dots**

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir {dataset_directory} --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20   

Ex:

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir connecting_dots_dataset/worksheets_source --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20   


**GPT-5 (medium) - Physics ball drop (exact same but change dataset)**

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir {dataset_directory} --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20   

Ex:

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir datasets/ball_path --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir datasets/second_batch_ball_path --max-tokens 20000 --reasoning-effort low --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir datasets/second_batch_ball_number --max-tokens 20000 --reasoning-effort low --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

To get the ball_path dataset (so it's formatted in same way as other datasets) from the large_run_split, run gather_ball_prompts.py. This will create almost two equivalent folders, where one just adds a small bit to the prompt that requires the model to sketch the path.


**Gemini-2.5-Pro - Connect Dots (just changed model)**

python collab_sketch_with_label.py --llm gemini --model gemini-2.5-pro --mixed-dir {dataset_directory} --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 

**Gemini-2.5-Pro - Physics ball drop**

python collab_sketch_with_label.py --llm gemini --model gemini-2.5-flash --mixed-dir datasets/second_batch_ball_path --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20




python collab_sketch_with_label.py --llm gemini --model gemini-2.5-flash --mixed-dir datasets/maze_v2/sketch_invalid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

python collab_sketch_with_label.py --llm gemini --model gemini-2.5-flash --mixed-dir datasets/maze_v2/sketch_valid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20


python collab_sketch_with_label.py --llm gemini --model gemini-2.5-flash --mixed-dir datasets/maze_v2/invalid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid

python collab_sketch_with_label.py --llm gemini --model gemini-2.5-flash --mixed-dir datasets/maze_v2/valid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid



python collab_sketch_with_label.py --llm gemini --model gemini-3-pro-preview --mixed-dir datasets/maze_v2/valid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid

python collab_sketch_with_label.py --llm gemini --model gemini-3-pro-preview --mixed-dir datasets/maze_v2/valid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid

python collab_sketch_with_label.py --llm gemini --model gemini-3-pro-preview --mixed-dir datasets/maze_v2/sketch_valid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

python collab_sketch_with_label.py --llm gemini --model gemini-3-pro-preview --mixed-dir datasets/maze_v2/sketch_invalid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --only "164"



python collab_sketch_with_label.py --llm gemini --model gemini-3-pro-preview --mixed-dir datasets/maze_v2/invalid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid --only "114,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199"



python collab_sketch_with_label.py --llm gpt --model gpt-5 --reasoning-effort low --mixed-dir datasets/maze_v2/sketch_invalid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

python collab_sketch_with_label.py --llm gpt --model gpt-5 --reasoning-effort low --mixed-dir datasets/maze_v2/sketch_valid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20


python collab_sketch_with_label.py --llm gpt --model gpt-5 --reasoning-effort low --mixed-dir datasets/maze_v2/sketch_valid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20


python collab_sketch_with_label.py --llm gpt --model gpt-5 --reasoning-effort low --mixed-dir datasets/ball_path --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --two-turn





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



# Open Router

Providers for Qwen2.5-VL 7B Instruct
Hyperbolic is the only provider for this model




python collab_sketch_with_label.py --llm qwen3 --model qwen/qwen-2.5-vl-7b-instruct --mixed-dir datasets/vpct_ball_drop_vqa --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid

python collab_sketch_with_label.py --llm qwen3 --model qwen/qwen-2.5-vl-7b-instruct --mixed-dir datasets/vpct_ball_drop_sketch --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20





python collab_sketch_with_label.py --llm qwen3 --model qwen/qwen-2.5-vl-7b-instruct --mixed-dir datasets/maze_v2/sketch_invalid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

python collab_sketch_with_label.py --llm qwen3 --model qwen/qwen-2.5-vl-7b-instruct --mixed-dir datasets/maze_v2/sketch_valid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

python collab_sketch_with_label.py --llm qwen3 --model qwen/qwen-2.5-vl-7b-instruct --mixed-dir datasets/maze_v2/invalid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid

python collab_sketch_with_label.py --llm qwen3 --model qwen/qwen-2.5-vl-7b-instruct --mixed-dir datasets/maze_v2/valid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid


python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-image-preview --mixed-dir datasets/ball_path --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid --only "73,82,90"


python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-image-preview --mixed-dir datasets/second_batch_ball_path --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid --only "3,34,36,54,62,65,76,84,94"


python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-preview --mixed-dir datasets/ball_path --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-preview --mixed-dir datasets/second_batch_ball_path --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-preview --mixed-dir datasets/ball_number --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid


python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-preview --mixed-dir datasets/second_batch_ball_number --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid


  python3 /Users/log/Github/sketchvlm/report_vpct_ball_drop_compare.py \
    --gt-root /Users/log/Github/sketchvlm/datasets/vpct-1 \
    --raw-dir /Users/log/Github/sketchvlm/results/mix_eval/vpct/vpct_qwen25vl_7b_sketch \
    --grid-dir /Users/log/Github/sketchvlm/results/mix_eval/vpct_qwen25vl_7b_sketch \
    --out /Users/log/Github/sketchvlm/analysis/vpct/report_sketch_qwen25vl_7b.html \
    --raw-fallback-last-number

```
python consistency/process_with_openrouter.py \
--input consistency/image_questions_thinkmorph.json \
--output consistency/consistency_results_thinkmorph.json 


python consistency/process_with_openrouter.py \
--input consistency/image_questions_vilasr.json \
--output consistency/consistency_results_vilasr.json 

python consistency/process_with_openrouter.py \
--input consistency/source_data/image_questions_geminipro3.json \
--output consistency/judge_output/consistency_results_geminipro3.json 



python consistency/process_with_openrouter.py \
--input consistency/source_data/gemini3_image_two_turn_batch2.json \
--output consistency/judge_output/gemini3_image_two_turn_batch2.json 

python consistency/process_with_openrouter.py \
--input consistency/source_data/gemini3_image_two_turn_batch1.json \
--output consistency/judge_output/gemini3_image_two_turn_batch1.json 



python consistency/process_with_openrouter.py \
--input consistency/source_data/gemini3_ball_paths_batch2.json \
--output consistency/judge_output/gemini3_ball_paths_batch2.json 

python consistency/process_with_openrouter.py \
--input consistency/source_data/gemini3_ball_paths_batch1.json \
--output consistency/judge_output/gemini3_ball_paths_batch1.json 

python consistency/calculate_consistency.py --judge-dir consistency/judge_output 





python consistency/process_with_openrouter.py \
--input consistency/source_data/vpct_thinkmorph.json \
--output consistency/judge_output/vpct_thinkmorph.json 


python consistency/process_with_openrouter.py \
--input consistency/source_data/vpct_vilasr.json \
--output consistency/judge_output/vpct_vilasr.json 

python consistency/process_with_openrouter.py \
--input consistency/source_data/vpct_geminipro3.json \
--output consistency/judge_output/vpct_geminipro3.json 



python /Users/log/Github/sketchvlm/consistency/process_with_openrouter.py \
  --input /Users/log/Github/sketchvlm/consistency/normal_gemini3/ball_batch2_gemini3_image_pro_response.json \
  --output /Users/log/Github/sketchvlm/consistency/normal_gemini3/output_ball_batch2_gemini3_image_pro_response.json \
  --model google/gemini-3-pro-preview




python /Users/log/Github/sketchvlm/consistency/process_with_openrouter.py \
     --input /Users/log/Github/sketchvlm/consistency/source_data/ball_quality/ball_batch1_nano_banana.json \
     --output /Users/log/Github/sketchvlm/consistency/judge_output/ball_quality/ball_batch1_nano_banana.json \
     --model google/gemini-3-flash-preview


python consistency/process_with_openrouter.py \
    --input consistency/source_data/normal_gemini3/source_ball_batch2_gemini3_image_pro_response.json \
     --output /Users/log/Github/sketchvlm/consistency/batch2_g3_pro_nb_results.json \
    --model google/gemini-3-pro-preview
o



  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gemini_25_flash.json \
      --output consistency/judge_output/ball_quality/batch2_gemini_25_flash.json \
      --model google/gemini-3-flash-preview

  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gemini3_pro.json \
      --output consistency/judge_output/ball_quality/batch2_gemini3_pro.json \
      --model google/gemini-3-flash-preview


  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gpt5_low.json \
      --output consistency/judge_output/ball_quality/batch2_gpt5_low.json \
      --model google/gemini-3-flash-preview


      

  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gemini_25_flash.json \
      --output consistency/judge_output/ball_quality/batch2_gemini_25_flash.json \
      --model google/gemini-3-flash-preview

  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gemini3_pro.json \
      --output consistency/judge_output/ball_quality/batch2_gemini3_pro.json \
      --model google/gemini-3-flash-preview


  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gpt5_low.json \
      --output consistency/judge_output/ball_quality/batch2_gpt5_low.json \
      --model google/gemini-3-flash-preview









  # Gemini 2.5 Flash
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gemini_25_flash.json \
      --output consistency/judge_output/ball_quality/batch2_gemini_25_flash.json \
      --model google/gemini-3-flash-preview

  # Gemini 2.5 Pro
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gemini_25_pro.json \
      --output consistency/judge_output/ball_quality/batch2_gemini_25_pro.json \
      --model google/gemini-3-flash-preview

  # Gemini 3 Flash
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gemini3_flash.json \
      --output consistency/judge_output/ball_quality/batch2_gemini3_pro.json \
      --model google/gemini-3-flash-preview

  # GPT-5 Low
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gpt5_low.json \
      --output consistency/judge_output/ball_quality/batch2_gpt5_low.json \
      --model google/gemini-3-flash-preview

  # GPT-5 Med
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gpt5_med.json \
      --output consistency/judge_output/ball_quality/batch2_gpt5_med.json \
      --model google/gemini-3-flash-preview

  # Qwen2.5 7B
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_qwen25_7b.json \
      --output consistency/judge_output/ball_quality/batch2_qwen25_7b.json \
      --model google/gemini-3-flash-preview

  # ThinkMorph
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_thinkmorph.json \
      --output consistency/judge_output/ball_quality/batch2_thinkmorph.json \
      --model google/gemini-3-flash-preview

  # ViLaSR
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_vilasr.json \
      --output consistency/judge_output/ball_quality/batch2_vilasr.json \
      --model google/gemini-3-flash-preview












 python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gemini_25_flash.json \
      --output consistency/judge_output/ball_quality/batch1_gemini_25_flash.json \
      --model google/gemini-3-flash-preview

  # Gemini 2.5 Pro
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gemini_25_pro.json \
      --output consistency/judge_output/ball_quality/batch2_gemini_25_pro.json \
      --model google/gemini-3-flash-preview

  # Gemini 3 Flash
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gemini3_pro.json \
      --output consistency/judge_output/ball_quality/batch1_gemini3_pro.json \
      --model google/gemini-3-flash-preview

  # GPT-5 Low
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gpt5_low.json \
      --output consistency/judge_output/ball_quality/batch1_gpt5_low.json \
      --model google/gemini-3-flash-preview

  # GPT-5 Med
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gpt5_med.json \
      --output consistency/judge_output/ball_quality/batch1_gpt5_med.json \
      --model google/gemini-3-flash-preview

  # Qwen2.5 7B
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_qwen25_7b.json \
      --output consistency/judge_output/ball_quality/batch1_qwen25_7b.json \
      --model google/gemini-3-flash-preview

  # Qwen3 235B Thinking
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_qwen3_235b_thinking.json \
      --output consistency/judge_output/ball_quality/batch1_qwen3_235b_thinking.json \
      --model google/gemini-3-flash-preview

  # Qwen3 8B Thinking
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_qwen3_8b_thinking.json \
      --output consistency/judge_output/ball_quality/batch1_qwen3_8b_thinking.json \
      --model google/gemini-3-flash-preview

  # ThinkMorph
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_thinkmorph.json \
      --output consistency/judge_output/ball_quality/batch2_thinkmorph.json \
      --model google/gemini-3-flash-preview

  # ViLaSR
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_vilasr.json \
      --output consistency/judge_output/ball_quality/batch2_vilasr.json \
      --model google/gemini-3-flash-preview


python /Users/log/Github/sketchvlm/consistency/process_with_openrouter.py \
     --input /Users/log/Github/sketchvlm/consistency/source_data/ball_quality/ball_batch1_nano_banana.json \
     --output /Users/log/Github/sketchvlm/consistency/judge_output/ball_quality/ball_batch1_nano_banana.json \
     --model google/gemini-3-flash-preview




# Maze Quality Evaluation Commands

# Gemini 2.5 Flash Invalid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_gemini25_flash_invalid.json \
    --output consistency/judge_output/grid_world_quality/gemini25_flash_invalid.json \
    --model google/gemini-3-flash-preview

# Gemini 2.5 Flash Valid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_gemini25_flash_valid.json \
    --output consistency/judge_output/grid_world_quality/gemini25_flash_valid.json \
    --model google/gemini-3-flash-preview

# Gemini 2.5 Pro Invalid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_gemini25_pro_invalid.json \
    --output consistency/judge_output/grid_world_quality/gemini25_pro_invalid.json \
    --model google/gemini-3-flash-preview

# Gemini 2.5 Pro Valid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_gemini25_pro_valid.json \
    --output consistency/judge_output/grid_world_quality/gemini25_pro_valid.json \
    --model google/gemini-3-flash-preview

# Gemini 3 Pro Invalid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_gemini3_pro_invalid.json \
    --output consistency/judge_output/grid_world_quality/gemini3_pro_invalid.json \
    --model google/gemini-3-flash-preview

# Gemini 3 Pro Valid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_gemini3_pro_valid.json \
    --output consistency/judge_output/grid_world_quality/gemini3_pro_valid.json \
    --model google/gemini-3-flash-preview

# GPT-5 Low Invalid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_gpt5_low_invalid.json \
    --output consistency/judge_output/grid_world_quality/gpt5_low_invalid.json \
    --model google/gemini-3-flash-preview

# GPT-5 Low Valid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_gpt5_low_valid.json \
    --output consistency/judge_output/grid_world_quality/gpt5_low_valid.json \
    --model google/gemini-3-flash-preview

# Qwen 2.5 7B Invalid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_qwen25_7b_invalid.json \
    --output consistency/judge_output/grid_world_quality/qwen25_7b_invalid.json \
    --model google/gemini-3-flash-preview

# Qwen 2.5 7B Valid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_qwen25_7b_valid.json \
    --output consistency/judge_output/grid_world_quality/qwen25_7b_valid.json \
    --model google/gemini-3-flash-preview

# ThinkMorph Invalid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_thinkmorph_invalid.json \
    --output consistency/judge_output/grid_world_quality/thinkmorph_invalid.json \
    --model google/gemini-3-flash-preview

# ThinkMorph Valid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_thinkmorph_valid.json \
    --output consistency/judge_output/grid_world_quality/thinkmorph_valid.json \
    --model google/gemini-3-flash-preview

# ViLaSR Invalid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_vilasr_invalid.json \
    --output consistency/judge_output/grid_world_quality/vilasr_invalid.json \
    --model google/gemini-3-flash-preview

# ViLaSR Valid
python consistency/process_with_openrouter.py \
    --input consistency/source_data/grid_world_quality/source_maze_quality_vilasr_valid.json \
    --output consistency/judge_output/grid_world_quality/vilasr_valid.json \
    --model google/gemini-3-flash-preview



python collab_sketch_with_label.py --llm gpt --model gpt-5 --reasoning-effort low --mixed-dir datasets/maze_v2/sketch_valid_flattened --no-grid --res-x 1000 --res-y 1000 --prompt-origin top_left --only "1,8,30,44,52,53,60,62,71,74,75,79,82,106,111,114,119,125,127,130,135,138,147,151,154,165,170,174,179,184"

python collab_sketch_with_label.py --llm gemini --model gpt-5 --reasoning-effort low --mixed-dir datasets/maze_v2/sketch_invalid_flattened --no-grid --res-x 1000 --res-y 1000 --prompt-origin top_left


python collab_sketch_with_label.py --llm qwen3 --model qwen/qwen-2.5-vl-7b-instruct --mixed-dir datasets/maze_v2/invalid_flattened --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid

python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-preview --mixed-dir datasets/ball_path --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20

python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-image-preview --mixed-dir datasets/vpct_ball_drop_sketch --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid --only "8,37,39,42,47,48,52,65,73,80,86,88,97,98"

python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-image-preview --mixed-dir datasets/maze_v2/sketch_invalid_flattened_nanob --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid 

python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-image-preview --mixed-dir datasets/maze_v2/sketch_valid_flattened_nanob --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid 


python collab_sketch_with_label.py --llm openrouter --model google/gemini-2.5-pro --mixed-dir datasets/maze_v2/sketch_invalid_flattened --no-grid --res-x 1000 --res-y 1000 --prompt-origin top_left


  python /Users/log/Github/sketchvlm/consistency/process_with_openrouter.py \
      --input /Users/log/Github/sketchvlm/consistency/normal_gemini3/source_data/image_questions_vpct_nanobanana.json \
      --model google/gemini-3-pro-preview


  # For maze valid paths
  python /Users/log/Github/sketchvlm/consistency/process_with_openrouter.py \
      --input /Users/log/Github/sketchvlm/consistency/normal_gemini3/source_data/image_questions_maze_nanob_valid.json \
      --model google/gemini-3-pro-preview

  python /Users/log/Github/sketchvlm/consistency/process_with_openrouter.py \
      --input /Users/log/Github/sketchvlm/consistency/normal_gemini3/source_data/image_questions_maze_nanob_invalid.json \
      --model google/gemini-3-pro-preview






# VPCT

  python consistency/process_with_openrouter.py \
      --input consistency/source_data/vpct_quality/source_ball_quality_vpct_gemini_flash.json \
      --output consistency/judge_output/vpct_quality/vpct_gemini_flash.json \
      --model google/gemini-3-flash-preview

  python consistency/process_with_openrouter.py \
      --input consistency/source_data/vpct_quality/source_ball_quality_vpct_gemini_pro25.json \
      --output consistency/judge_output/vpct_quality/vpct_gemini_pro25.json \
      --model google/gemini-3-flash-preview

  python consistency/process_with_openrouter.py \
      --input consistency/source_data/vpct_quality/source_ball_quality_vpct_gpt5low.json \
      --output consistency/judge_output/vpct_quality/vpct_gpt5low.json \
      --model google/gemini-3-flash-preview


  python consistency/process_with_openrouter.py \
      --input consistency/source_data/vpct_quality/source_ball_quality_vpct_gpt5med.json \
      --output consistency/judge_output/vpct_quality/vpct_gpt5med.json \
      --model google/gemini-3-flash-preview

  python consistency/process_with_openrouter.py \
      --input consistency/source_data/vpct_quality/source_ball_quality_vpct_nanobanana.json \
      --output consistency/judge_output/vpct_quality/vpct_nanobanana.json \
      --model google/gemini-3-flash-preview

  python consistency/process_with_openrouter.py \
      --input consistency/source_data/vpct_quality/source_ball_quality_vpct_thinkmorph.json \
      --output consistency/judge_output/vpct_quality/vpct_thinkmorph.json \
      --model google/gemini-3-flash-preview

  python consistency/process_with_openrouter.py \
      --input consistency/source_data/vpct_quality/source_ball_quality_vpct_vilasr.json \
      --output consistency/judge_output/vpct_quality/vpct_vilasr.json \
      --model google/gemini-3-flash-preview






# ball path
⏺ # Batch 1 - Gemini 3 Pro
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gemini3_pro.json \
      --output consistency/judge_output/ball_quality/batch1_gemini3_pro.json \
      --model google/gemini-3-flash-preview

  # Batch 1 - Gemini 2.5 Flash
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gemini_25_flash.json \
      --output consistency/judge_output/ball_quality/batch1_gemini_25_flash.json \
      --model google/gemini-3-flash-preview

  # Batch 1 - Gemini 2.5 Pro
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gemini_25_pro.json \
      --output consistency/judge_output/ball_quality/batch1_gemini_25_pro.json \
      --model google/gemini-3-flash-preview

  # Batch 1 - GPT-5 Low
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gpt5_low.json \
      --output consistency/judge_output/ball_quality/batch1_gpt5_low.json \
      --model google/gemini-3-flash-preview

  # Batch 1 - GPT-5 Med
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_gpt5_med.json \
      --output consistency/judge_output/ball_quality/batch1_gpt5_med.json \
      --model google/gemini-3-flash-preview

  # Batch 1 - Nano Banana
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_nano_banana.json \
      --output consistency/judge_output/ball_quality/batch1_nano_banana.json \
      --model google/gemini-3-flash-preview

  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_nano_banana.json \
      --output consistency/judge_output/ball_quality/batch2_nano_banana.json \
      --model google/gemini-3-flash-preview

  # Batch 1 - Qwen 2.5 7B
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_qwen25_7b.json \
      --output consistency/judge_output/ball_quality/batch1_qwen25_7b.json \
      --model google/gemini-3-flash-preview

  # Batch 1 - Qwen 3 235B Thinking
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_qwen3_235b_thinking.json \
      --output consistency/judge_output/ball_quality/batch1_qwen3_235b_thinking.json \
      --model google/gemini-3-flash-preview

  # Batch 1 - Qwen 3 8B Thinking
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_qwen3_8b_thinking.json \
      --output consistency/judge_output/ball_quality/batch1_qwen3_8b_thinking.json \
      --model google/gemini-3-flash-preview

  # Batch 1 - ThinkMorph
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_thinkmorph.json \
      --output consistency/judge_output/ball_quality/batch1_thinkmorph.json \
      --model google/gemini-3-flash-preview

  # Batch 1 - ViLaSR
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch1_vilasr.json \
      --output consistency/judge_output/ball_quality/batch1_vilasr.json \
      --model google/gemini-3-flash-preview

  # Batch 2 - Gemini 3 Pro
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gemini3_pro.json \
      --output consistency/judge_output/ball_quality/batch2_gemini3_pro.json \
      --model google/gemini-3-flash-preview

  # Batch 2 - Gemini 2.5 Flash
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gemini_25_flash.json \
      --output consistency/judge_output/ball_quality/batch2_gemini_25_flash.json \
      --model google/gemini-3-flash-preview

  # Batch 2 - Gemini 2.5 Pro
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gemini_25_pro.json \
      --output consistency/judge_output/ball_quality/batch2_gemini_25_pro.json \
      --model google/gemini-3-flash-preview

  # Batch 2 - GPT-5 Low
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gpt5_low.json \
      --output consistency/judge_output/ball_quality/batch2_gpt5_low.json \
      --model google/gemini-3-flash-preview

  # Batch 2 - GPT-5 Med
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_gpt5_med.json \
      --output consistency/judge_output/ball_quality/batch2_gpt5_med.json \
      --model google/gemini-3-flash-preview

  # Batch 2 - Qwen 2.5 7B
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_qwen25_7b.json \
      --output consistency/judge_output/ball_quality/batch2_qwen25_7b.json \
      --model google/gemini-3-flash-preview

  # Batch 2 - ThinkMorph
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_thinkmorph.json \
      --output consistency/judge_output/ball_quality/batch2_thinkmorph.json \
      --model google/gemini-3-flash-preview

  # Batch 2 - ViLaSR
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/ball_quality/source_ball_quality_batch2_vilasr.json \
      --output consistency/judge_output/ball_quality/batch2_vilasr.json \
      --model google/gemini-3-flash-preview




  # Gemini 3 Pro - Valid Paths (0-1000)
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/grid_world_quality/source_maze_quality_gemini3pro_validpaths_0_1000.json \
      --output consistency/judge_output/grid_world_quality/gemini3pro_validpaths_0_1000.json \
      --model google/gemini-3-flash-preview

  # Gemini 3 Pro - Invalid Paths (0-1000)
  python consistency/process_with_openrouter.py \
      --input consistency/source_data/grid_world_quality/source_maze_quality_gemini3pro_invalidpaths_0_1000.json \
      --output consistency/judge_output/grid_world_quality/gemini3pro_invalidpaths_0_1000.json \
      --model google/gemini-3-flash-preview






  python consistency/process_with_openrouter.py \
    --input consistency/normal_gemini3/source_data/image_questions_maze_nanob_invalid.json \
    --output consistency/normal_gemini3/judge_output/maze_nanob_gemini3pro_invalid.json \
    --model google/gemini-3-pro-preview

  python consistency/process_with_openrouter.py \
    --input consistency/normal_gemini3/source_data/image_questions_maze_nanob_valid.json \
    --output consistency/normal_gemini3/judge_output/maze_nanob_gemini3pro_valid.json \
    --model google/gemini-3-pro-preview


python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-image-preview --mixed-dir datasets/maze_v2/sketch_valid_flattened_nanob --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid --only "57,71,80"


python collab_sketch_with_label.py --llm openrouter --model google/gemini-3-pro-image-preview --mixed-dir datasets/maze_v2/sketch_invalid_flattened_nanob --max-tokens 20000 --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --no-system-prompt --no-grid --only "37,41,57,60,91,100,106"





                                                                                                                                                                              
python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/gpt5_low_invalid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_gpt5_low_invalid.json --model google/gemini-3-flash-preview       

python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/gpt5_low_invalid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_gpt5_low_invalid.json --model google/gemini-3-flash-preview       


  python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/gemini3pro_invalid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_gemini3pro_invalid.json --model google/gemini-3-flash-preview           

python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/gemini3pro_valid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_gemini3pro_valid.json --model google/gemini-3-flash-preview                         


  python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/gemini25_flash_invalid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_gemini25_flash_invalid.json --model google/gemini-3-flash-preview  

  python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/gemini25_flash_valid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_gemini25_flash_valid.json --model google/gemini-3-flash-preview     


python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/gemini25_pro_invalid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_gemini25_pro_invalid.json --model google/gemini-3-flash-preview    

python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/gemini25_pro_valid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_gemini25_pro_valid.json --model google/gemini-3-flash-preview   

  python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/thinkmorph_invalid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_thinkmorph_invalid.json --model google/gemini-3-flash-preview       
  
  python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/thinkmorph_valid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_thinkmorph_valid.json --model google/gemini-3-flash-preview       


python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/vilasr_invalid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_vilasr_invalid.json --model google/gemini-3-flash-preview        

python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/vilasr_valid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_vilasr_valid.json --model google/gemini-3-flash-preview        


python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/nanob_gemini3_pro_invalid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_nanob_gemini3_pro_invalid.json --model google/gemini-3-flash-preview              
        
python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/nanob_gemini3_pro_valid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_nanob_gemini3_pro_valid.json --model google/gemini-3-flash-preview                



python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/nanob_gemini3_pro_invalid_retry.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_nanob_gemini3_pro_invalid_retry.json --model google/gemini-3-flash-preview              
        
python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/nanob_gemini3_pro_valid_retry.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_nanob_gemini3_pro_valid_retry.json --model google/gemini-3-flash-preview                


python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/nanob_gemini3_pro_invalid_missing.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/nanob_gemini3_pro_invalid_missing_results.json --model google/gemini-3-pro-preview      


python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/nanob_gemini3_pro_valid_missing.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/nanob_gemini3_pro_valid_missing_results.json --model google/gemini-3-pro-preview            



python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/nanob_gemini3_pro_invalid_newly_available.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/nanob_gemini3_pro_invalid_newly_available_results.json --model google/gemini-3-pro-preview     
                             
python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/nanob_gemini3_pro_valid_newly_available.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/nanob_gemini3_pro_valid_newly_available_results.json --model google/gemini-3-pro-preview       
       

python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/nanob_gemini3_pro_invalid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_nanob_gemini3_pro_invalid.json --model google/gemini-3-flash-preview                
               
python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/grid_world/nanob_gemini3_pro_valid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_consistency/consistency_results_nanob_gemini3_pro_valid.json --model google/gemini-3-flash-preview                  






python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir datasets/maze_v2/sketch_valid_flattened --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir datasets/maze_v2/sketch_invalid_flattened --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir datasets/maze_v2/sketch_invalid_flattened --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 


export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/cairolib:$DYLD_FALLBACK_LIBRARY_PATH && python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir datasets/maze_v2/sketch_valid_flattened --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20



python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/thinkmorph_ball_paths_batch1.json --output /Users/log/Github/sketchvlm/consistency/judge_output/gemini3_flash_judge/consistency_results_thinkmorph_ball_paths_batch1.json --model google/gemini-3-flash-preview    

python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/thinkmorph_ball_paths_batch2.json --output /Users/log/Github/sketchvlm/consistency/judge_output/gemini3_flash_judge/consistency_results_thinkmorph_ball_paths_batch2.json --model google/gemini-3-flash-preview    



python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/vilasr_ball_paths_batch1.json --output /Users/log/Github/sketchvlm/consistency/judge_output/gemini3_flash_judge/consistency_results_vilasr_ball_paths_batch1.json --model google/gemini-3-flash-preview

python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/vilasr_ball_paths_batch1.json --output /Users/log/Github/sketchvlm/consistency/judge_output/gemini3_flash_judge/consistency_results_vilasr_ball_paths_batch1.json --model google/gemini-3-flash-preview
     

python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/vilasr_ball_paths_batch1.json --output /Users/log/Github/sketchvlm/consistency/judge_output/gemini3_flash_judge/consistency_results_vilasr_ball_paths_batch1.json --model google/gemini-3-flash-preview    

python consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/consistency/vilasr_ball_paths_batch2.json --output /Users/log/Github/sketchvlm/consistency/judge_output/gemini3_flash_judge/consistency_results_vilasr_ball_paths_batch2.json --model google/gemini-3-flash-preview    



python /Users/log/Github/sketchvlm/consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/grid_world_quality/nano_banana_invalid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_quality/consistency_results_nano_banana_invalid.json --model google/gemini-3-flash-preview   

python /Users/log/Github/sketchvlm/consistency/process_with_openrouter.py --input /Users/log/Github/sketchvlm/consistency/source_data/grid_world_quality/nano_banana_valid.json --output /Users/log/Github/sketchvlm/consistency/judge_output/grid_world_quality/consistency_results_nano_banana_valid.json --model google/gemini-3-flash-preview   



python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir datasets/maze_v2/valid_flattened --max-tokens 20000 --reasoning-effort medium --no-system-prompt --no-grid 

python collab_sketch_with_label.py --llm gpt --model gpt-5 --mixed-dir datasets/maze_v2/invalid_flattened --max-tokens 20000 --reasoning-effort medium --no-system-prompt --no-grid 


python collab_sketch_with_label.py --llm openrouter --model openai/gpt-5 --mixed-dir datasets/maze_v2/sketch_invalid_flattened --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --only "119,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199"


python collab_sketch_with_label.py --llm openrouter --model openai/gpt-5 --mixed-dir datasets/maze_v2/sketch_valid_flattened --max-tokens 20000 --reasoning-effort medium --adaptive-grid --target-cols 50 --target-rows 50 --min-cell-px 20 --only "48,123,125,126,127,128,129,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199"

python collab_sketch_with_label.py --llm openrouter --model openai/gpt-5 --mixed-dir datasets/maze_v2/valid_flattened --max-tokens 20000 --reasoning-effort medium 

python collab_sketch_with_label.py --llm openrouter --model openai/gpt-5 --mixed-dir datasets/maze_v2/invalid_flattened --max-tokens 20000 --reasoning-effort medium 





python consistency/process_with_openrouter.py --input consistency/normal_gemini3/source_data/image_questions_maze_nanob_valid.json --output consistency/normal_gemini3/judge_output/maze_nanob_gemini3pro_valid.json --model google/gemini-3-pro-preview



python consistency/process_with_openrouter.py --input consistency/normal_gemini3/source_data/image_questions_maze_nanob_invalid.json --output consistency/normal_gemini3/judge_output/maze_nanob_gemini3pro_invalid.json --model google/gemini-3-pro-preview


python consistency/process_with_openrouter.py --input consistency/normal_gemini3/source_data/image_questions_maze_nanob_invalid_retry.json --output consistency/normal_gemini3/judge_output/maze_nanob_gemini3pro_invalid_retry.json --model google/gemini-3-pro-preview

python consistency/process_with_openrouter.py --input consistency/normal_gemini3/source_data/image_questions_maze_nanob_valid_retry.json --output consistency/normal_gemini3/judge_output/maze_nanob_gemini3pro_valid_retry.json --model google/gemini-3-pro-preview


python3 consistency/process_with_openrouter.py --input consistency/source_data/consistency/grid_world/consistency_check_nanob_gemini3_invalid.json --output consistency/judge_output/grid_world_consistency/consistency_results_nanob_gemini3_invalid.json --model google/gemini-3-flash-preview   

python3 consistency/process_with_openrouter.py --input consistency/source_data/consistency/grid_world/consistency_check_nanob_gemini3_valid.json --output consistency/judge_output/grid_world_consistency/consistency_results_nanob_gemini3_valid.json --model google/gemini-3-flash-preview   







file:///Users/log/Github/sketchvlm/consistency/html_output/vpct_quality_scores.html
vpct_gemini3pro_0_1000
vpct_gpt5low
vpct_thinkmorph
vpct_nanobanana
vpct_gemini_flash
vpct_vilasr

/Users/log/Github/sketchvlm/consistency/compare_figure_pdf.py
^^ want to be in same style as this


/Users/log/Github/sketchvlm/consistency
Please make a different visualization for vpct_quality that works very similarly to compare_figure_pdf. I want it in latex. Don't include rows where there's any sort of errors. 