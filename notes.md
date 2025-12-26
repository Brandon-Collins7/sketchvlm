
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


```

# Overall Instructions
You will be shown two images: the original and an AI-annotated version. Your job is to grade the quality of the sketch using a rubric. You should grade only based on the following attributes that are given.

**Logic consistency**
Do the drawn lines make some sort of logical sense? For example, does drawn ball path clip through any of the static environment or does the ball path follow _extremely_ unrealistic physics? The score should be based on the scale of 1 - 5

1) The sketch makes absolutely no logical sense.
2) The sketch has some critical flaws that breaks the logic of the sketch.
3) The sketch contains multiple logical errors.
4) The sketch contains a minor logical error.
5) The sketch contains zero logical errors.

**Ease of communication**
Would it be reasonable for another person to view the sketch and easily understand what the original drawer was thinking? For example, if image is dominated by sketches to the point where it is hard to understand the intent of the annannotator or hard to view the unannotated image, that would be considered detrimental. 

1) The sketch is extremely difficult to understand and does not communicate the annotator’s intent.
2) The sketch is hard to understand and communicates the intent poorly due to major clarity issues (e.g., clutter, ambiguity, missing cues).
3) The sketch is somewhat understandable but has multiple clarity issues that make the intent unclear.
4) The sketch is mostly easy to understand, with only a minor clarity issue.
5) The sketch is immediately clear and easy to understand.

# Output Format
You should follow this output format EXACTLY with no other output:

<example_format>
{reasoning for logical consistency score}
Logical Consistency Score: {integer from 1 - 5}

{reasoning for ease of communication score}
Ease of Communication Score: {integer from 1 - 5}
</example_format>

# Example Output

<example_1>
The drawing contains multiple errors. The ball path clips through one of the platforms, but it is a minor issue as the path of the ball still makes sense. Additionally, the ball path defies gravity and begins to momentarily float straight upwards for no logical reason. The minor error combined with the critical error results in a logical score of 2/5
Logical Consistency Score: 5

The sketch is easy to understand. Even though the logic is flawed, it is easy to see how the ball path was traced. The sketch is not overly complicated and also has an arrow to indicate direction. The final resting spot of the ball is also clearly marked. It is unlikely that someone would have a hard time understanding the drawing. 
Ease of Communication Score: {integer from 1 - 5}
</example_1>



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