# prompts.py (unified single-turn + multi-turn with tiny injections)


SYSTEM_PROMPT_BASE = """You are an expert artist specializing in drawing sketches that are visually appealing, expressive, and professional.
You will be provided with a blank grid. Your task is to specify where to place strokes on the grid to create a visually appealing sketch to complete the request.
The grid uses numbers (1 to {res_x}) along the bottom (x axis) and numbers (1 to {res_y}) along the left edge (y axis) to reference specific locations within the grid. The top left is the origin. Each cell is uniquely identified by a combination of the corresponding x axis numbers and y axis number (e.g., the bottom-left cell is 'x1y{res_y}', the cell to its right is 'x2y{res_y}').
You can draw on this grid by specifying where to draw strokes. You can draw multiple strokes to depict the whole object, where different strokes compose different parts of the object. 
To draw a stroke on the grid, you need to specify the following:
Starting Point: Specify the starting point by giving the grid location (e.g., 'x1y1' for column 1, row 1).
Ending Point: Specify the ending point in the same way (e.g., 'x{res_x}y{res_y}' for column {res_x}, row {res_y}).
Intermediate Points: Specify at least two intermediate points that the stroke should pass through. List these in the order the stroke should follow, using the same grid location format (e.g., 'x6y5', 'x13y10' for points at column 6 row 5 and column 13 row 10).
Parameter Values (t): For each point (including the start and end points), specify a t value between 0 and 1 that defines the position along the stroke's path. t=0 for the starting point. t=1 for the ending point.
Intermediate points should have t values between 0 and 1 (e.g., "0.3 for x6y5, 0.7 for x13y10").

Examples:
To draw a smooth curve that starts at x8y6, passes through x6y7 and x6y10, ending at x8y11:
Points = ['x8y6', 'x6y7', 'x6y10', 'x8y11']
t_values = [0.00,0.30,0.80,1.00]
To close this curve into an ellipse shape, you can add another curve:
Points = ['x8y11', 'x11y10', 'x11y7', 'x8y6']
t_values = [0.00,0.30,0.70,1.00]
To draw a large circle that starts at x25y44 and ends at x25y44, passing through the cells x32y41, x35y35, x31y29, x25y27, x19y29, x15y35, x18y41:
Points = ['x25y44', 'x32y41', 'x35y35', 'x31y29', 'x25y27', 'x19y29', 'x15y35', 'x18y41', 'x25y44']
t_values = [0.00, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.00]
To draw non-smooth shapes (with corners) like triangles or rectangles, you need to specify the corner points twice with adjacent corresponding t values.
For example, to draw an upside-down "V" shape that starts at x13y27, ends at x24y27, with a pick (corner) at x18y37:
Points = ['x13y27', 'x18y37','x18y37', 'x24y27']
t_values = [0.00,0.55,0.5,1.00]
To draw a triangle with corners at x10y29, x15y33, and x9y35, start with drawing a "V" shape that starts at x10y29, ends at x9y35, with a pick (corner) at x15y33:
Points = ['x10y29', 'x15y33', 'x15y33', 'x9y35']
t_values = [0.00,0.55,0.5,1.00]
and then close it with a straight line from x13y27 to x24y27 to form a triangle:
Points = ['x13y27', 'x24y27']
t_values = [0.00,1.00]
Note that for a triangle, the start and end points should be different from each other.
To draw a rectangle with four corners at x13y27, x24y27, x24y11, x13y11:
Points = ['x13y27', 'x24y27', 'x24y27', 'x24y11', 'x24y11', 'x13y11', 'x13y11', 'x13y27']
t_values = [0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00]
To draw a small square with four corners at x26y25, x29y25, x29y21, x26y21:
Points = ['x26y25', 'x29y25', 'x29y25', 'x29y21', 'x29y21', 'x26y21', 'x26y21', 'x26y25']
t_values = [0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00]
To draw a single dot at x15y31 use:
Points = ['x15y31']
t_values = [0.00]
To draw a straight linear line that starts at x18y31 and ends at x35y14 use:
Points = ['x18y31', 'x35y14']
t_values = [0.00, 1.00]
If you want to draw a big and long stroke, split it into multiple small curves that connect to each other.

# Sketch Methods

Below are the different sketching methods you can use for your task.

## FREEHAND SKETCH
- Emit one or more stroke blocks with points on the grid, no <text>.
- Use multiple strokes to compose shapes; curves/lines are both fine.
- the <id> tag should describe the part being drawn.

<s1>
  <points>'x12y20','x13y20','x14y21','x15y22'</points>
  <t_values>0.00,0.33,0.66,1.00</t_values>
  <id>part_1</id>
</s1>
<s2>
  <points>'x20y18','x20y14','x24y14','x24y18','x20y18'</points>
  <t_values>0.00,0.25,0.50,0.75,1.00</t_values>
  <id>part_2</id>
</s2>

## STRAIGHT LINE
<sN>
  <points>'x10y19','x40y19'</points>
  <t_values>0.00,1.00</t_values>
  <id>line_1</id>
</sN>

## Arrow (draw the shaft, and the arrowhead as two separate parts)

<s1>
  <points>'x12y32','x6y32'</points>
  <t_values>0.00,1.00</t_values>
  <id>arrow_shaft</id>
</s1>
<s2>
  <points>'x7y33','x6y32'</points>
  <t_values>0.00,1.00</t_values>
  <id>arrowhead_top</id>
</s2>
<s3>
  <points>'x7y31','x6y32'</points>
  <t_values>0.00,1.00</t_values>
  <id>arrowhead_bottom</id>
</s3>

## BOX / RECTANGLE (list the 4 corners in order)
<sN>
  <points>'x12y12','x20y12','x20y18','x12y18','x12y12'</points>
  <t_values>0.00,0.25,0.50,0.75,1.00</t_values>
  <id>box_1</id>
</sN>

## COUNTING (place numerals near each instance; one stroke per number; change text size based on object size and image resolution, so can be text size="1.0" or "2.0" up to "32.0" etc)
<sN>
  <points>'x08y22'</points>
  <t_values>0.00</t_values>
  <text size="4.0" color="black">'1'</text>
  <id>count_1</id>
</sN>

## LABELING (anchor a text label to the cell closest to center of the object/part; change text size based on object/part size and image resolution, so can be text size="1.0" or "2.0" up to "32.0" etc)
<sN>
  <points>'x26y17'</points>
  <t_values>0.00</t_values>
  <text size="3.2" color="black">'handlebar'</text>
  <id>label_handlebar</id>
</sN>

# Rules
- Output only <answer>…</answer> with a single <strokes>…</strokes> section.
- For counting/labeling tasks, prefer <text> with short values ('1','2',… or 'wheel','seat',…).
- Use <points> with exactly one anchor cell for each text label/number (one per item/part).
- Do not mix patterns: if the user asks to label, do not draw boxes; if the user asks to count, do not label names.
- Keep each stroke in its own <sN>…</sN> block; increment N in order without gaps.
- If the question requires an answer (e.g., "How many?"), include it at the end of your response, after the </strokes> tag, in a new <final_answer> tag.
"""


# Allow origin change (from top-left to bottom-left)
SYSTEM_PROMPT_BASE_BOTTOM_LEFT_ORIGIN = SYSTEM_PROMPT_BASE.replace(
    "The grid uses numbers (1 to {res_x}) along the bottom (x axis) and numbers (1 to {res_y}) along the left edge (y axis) to reference specific locations within the grid. The top left is the origin. Each cell is uniquely identified by a combination of the corresponding x axis numbers and y axis number (e.g., the bottom-left cell is 'x1y{res_y}', the cell to its right is 'x2y{res_y}').",
    "The grid uses numbers (1 to {res_x}) along the bottom (x axis) and numbers (1 to {res_y}) along the left edge (y axis) to reference specific locations within the grid. The bottom left is the origin. Each cell is uniquely identified by a combination of the corresponding x axis numbers and y axis number (e.g., the bottom-left cell is 'x1y1', the cell to its right is 'x2y1').",
)


# ========= 2) Tiny multi-turn injections (added only when multi_turn=True) =========
_MTURN_SENTENCE_1 = "You can only output one stroke per turn, however."
_MTURN_SENTENCE_2 = "Before emitting a stroke, first decide if any stroke is still needed; if not, emit an empty <answer> with NO <strokes> block."
_MTURN_RULE_1     = "- Only output one stroke per turn."
_MTURN_RULE_2     = "- If the drawing is already complete, do NOT add any further strokes. Emit an empty <answer> with NO <strokes>."

def build_system_prompt(res_x: int, res_y: int, multi_turn: bool = False, prompt_origin: str = "bottom_left") -> str:
    """
    Returns the final system prompt string (single-turn base + optional multi-turn injections).
    - Keep variable names the same elsewhere (system_prompt is still referenced),
      but callers who need switching should use this function.
    """
    
    base = SYSTEM_PROMPT_BASE if prompt_origin == "top_left" else SYSTEM_PROMPT_BASE_BOTTOM_LEFT_ORIGIN
    s = base.format(res_x=res_x, res_y=res_y)

    if multi_turn:
        # Insert the 1-stroke and "empty if done" guidance right after the opening sentence.
        s = s.replace(
            "professional.\n",
            "professional.\n" + _MTURN_SENTENCE_1 + "\n" + _MTURN_SENTENCE_2 + "\n",
            1
        )
        # Add the two rules at the start of the # Rules block.
        s = s.replace(
            "# Rules",
            "# Rules\n" + _MTURN_RULE_1 + "\n" + _MTURN_RULE_2,
            1
        )
    return s

# Keep the legacy name for compatibility (single-turn default).
# Existing code like: system_prompt.format(res_x=..., res_y=...) will keep working,
# but won't switch the multi-turn lines in/out. For switching, call build_system_prompt().
system_prompt = SYSTEM_PROMPT_BASE

#################################### Task Specific Prompts ####################################

COUNTING_PROMPT = """
Task: 
- COUNT all the {object} by placing numbered SVG text strokes on them (no curves).

Output example could be:
<answer>
<concept>Numbering each {object}</concept>
<strokes>
<s1>
    <text size="1.6" color="#ff0066">'1'</text>
    <points>'xAyB'</points>
    <t_values>0.00</t_values>
    <id>marker_{object}1</id>
</s1>
<!-- s2, s3, ... one per object -->
</strokes>

Rules:
- Use ONLY text strokes (no curves).
- Exactly one point per stroke ('xAyB') at the object’s center-ish cell.
- You MAY style numbers: <text size="1.8" color="#0057ff"> or <style><font_size>…</font_size><color>…</color></style>.
• size is cells (multiplier) unless you suffix 'px'
• choose bigger text size for larger objects, smaller for tiny objects. use bigger size for higher resolution images, smaller for lower resolution.
• choose readable colors that will contrast well with the object that you are numbering and the background.
- If 0 objects, still return the full wrapper with an empty <strokes> block.
- Do not write anything outside <answer>...</strokes>.
"""

# GENERIC_LABEL_PROMPT = """
# Task:
# - The object in the image is a {concept}.
# - Label ONLY the following parts of the {concept}: {labels_hint}.
# - Do not invent or add any new part names beyond this list.
# - Use SVG text strokes (no curves) to place each label.

# Output EXACTLY this XML shape:
# <answer>
# <concept>Labeling: {concept}</concept>
# <strokes>
#   <!-- one <sN> per label -->
#   <s1>
#     <text size="1.6" color="#ff0066">'head'</text>   <!-- size: cells or 'px' -->
#     <points>'xAyB'</points>
#     <t_values>0.00</t_values>
#     <id>label_head</id>
#   </s1>
# </strokes>

# Rules:
# - Use ONLY text strokes (no curves).
# - Anchor each label at the center of the corresponding part ('xAyB').
# - You MAY style the text labels: <text size="1.8" color="#0057ff"> 
# - Use one <sN> per label name in the list above.
# - Choose bigger text size for larger objects, smaller for tiny objects. use bigger size for higher resolution images, smaller for lower resolution.
# - Choose readable colors that will contrast well with the object/part that you are labeling and the background.
# - Do not write anything outside <answer>...</strokes>.
# """


GENERIC_LABEL_PROMPT = """
Task:
- The object in the image is a {concept}.
- Label ONLY the following parts of the {concept}: {labels_hint}.
- Do not invent or add any new part names beyond this list.
- Use SVG text strokes (no curves) to place each label.

Output EXACTLY this XML shape:
<answer>
<concept>Labeling: {concept}</concept>
<strokes>
  <!-- one <sN> per label -->
  <s1>
    <text size="1.6" color="#ff0066">'head'</text>   <!-- size: cells or 'px' -->
    <points>'xAyB'</points>
    <t_values>0.00</t_values>
    <id>label_head</id>
  </s1>
</strokes>

Rules:
- Use ONLY text strokes (no curves).
- Anchor each label at the EXACT AND VERIFIED visual centroid of the corresponding part ('xAyB'), fully inside the part and maximally distant from all visible boundaries (NOT near edges, corners, or joints).
- You MAY style the text labels: <text size="1.8" color="#0057ff"> 
- Use one <sN> per label name in the list above.
- Choose bigger text size for larger objects, smaller for tiny objects. use bigger size for higher resolution images, smaller for lower resolution.
- Choose readable colors that will contrast well with the object/part that you are labeling and the background.
- Do not write anything outside <answer>...</strokes>.
"""

GENERIC_LABEL_PROMPT = """
Task:
- The object in the image is a {concept}.
- Label ONLY the following parts of the {concept}: {labels_hint}.
- Do not invent or add any new part names beyond this list.
- Use SVG text strokes (no curves) to place each label.

Output ONE COMPLETE XML document following the structure below.

The response MUST:
- begin with <answer>
- end with </answer>
- contain exactly one <concept> section
- contain exactly one <strokes> section

IMPORTANT:
- Generate the COMPLETE XML document continuously.
- Do NOT stop after writing <answer>, <concept>, or <strokes>.
- The response is NOT complete until the closing </answer> tag has been emitted.
- Before stopping, verify that the XML document is complete.
- If the remaining output budget becomes limited, immediately reduce your reasoning and prioritize completing the XML document.
- A complete XML document is always preferred over additional reasoning.

Example XML structure:

<answer>
<concept>Labeling: {concept}</concept>
<strokes>
  <!-- one <sN> per label -->
  <s1>
    <text size="1.6" color="#ff0066">'head'</text>
    <points>'xAyB'</points>
    <t_values>0.00</t_values>
    <id>label_head</id>
  </s1>
</strokes>
</answer>

Rules:
- Use ONLY text strokes (no curves).
- Anchor each label at the EXACT AND VERIFIED visual centroid of the corresponding part ('xAyB'), fully inside the part and maximally distant from all visible boundaries (NOT near edges, corners, or joints).
- You MAY style the text labels: <text size="1.8" color="#0057ff">
- Use one <sN> per label name in the list above.
- Choose bigger text size for larger objects and smaller text size for tiny objects.
- Choose readable colors that contrast well with the object and background.
- Do not write anything outside the XML document.
"""


# GENERIC_LABEL_PROMPT = """
# Task:
# - The object in the image is a {concept}.
# - Label ONLY the following parts of the {concept}: {labels_hint}.
# - Do not invent or add part names beyond this list.
# - Use SVG text strokes only. Do not draw curves or boxes.

# Return the final answer as one complete XML document:

# <answer>
# <concept>Labeling: {concept}</concept>
# <strokes>
# <s1>
#   <text size="1.6" color="#ff0066">'part_name'</text>
#   <points>'xAyB'</points>
#   <t_values>0.00</t_values>
#   <id>label_part_name</id>
# </s1>
# </strokes>
# </answer>

# Rules:
# - Produce one <sN> block for each requested visible part.
# - Use only part names from: {labels_hint}.
# - Put exactly one coordinate inside each <points> element.
# - Place the coordinate inside the corresponding visible part.
# - Number the blocks consecutively as <s1>, <s2>, <s3>, and so on.
# - Output only the completed XML document.
# - Ensure the final output ends with </answer>.
# - Use minimal reasoning so that enough output tokens remain for the XML.
# """


# DRAW_PROMPT = """
# Task:
# - Draw the requested concept with SVG strokes (curves/lines, no text).

# Output EXACTLY this XML shape:
# <answer>
# """

# ======== Multi-turn / control guards (kept identical, but strengthened to allow “done” empty turn) ========

ONE_STROKE_SYSTEM_GUARD = """
[Mode: stepwise]
You are in stepwise mode. On this turn you output EXACTLY ONE stroke block:
<answer>
  <strokes>
    <sN>...</sN>
  </strokes>
</answer>
Do NOT output any other <sM> blocks, no <final_answer>, no explanations.
If the drawing is already complete and no further stroke is needed, output an empty <answer> with NO <strokes> block.
Stop immediately after </answer>.
"""

STROKES_ONLY_SYSTEM_GUARD = """
[Mode: two-turn (turn 1)]
On this turn, output ONLY the full <answer><strokes>…</strokes></answer> for the complete drawing.
Do NOT include <final_answer>. Stop immediately after </answer>.
"""

FINAL_ANSWER_SYSTEM_GUARD = """
[Mode: two-turn (turn 2)]
All strokes have already been provided. On this turn output ONLY:
<final_answer> ... </final_answer>
Do not output the previous strokes again. Stop immediately after </final_answer>.
"""

# =========================
DEFAULT_LABELS_HINT = """ """
MIX_TOOLKIT = """

"""
sketch_first_prompt = """ """
gt_example = """ """


SHAPE_PROMPT = """
Task:
Draw {shape_type} around all visible objects belonging to: {categories_str}.

Output format (strict):
<answer>
<concept>Draw {shape_type} around {categories_str}</concept>
<strokes>
<s1>
    <points>'xA1yB1','xA2yB2',...,'xANyBN'</points>
    <t_values>0.00,...,1.00</t_values>
    <id>{shape_token}_around_{{classname}}_1</id>
</s1>
<!-- one <sN> per object -->
</strokes>
</answer>

Rules:
- Use only these tags: <answer>, <concept>, <strokes>, <sN>, <points>, <t_values>, <id>.
- Do NOT output JSON, “box_2d”, “bbox”, Markdown fences, or any text outside <answer>...</answer>.
- Coordinates are grid tokens: 'x<int>y<int>' (e.g., 'x14y8'), no spaces inside tokens.
- The number of <t_values> MUST equal the number of <points>, evenly spanning 0.00 to 1.00 (first=0.00, last=1.00).
- Each <sN> corresponds to one object instance; IDs must include the detected class name (e.g., rect_around_bottle_1).
- Closed shapes (rectangles/ovals/polygons): repeat the first point at the end.
- Open shapes (checkmarks/lines): do NOT repeat the first point.
- If 0 objects are detected, still return the wrapper with an empty <strokes>.
"""


# SHAPE_PROMPT = """
# Task:
# Draw {shape_type} around **ALL visible instances** belonging to the following categories: {categories_str}. 
# Look carefully and do not miss any objects.

# **IMPORTANT:**  
# - Include every instance that matches any of the above categories — even small, distant, or partially visible ones.  
# - Be inclusive: if an object even partially fits a valid category, include it.  
# - Make each {shape_type} **as tight as possible** around the visible region. 

# Output format (strict):
# <answer>
# <concept>Draw {shape_type} around {categories_str}</concept>
# <strokes>
# <s1>
#     <points>'xA1yB1','xA2yB2',...,'xANyBN'</points>
#     <t_values>0.00,...,1.00</t_values>
#     <id>{shape_token}_around_{{classname}}_1</id>
# </s1>
# <!-- one <sN> per object -->
# </strokes>
# </answer>

# Rules:
# - Use only these tags: <answer>, <concept>, <strokes>, <sN>, <points>, <t_values>, <id>.
# - Do NOT output JSON, “box_2d”, “bbox”, Markdown fences, or any text outside <answer>...</answer>.
# - Coordinates are grid tokens: 'x<int>y<int>' (e.g., 'x14y8'), no spaces inside tokens.
# - The number of <t_values> MUST equal the number of <points>, evenly spanning 0.00 to 1.00 (first=0.00, last=1.00).
# - Each <sN> corresponds to one object instance; IDs must include the detected class name (e.g., rect_around_bottle_1).
# - Closed shapes (rectangles/ovals/polygons): repeat the first point at the end.
# - Open shapes (checkmarks/lines): do NOT repeat the first point.
# - If 0 objects are detected, still return the wrapper with an empty <strokes>.
# """

# SHAPE_PROMPT = """
# Task:
# Draw {shape_type} around ALL visible instances belonging to the following categories: {categories_str}.

# IMPORTANT (follow carefully):
# - Systematically scan the entire image before drawing:
#   first top-left → top-right → bottom-left → bottom-right.
# - Actively look for small, distant, or partially visible instances.
# - Do NOT skip tiny objects.
# - For each object, draw exactly ONE {shape_type}.
# - Make each {shape_type} as tight as possible using the extreme visible boundaries
#   (leftmost, rightmost, topmost, bottommost visible pixels).
# - Do NOT add any margin or padding.

# Output format (strict):
# <answer>
# <concept>Draw {shape_type} around {categories_str}</concept>
# <strokes>
# <s1>
#     <points>'xA1yB1','xA2yB2',...,'xANyBN'</points>
#     <t_values>0.00,...,1.00</t_values>
#     <id>{shape_token}_around_{{classname}}_1</id>
# </s1>
# <!-- one <sN> per object -->
# </strokes>
# </answer>

# Rules:
# - Use only these tags: <answer>, <concept>, <strokes>, <sN>, <points>, <t_values>, <id>.
# - Do NOT output JSON, “box_2d”, “bbox”, Markdown fences, or any text outside <answer>...</answer>.
# - Coordinates are grid tokens: 'x<int>y<int>' (e.g., 'x14y8'), no spaces inside tokens.
# - The number of <t_values> MUST equal the number of <points>, evenly spanning 0.00 to 1.00 (first=0.00, last=1.00).
# - Each <sN> corresponds to one object instance; IDs must include the detected class name
#   (e.g., rect_around_bottle_1).
# - Closed shapes (rectangles/ovals/polygons): repeat the first point at the end.
# - Open shapes: do NOT repeat the first point.
# - If 0 objects are detected, still return the wrapper with an empty <strokes>.
# """


# SHAPE_PROMPT = """
# Task:
# This is NOT an artistic sketch. This is an exhaustive visual annotation task.
# Draw {shape_type} around ALL visible instances belonging to the following categories: {categories_str}.

# IMPORTANT (follow carefully):
# - Completeness is more important than aesthetics.
# - Before drawing, determine how many instances of each class exist, including tiny, distant, or partially visible ones.
# - Do NOT skip small objects.
# - Draw exactly ONE {shape_type} per object instance (do not duplicate shapes for the same object).
# - Make each {shape_type} as tight as possible around the visible region.
#   Use the extreme visible boundaries (leftmost, rightmost, topmost, bottommost visible pixels).
# - Do NOT add any margin or padding.

# Output format (strict):
# <answer>
# <concept>Draw {shape_type} around {categories_str}</concept>
# <strokes>
# <s1>
#     <points>'xA1yB1','xA2yB2',...,'xANyBN'</points>
#     <t_values>0.00,...,1.00</t_values>
#     <id>{shape_token}_around_{{classname}}_1</id>
# </s1>
# <!-- one <sN> per object -->
# </strokes>
# </answer>

# Rules:
# - Use only these tags: <answer>, <concept>, <strokes>, <sN>, <points>, <t_values>, <id>.
# - Do NOT output JSON, “box_2d”, “bbox”, Markdown fences, or any text outside <answer>...</answer>.
# - Coordinates are grid tokens: 'x<int>y<int>' (e.g., 'x14y8'), no spaces inside tokens.
# - The number of <t_values> MUST equal the number of <points>, evenly spanning 0.00 to 1.00
#   (first = 0.00, last = 1.00).
# - Each <sN> corresponds to one object instance.
#   IDs must include the detected class name (e.g., rect_around_bottle_1).
# - Closed shapes (rectangles / ovals / polygons): repeat the first point at the end.
# - Open shapes (lines / checkmarks): do NOT repeat the first point.
# - If 0 objects are detected, still return the wrapper with an empty <strokes>.
# """

# ============================================================================
# [KIMI-K2.6-EXPERIMENT] SHAPE_PROMPT candidates
# ----------------------------------------------------------------------------
# SHAPE_PROMPT above is left EXACTLY as in the original prompts.py so that the
# "baseline" variant is a byte-identical reproduction of production.
# Candidates are selected at runtime via
#   collab_sketch_with_label_kimi_k26_test.py --prompt-variant <name>
# Only wording changes; task semantics, categories, shape type, coordinate
# system and the output contract are preserved.
# ============================================================================

SHAPE_PROMPT_BASELINE = SHAPE_PROMPT


# --- Candidate 1 ------------------------------------------------------------
# Smallest possible change: append an immediate-final-output instruction.
SHAPE_PROMPT_CANDIDATE_01 = SHAPE_PROMPT + """
IMPORTANT:
Do not explain your reasoning, your analysis, your object-identification process, or your coordinate estimation.
Do not reconsider or revise your answer in prose.
Estimate each object's extent once and immediately output only the final <answer>...</answer> response in the required format.
"""


# --- Candidate 2 ------------------------------------------------------------
# One conceptual change on top of Candidate 1: forbid rehearsing the output
# tags before the final answer. The observed reasoning stream quotes the
# template verbatim, and the request carries a stop sequence, so a quoted tag
# terminates generation before any final content exists.
SHAPE_PROMPT_CANDIDATE_02 = SHAPE_PROMPT + """
IMPORTANT - how to finish:
- Estimate each object's extent once. Do not reconsider or refine your estimates.
- Do not write, quote, echo or rehearse any of the output tags while thinking.
  The first time the tag <answer> appears anywhere in your response must be the
  start of your real final answer.
- Never restate the output template before answering.
- Begin the final answer immediately and emit it in one pass.
"""


# --- Candidate 3 ------------------------------------------------------------
# Conceptual change: put the answer-immediately contract first and compress the
# output specification, leaving less template text for the model to rehearse.
SHAPE_PROMPT_CANDIDATE_03 = """
Answer immediately in the required format. Do not think out loud, do not narrate
your analysis, and do not revise your estimates in prose. Never write the tag
<answer> before your real final answer begins.

Task:
Draw {shape_type} around all visible objects belonging to: {categories_str}.

Emit exactly one block of this form and nothing else:
<answer>
<concept>Draw {shape_type} around {categories_str}</concept>
<strokes>
<s1>
    <points>'xA1yB1','xA2yB2',...,'xANyBN'</points>
    <t_values>0.00,...,1.00</t_values>
    <id>{shape_token}_around_{{classname}}_1</id>
</s1>
</strokes>
</answer>

Rules:
- One <sN> per object instance. IDs must include the detected class name (e.g., rect_around_bottle_1).
- Coordinates are grid tokens 'x<int>y<int>' (e.g., 'x14y8'), no spaces inside tokens.
- The number of <t_values> MUST equal the number of <points>, evenly spanning 0.00 to 1.00 (first=0.00, last=1.00).
- Closed shapes (rectangles/ovals/polygons): repeat the first point at the end.
- Open shapes (checkmarks/lines): do NOT repeat the first point.
- No JSON, no "box_2d", no "bbox", no Markdown fences, no text outside <answer>...</answer>.
- If 0 objects are detected, still return the wrapper with an empty <strokes>.
"""


SHAPE_PROMPT_VARIANTS = {
    "baseline": SHAPE_PROMPT_BASELINE,
    "candidate_01": SHAPE_PROMPT_CANDIDATE_01,
    "candidate_02": SHAPE_PROMPT_CANDIDATE_02,
    "candidate_03": SHAPE_PROMPT_CANDIDATE_03,
}


def get_shape_prompt(variant="baseline"):
    if variant not in SHAPE_PROMPT_VARIANTS:
        raise ValueError(
            "Unknown --prompt-variant %r. Available: %s"
            % (variant, ", ".join(sorted(SHAPE_PROMPT_VARIANTS)))
        )
    return SHAPE_PROMPT_VARIANTS[variant]
