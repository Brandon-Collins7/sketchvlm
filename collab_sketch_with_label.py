# sketch_app.py  (Py 3.7–3.9 compatible)
import argparse
import ast
import base64
import cairosvg
import io
import json
import math
import os
import random
import re
import signal
import socket
import time
import traceback
import uuid
from datetime import datetime
from typing import Optional, List, Dict
import html


import hashlib, io, base64, os, re
from pathlib import Path


from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

import utils
from prompts import (
    sketch_first_prompt, system_prompt, gt_example, COUNTING_PROMPT, MIX_TOOLKIT, GENERIC_LABEL_PROMPT, DEFAULT_LABELS_HINT,
    ONE_STROKE_SYSTEM_GUARD, STROKES_ONLY_SYSTEM_GUARD, FINAL_ANSWER_SYSTEM_GUARD,
    build_system_prompt,
)


from grid_manager import GridManager
from llm_adapters import BaseLLMAdapter, GeminiAdapter, make_adapter
from PIL import Image, ImageOps

# batch eval / datasets
from pathlib import Path
from datasets import load_dataset  # pip install datasets
try:
    from tqdm import tqdm  # optional, pretty progress bar
except Exception:
    tqdm = lambda x, **k: x



# --- axis-aligned rectangle detection (grid -> px) ---

def _parse_grid_points(raw_points: str):
    # e.g. "'x6y6','x63y6','x63y43','x6y43','x6y6'" -> [(6,6),(63,6),(63,43),(6,43),(6,6)]
    return [(int(x), int(y)) for x, y in re.findall(r"x(\d+)y(\d+)", raw_points)]

def _is_closed(points):
    return len(points) >= 2 and points[0] == points[-1]

def _is_axis_aligned_rect(points):
    # 4 unique corners or 5 with explicit closure (first==last), axis-aligned
    if not _is_closed(points):
        return False
    uniq = points[:-1] if len(points) >= 5 else points  # ignore duplicate last if present
    if len(uniq) != 4:
        return False
    xs = sorted({p[0] for p in uniq})
    ys = sorted({p[1] for p in uniq})
    if len(xs) != 2 or len(ys) != 2:
        return False
    return True



# =========================
# Flask App
# =========================
class SketchApp:
    def __init__(
        self, res, cell_size, grid_size, stroke_width, target_concept,
        user_always_first, llm_adapter: BaseLLMAdapter, show_full_grid: bool = False,
        dynamic_grid: bool = True, min_grid: int = 10, max_grid: int = 100,
        adaptive_grid: bool = False,
        target_cols: int = 50,
        target_rows: int = 50,
        min_cell_px: int = 18,
        max_cell_px: int = 64,
        no_grid: bool = False,
        no_system_prompt: bool = False,
        prompt_origin: str = "bottom_left",
        # NEW: allow overriding coordinate-space / canvas resolution.
        # For your no-grid coord trials, set these to 1000/1000.
        res_x: Optional[int] = None,
        res_y: Optional[int] = None,
    ):
        self.app = Flask(__name__)
        self.session_id = str(uuid.uuid4())

        # LLM Setup
        self.seed_mode = "stochastic"
        self.cache = llm_adapter.cache
        self.max_tokens = llm_adapter.max_tokens
        self.llm = llm_adapter
        
        
        # Needed for baseline
        self.no_grid = bool(no_grid)
        self.no_system_prompt = bool(no_system_prompt)


        # Grid setup
        self.grid_manager = GridManager(
        cell_size=cell_size, min_grid=min_grid, max_grid=max_grid,
        adaptive_grid=adaptive_grid, target_cols=target_cols, target_rows=target_rows,
        min_cell_px=min_cell_px, max_cell_px=max_cell_px,
        )
        self.show_full_grid = show_full_grid
        self.multi_stroke = True
        self.dynamic_grid = dynamic_grid
        
        self.cell_size = self.grid_manager.cell_size
        
        #origin is sometimes bottom left, sometimes top left
        self.prompt_origin = (prompt_origin or "bottom_left").strip().lower()
        
        # Backward compatibility properties
        # NOTE: res_x/res_y are used by the system prompt and (in no-grid mode)
        # define the coordinate-space / canvas size.
        self.res_x = int(res_x) if res_x is not None else int(res)
        self.res_y = int(res_y) if res_y is not None else int(res)
        self.res = max(self.res_x, self.res_y)
        self.num_cells = self.res
        self.cell_size = int(cell_size)
        self.grid_size = tuple(grid_size)

        # If we're in no-grid mode and the caller provided res_x/res_y, treat
        # that as the canvas size. This makes the (0..res_x, 0..res_y) coordinate
        # system line up with pixels, with origin at top-left.
        if self.no_grid and (res_x is not None or res_y is not None):
            self.grid_size = (self.res_x, self.res_y)
            # cell_size isn't meaningful without a grid, but some code
            # uses it for defaults; keep it sane.
            self.cell_size = 1
        
        # Initialize default grid
        self.init_canvas_grid = self.grid_manager.grid_image
        self.positions = self.grid_manager.positions
        
        self.init_canvas = Image.new('RGB', self.grid_size, 'white')
        self.init_canvas.save("static/init_canvas.png")

        self.base_canvas = self.init_canvas.copy()
        self.last_canvas_b64: Optional[str] = None
        self._update_canvas_b64("static/init_canvas.png")

        self.stroke_width = stroke_width
        self.num_sampled_points = 100

        # Program init
        self.user_always_first = user_always_first
        self.folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.drawn_concepts = []

        self.target_concept = target_concept
        self.sketch_mode = "solo"
        self.cur_svg_to_render = "None"
        self.initialize_all()
        
        
        self.text_font_family = "Arial"
        self.text_font_scale  = 3.2   # ~3.2 * cell_size (e.g., 12 -> 38.4px). Tweak to taste.
        
        
        #  stroke-color cycling (human-view only) ----
        self.cycle_stroke_colors = False      # set from CLI later
        self.save_colored_steps  = False      # set from CLI later
        self._palette = ["#ff0000","#ff7f00","#ffff00","#00a000","#0000ff","#800080"]  # ROYGBP
        self._colored_svg = self._svg_root_open()  # parallel SVG just for human-colored composites



        # Routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/update-mode', 'set_sketch_mode', self.set_sketch_mode, methods=['POST'])
        self.app.add_url_rule('/send-user-strokes', 'get_user_stroke', self.get_user_stroke, methods=['POST'])
        self.app.add_url_rule('/call-agent', 'call_agent', self.call_agent, methods=['POST'])
        self.app.add_url_rule('/clear-canvas', 'clear_canvas', self.clear_canvas, methods=['POST'])
        self.app.add_url_rule('/submit-sketch', 'submit_sketch', self.submit_sketch, methods=['POST'])
        self.app.add_url_rule('/get-new-concept', 'get_new_concept', self.get_new_concept, methods=['POST'])
        self.app.add_url_rule('/draw-sketch', 'draw_sketch', self.draw_entire_sketch, methods=['POST'])
        self.app.add_url_rule('/shutdown', 'shutdown', self.shutdown, methods=['POST'])
        self.app.add_url_rule('/toggle-grid', 'toggle_grid', self.toggle_grid, methods=['POST'])
        self.app.add_url_rule('/upload-image', 'upload_image', self.upload_image, methods=['POST'])
        self.app.add_url_rule("/set-multi-stroke", "set_multi_stroke", self.set_multi_stroke, methods=["POST"])
        
        self.app.add_url_rule('/skip-turn', 'skip_turn', self.skip_turn, methods=['POST'])
        self.app.add_url_rule('/set-turn-order', 'set_turn_order', self.set_turn_order, methods=['POST'])

    # ---------- small helpers ----------
    
    def _sys_prompt_or_none(self):
        if self.no_system_prompt:
            return None
        # Treat “colab”, stepwise, and two-turn as multi-turn prompt mode.
        multi = bool(getattr(self, "_prompt_multi_turn", False) or self.sketch_mode == "colab")
        return build_system_prompt(
            self.res_x, self.res_y,
            multi_turn=multi,
            prompt_origin=args.prompt_origin,
        )

    
    def skip_turn(self):
        """
        User presses ‘Skip Turn’ during COLAB → let the agent draw now.
        We just reuse the same flow as /call-agent.
        """
        return self.call_agent()

    def set_turn_order(self):
        """
        Toggle who starts in COLAB rounds.
        Payload: {"user_first": true|false}
        """
        data = request.get_json(force=True, silent=True) or {}
        self.user_always_first = bool(data.get("user_first", False))
        # If COLAB just started (fresh init), we respect this flag in init_thinking_tags:
        #   if not self.user_always_first: self.call_agent()
        return jsonify({"status": "ok", "user_first": self.user_always_first})

    def set_multi_stroke(self):
        self.multi_stroke = bool(request.get_json().get("enabled"))
        return jsonify({"status": "ok", "multi": self.multi_stroke})

    def _update_canvas_b64(self, path: str):
        with open(path, "rb") as f:
            self.last_canvas_b64 = base64.b64encode(f.read()).decode()
            
            
    # collab_sketch_with_label.py
    def _svg_root_open(self):
        W, H = self.base_canvas.size
        return (f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                f'xmlns="http://www.w3.org/2000/svg">')

    

    def _composite_svg_on_base(self, svg_text: str, out_png: str):
        over = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg_text.encode()))).convert("RGBA")
        base = getattr(self, "base_canvas_clean", self.base_canvas).convert("RGBA")
        if over.size != base.size:
            inner = re.sub(r'^.*?<svg[^>]*>|</svg>\s*$', '', svg_text, flags=re.S)
            svg_text = (f'<svg width="{base.size[0]}" height="{base.size[1]}" '
                        f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>')
            over = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg_text.encode()))).convert("RGBA")
        Image.alpha_composite(base, over).convert("RGB").save(out_png)
        self._update_canvas_b64(out_png)

    def _composite_svg_on_raw_image(self, svg_text: str, raw_img: Image.Image, out_png: str):
        """Composite stroke SVG onto the *raw* image (no grid).

        When the model drew on a grid image, the SVG coordinates live in the grid-canvas
        space. We translate them back into the raw-image space using the placement offset.
        """
        raw_rgb = ImageOps.exif_transpose(raw_img).convert("RGB")
        ox, oy = getattr(self, "_raw_image_offset", (0, 0))
        rw, rh = raw_rgb.size

        inner = re.sub(r'^.*?<svg[^>]*>|</svg>\s*$', '', svg_text, flags=re.S)
        if ox or oy:
            inner = f'<g transform="translate({-ox},{-oy})">{inner}</g>'

        svg2 = (f'<svg width="{rw}" height="{rh}" viewBox="0 0 {rw} {rh}" '
                f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>')

        over = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg2.encode()))).convert("RGBA")
        base = raw_rgb.convert("RGBA")
        if over.size != base.size:
            over = over.resize(base.size)

        Image.alpha_composite(base, over).convert("RGB").save(out_png)



    def _next_undrawn(self, xml, expected_s_no: Optional[int] = None) -> List[str]:
        # normalise input to a single string and strip any code fences
        if isinstance(xml, list):
            xml_str = "\n".join(xml)
        else:
            xml_str = str(xml)
        xml_str = re.sub(r"^```(?:xml|html)?\s*|\s*```$", "", xml_str.strip())

        # ids already drawn on the visible canvas (source of truth)
        drawn_ids = set()
        for blk in re.findall(r"<s\d+>.*?</s\d+>", self.all_strokes_svg, re.S):
            m = re.search(r"<id>(.*?)</id>", blk, re.S)
            if m:
                drawn_ids.add(m.group(1).strip())

        # candidate blocks in this answer
        blocks = re.findall(r"(<s\d+>.*?</s\d+>)", xml_str, re.S)

        # If caller requested a specific s-number (single-stroke mode), return only that.
        if expected_s_no is not None:
            for blk in blocks:
                m_s = re.search(r"<s(\d+)>", blk)
                if not m_s:
                    continue
                if int(m_s.group(1)) != expected_s_no:
                    continue  # not our turn number
                # still honor id de-dup vs what we've already drawn
                m_id = re.search(r"<id>(.*?)</id>", blk, re.S)
                if m_id and m_id.group(1).strip() in drawn_ids:
                    continue
                return [blk]  # exact match found
            return []  # no exact sN in this answer → draw nothing

        # Otherwise (multi-stroke mode): keep unseen-by-id, dedup repeated s# within this answer
        keep: List[str] = []
        id_used, s_used = set(), set()
        for blk in blocks:
            m_s = re.search(r"<s(\d+)>", blk)
            s_no = m_s.group(1) if m_s else None
            if s_no and s_no in s_used:
                continue

            m_id = re.search(r"<id>(.*?)</id>", blk, re.S)
            if not m_id:
                keep.append(blk)  # permissive if no id
            else:
                _id = m_id.group(1).strip()
                if _id not in id_used and _id not in drawn_ids:
                    keep.append(blk)
                    id_used.add(_id)

            if s_no:
                s_used.add(s_no)

        return keep

    def _recolor_svg_group(self, g_block: str, color: str) -> str:
        """Return a copy of <g ...>...</g> with stroke/fill set to `color` (if present)."""
        if not color:
            return g_block
        # stroke on paths/shapes
        g2 = re.sub(r'stroke="[^"]+"', f'stroke="{color}"', g_block)
        # text fill (keep stroke-none on text)
        g2 = re.sub(r'(<text\b[^>]*\bfill=")[^"]+(")', r'\1'+color+r'\2', g2)
        # if text has no fill, inject one
        g2 = re.sub(r'(<text\b(?![^>]*\bfill=)[^>]*)(>)', r'\1 fill="'+color+r'"\2', g2)
        return g2
    
    
    def _render_svg_to_png_no_b64(self, svg_text: str, out_png: str):
        over = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg_text.encode()))).convert("RGBA")
        base = getattr(self, "base_canvas_clean", self.base_canvas).convert("RGBA")
        if over.size != base.size:
            inner = re.sub(r'^.*?<svg[^>]*>|</svg>\s*$', '', svg_text, flags=re.S)
            svg_text = (f'<svg width="{base.size[0]}" height="{base.size[1]}" '
                        f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>')
            over = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg_text.encode()))).convert("RGBA")
        Image.alpha_composite(base, over).convert("RGB").save(out_png)

    
    # --- model-view control (original vs re-render) ---
    def _encode_pil_to_b64(self, pil_img):
        import io, base64
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _canvas_b64_for_model(self):
        """
        In stepwise mode: return the b64 the provider should see this turn.
        If --stepwise-original-only is on, prefer the raw original; else use last re-render.
        """
        if getattr(self, "stepwise_send_original_only", False):
            # prefer raw original; fall back to original-with-grid; then to last
            return getattr(self, "orig_raw_b64", None) or getattr(self, "orig_canvas_b64", None) or self.last_canvas_b64
        return self.last_canvas_b64




    # ---------- routes ----------
    def toggle_grid(self):
        self.show_full_grid = not self.show_full_grid
        
        # Update grid manager
        self.grid_manager.grid_image, self.grid_manager.positions = utils.create_grid_image(
            res_x=self.res_x, res_y=self.res_y, cell_size=self.cell_size, header_size=self.cell_size, full=self.show_full_grid
        )
        
        # Update backward compatibility properties
        self.init_canvas_grid = self.grid_manager.grid_image
        self.positions = self.grid_manager.positions
        self.base_canvas = self.base_canvas.convert("RGBA")
        grid = self.init_canvas_grid.convert("RGBA")
        mask = grid.convert("L").point(lambda p: 255 if p < 200 else 0)
        self.base_canvas.paste(grid, (0, 0), mask)
        self.base_canvas.save("static/init_canvas.png")
        self.base_canvas.save("static/cur_canvas_user.png")
        self.base_canvas.save("static/cur_canvas_agent.png")
        self._update_canvas_b64("static/init_canvas.png")
        return jsonify({"status": "success", "full_grid": self.show_full_grid})

    '''
    def upload_image(self):
        if 'image' not in request.files:
            return jsonify({"status": "error", "msg": "no file"}), 400
        file = request.files['image']
        fname = secure_filename(file.filename or "photo.png")

        img = Image.open(file.stream).convert("RGB").resize(self.grid_size, Image.LANCZOS)
        grid = self.init_canvas_grid.convert("RGBA")
        mask = grid.convert("L").point(lambda p: 255 if p < 200 else 0)
        img = img.convert("RGBA")
        img.paste(grid, (0, 0), mask)

        self.base_canvas = img.convert("RGB")
        self.base_canvas.save("static/init_canvas.png")
        self.base_canvas.save("static/cur_canvas_user.png")
        self.base_canvas.save("static/cur_canvas_agent.png")
        self._update_canvas_b64("static/init_canvas.png")

        return jsonify({"status": "success", "filename": fname})
    '''
    
    def upload_image(self):
        if 'image' not in request.files:
            return jsonify({"status": "error", "msg": "no file"}), 400
        file = request.files['image']
        fname = secure_filename(file.filename or "photo.png")

        img = Image.open(file.stream).convert("RGB")

        # Use the new grid manager workflow
        self.set_background_from_pil(img)

        return jsonify({
            "status": "success", 
            "filename": fname, 
            "grid_size": self.res,
            "grid_info": self.grid_manager.get_grid_info()
        })


    def get_agent_svg(self):
        return self.cur_svg_to_render

    def set_sketch_mode(self):
        data = request.get_json()
        self.sketch_mode = data.get("mode", "solo")
        self.init_canvas.save("static/init_canvas.png")
        return jsonify({"status": "success", "message": f"Mode set to {self.sketch_mode}"})

    def setup_path2save(self):
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path2save = f"results/collab_sketching/{self.folder_name}_{self.session_id}/{self.target_concept}/{self.sketch_mode}_{folder_name}"
        if not os.path.exists(self.path2save):
            os.makedirs(self.path2save)
        with open(f"{self.path2save}/data_history.json", "w") as f:
            json.dump([{"session_ID": self.session_id}], f)

    def initialize_all(self):
        self.input_prompt = sketch_first_prompt.format(concept=self.target_concept, gt_sketches_str=gt_example)
        self.all_strokes_svg = self._svg_root_open()
        self.assitant_history = ""
        self.stroke_counter = 0
        self.setup_path2save()

        self.init_canvas = self.base_canvas.copy()
        self.init_canvas.save("static/init_canvas.png")
        self.init_canvas.save("static/cur_canvas_user.png")
        self.init_canvas.save("static/cur_canvas_agent.png")
        if self.sketch_mode == "colab":
            self.init_thinking_tags()

    def get_new_concept(self):
        data = request.get_json()
        self.target_concept = data.get('concept')
        self.initialize_all()
        return jsonify({"target_concept": self.target_concept, "SVG": self.get_agent_svg()})

    def submit_sketch(self):
        self.all_strokes_svg += "</svg>"
        with open(f"{self.path2save}/final_sketch.svg", "w") as svg_file:
            svg_file.write(self.all_strokes_svg)
        cairosvg.svg2png(url=f"{self.path2save}/final_sketch.svg",
                         write_to=f"{self.path2save}/final_sketch.png",
                         background_color="white")
        with open(f"{self.path2save}/data_history.json", "r") as f:
            data = json.load(f)
            data.append({"all_history": self.assitant_history})
        with open(f"{self.path2save}/data_history.json", "w") as f:
            json.dump(data, f)
        return jsonify({"new_category": "yes", "mode": "colab", "message": "Sketch saved! Continue to next concept!"})

    def clear_canvas(self, same_session=True):
        self.all_strokes_svg = self._svg_root_open()
        self.stroke_counter = 0
        self.assitant_history = ""
        bg = self.base_canvas.copy()
        bg.save("static/init_canvas.png")
        bg.save("static/cur_canvas_user.png")
        bg.save("static/cur_canvas_agent.png")
        if same_session and os.path.exists(f"{self.path2save}/sketch.svg"):
            os.remove(f"{self.path2save}/sketch.svg")
        return jsonify({"message": "cleaned!"})

    def index(self):
        return render_template('index.html', target_concept=self.target_concept)

    def shutdown(self):
        self.shutdown_server()
        return 'Server shutting down...'

    def shutdown_server(self):
        os.kill(os.getpid(), signal.SIGINT)

    def update_history(self, txt_update, replace=False):
        if replace:
            self.assitant_history = txt_update
        else:
            self.assitant_history += txt_update
        with open(f"{self.path2save}/data_history.json", "r") as f:
            data = json.load(f)
            data.append({f"stroke_{self.stroke_counter}": self.assitant_history})
        with open(f"{self.path2save}/data_history.json", "w") as f:
            json.dump(data, f)
            
    def _truncate(s, n=1200):
        try:
            return s if len(s) <= n else (s[:n] + f"... <+{len(s)-n} chars>")
        except Exception:
            return s

    def _redact_b64_in_messages(msgs):
        """Make messages JSON-safe & small (remove giant base64 payloads)."""
        redacted = []
        for m in msgs:
            m2 = {"role": m.get("role")}
            parts = []
            for p in m.get("content", []):
                if isinstance(p, dict):
                    p = dict(p)  # shallow copy
                    # redact OpenAI-style data URI
                    if p.get("type") == "image_url":
                        url = p.get("image_url", {}).get("url", "")
                        if isinstance(url, str) and url.startswith("data:image/"):
                            p["image_url"] = {"url": url[:80] + "... <redacted>"}
                    # redact Anthropic-style base64
                    if p.get("type") == "image" and isinstance(p.get("source"), dict):
                        if p["source"].get("type") == "base64":
                            p["source"]["data"] = "<redacted b64>"
                    # pass through text safely
                    if "text" in p and isinstance(p["text"], str):
                        p["text"] = _truncate(p["text"], 2000)
                    parts.append(p)
                else:
                    # strings or anything else
                    parts.append(_truncate(str(p), 2000))
            m2["content"] = parts
            redacted.append(m2)
        return redacted
    
    

    def get_user_stroke(self):
        try:
            data = request.get_json()
            self.user_name = data.get('name')
            sketch_data = data.get('strokes')
            assert len(sketch_data[0]) > 0, "No strokes provided."

            self.stroke_counter += 1
            try:
                user_stroke = self.parse_stroke_from_canvas(sketch_data)
                user_stroke_svg = self.parse_model_to_svg(f"{user_stroke}</s{self.stroke_counter}>")
            except Exception as e:
                traceback.print_exc()
                self.stroke_counter -= 1
                return jsonify({"message": str(e), "status": "error"}), 400

            self.all_strokes_svg += user_stroke_svg
            cur_svg_to_render = f"{self.all_strokes_svg}</svg>"
            with open(f"{self.path2save}/sketch.svg", "w") as svg_file:
                svg_file.write(cur_svg_to_render)

            self._composite_svg_on_base(cur_svg_to_render, "static/cur_canvas_user.png")
            self.update_history(user_stroke)
            if self.sketch_mode == "solo":
                self.update_history(f"</s{self.stroke_counter}>")
            return jsonify({"message": "User strokes received successfully!"})

        except Exception as e:
            traceback.print_exc()
            return jsonify({"message": str(e), "status": "error"}), 400

    def call_agent(self):
        try:
            model_stroke_svg = self.predict_next_stroke()
            self.all_strokes_svg += model_stroke_svg
            self.cur_svg_to_render = f"{self.all_strokes_svg}</svg>"

            with open(f"{self.path2save}/sketch.svg", "w") as f:
                f.write(self.cur_svg_to_render)
            self._composite_svg_on_base(self.cur_svg_to_render, "static/cur_canvas_agent.png")

            if not self.user_always_first:
                self._composite_svg_on_base(self.cur_svg_to_render, "static/init_canvas.png")

            return jsonify({"status": "success", "SVG": self.cur_svg_to_render})

        except Exception as e:
            traceback.print_exc()
            return jsonify({"message": str(e), "status": "error"}), 400

    # ---------- LLM I/O ----------
    def define_input_to_llm(self, msg_history, init_canvas_str: Optional[str], msg: str):
        content = self.llm.build_user_content(init_canvas_str, msg)
        return msg_history + [{"role": "user", "content": content}]

    def call_llm(self, system_message, other_msg, additional_args):
        return self.llm.call(system_message, other_msg, additional_args)

    def get_response_from_llm(
        self,
        msg,
        system_message,
        msg_history=None,
        init_canvas_str: Optional[str] = None,
        prefill_msg: Optional[str] = None,
        seed_mode: str = "stochastic",
        stop_sequences: Optional[str] = None,
        gen_mode: str = "generation",
        **kw,
        
    ):
        
        
        if msg_history is None:
            msg_history = []

        additional_args: Dict = {}
        
        if kw:
            additional_args.update(kw)
        
        if seed_mode == "deterministic":
            additional_args["temperature"] = 0.0
            additional_args["top_k"] = 1  # ignored by OpenAI adapter
            
        if getattr(self, "reasoning_effort", None):
            additional_args["reasoning_effort"] = self.reasoning_effort

        other_msg = self.define_input_to_llm(msg_history, init_canvas_str, msg)

        if gen_mode == "completion" and prefill_msg:
            other_msg = other_msg + [{"role": "assistant", "content": f"{prefill_msg}"}]

        additional_args["stop_sequences"] = stop_sequences if stop_sequences else "</answer>"
        
        # Gemini tends to prematurely STOP on XML-ish stop strings; disable for completion.
        '''
        if isinstance(self.llm, GeminiAdapter) and gen_mode == "completion":
            additional_args.pop("stop_sequences", None)
        '''
        #NOTE: this code was causing duplication issue for GEMINI
        # # Do not use stop-sequences with Gemini (causes premature STOP and empties)
        # if isinstance(self.llm, GeminiAdapter) and "stop_sequences" in additional_args:
        #     additional_args.pop("stop_sequences", None)
        
        
        # optional tiny throttle (helps Gemini avoid empty outputs)
        delay = getattr(self, "api_delay_sec", 0.0) or 0.0
        if delay > 0:
            time.sleep(delay)

        # ---- call provider ----
        response = self.call_llm(system_message, other_msg, additional_args)
        self._last_raw_response = response
        content  = self.llm.extract_text(response)

        if gen_mode == "completion" and prefill_msg:
            other_msg = other_msg[:-1]
        
        # ---- extract generated images (Gemini image model) ----
        self._last_generated_images = []
        if hasattr(self.llm, "extract_images"):
            gen_images = self.llm.extract_images(response)
            self._last_generated_images = gen_images
            if self.path2save and gen_images:
                for idx, img_data_url in enumerate(gen_images):
                    # img_data_url is like "data:image/png;base64,..."
                    if img_data_url.startswith("data:"):
                        header, b64_data = img_data_url.split(",", 1)
                        ext = "png" if "png" in header else "jpeg"
                        img_bytes = base64.b64decode(b64_data)
                        img_path = f"{self.path2save}/generated_image_{idx}.{ext}"
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        print(f"[IMAGE GEN] Saved generated image to: {img_path}")

        if gen_mode == "completion" and prefill_msg:
            other_msg = other_msg[:-1]

        # ----- NEW LOGGING -----
        # ... after you compute `content` and still inside get_response_from_llm
        if self.path2save is not None:
            # (a) Keep your original snapshot (overwrites the JSON)
            system_message_json = [{"role": "system", "content": system_message}]
            new_msg_history = other_msg + [{"role": "assistant", "content": [{"type": "text", "text": content}]}]
            with open(f"{self.path2save}/experiment_log.json", 'w', encoding="utf-8") as json_file:
                json.dump(system_message_json + new_msg_history, json_file, indent=4)

            # (b) Append provider debug row (JSONL) + console preview
            try:
                provider_debug = self.llm.debug_dump(response)
            except Exception as _e:
                provider_debug = {"error": str(_e)}
                
            # ----- NEW: cache for per-item JSONs -----
            self._last_provider_debug = provider_debug
            self._last_assistant_text = content

            try:
                redacted_msgs = self._redact_b64_in_messages(other_msg)
            except Exception:
                redacted_msgs = None
            self._last_redacted_request = {
                "system": system_message,
                "messages": redacted_msgs,
            }


            row = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "adapter": type(self.llm).__name__,
                "model": self.llm.model,
                "seed_mode": seed_mode,
                "gen_mode": gen_mode,
                "stop_sequences": additional_args.get("stop_sequences"),
                "has_image": self.llm.request_has_image(other_msg),
                "assistant_text_preview": (content[:1000] if isinstance(content, str) else None),
                "provider_debug": provider_debug,
            }
            with open(f"{self.path2save}/experiment_log.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Console helper
            print(
                "\n[GEMINI DEBUG]"
                f"\n gen_mode={gen_mode}"
                f"\n stop_sequences={additional_args.get('stop_sequences')}"
                f"\n combined_text preview:\n{(provider_debug.get('combined_text','')[:1000] or '<EMPTY>')}"
                f"\n finish_reason={provider_debug.get('finish_reason')}"
                f"\n prompt_block_reason={provider_debug.get('prompt_block_reason')}"
                "\n[END GEMINI DEBUG]\n",
                flush=True
            )



        return content

    # ---------- sketch flow ----------
    def init_thinking_tags(self):
        add_args = {"stop_sequences": "<strokes>"}
        assistant_suffix = self.get_response_from_llm(
            msg=self.input_prompt,
            system_message=self._sys_prompt_or_none(),
            msg_history=[],
            init_canvas_str=self.last_canvas_b64,
            seed_mode=self.seed_mode,
            gen_mode="generation",
            **add_args
        )
        self.thinking_tags = assistant_suffix + "<strokes>"
        self.update_history(self.thinking_tags)
        if not self.user_always_first:
            self.call_agent()

    def draw_entire_sketch(self):
        all_sketch = self.get_response_from_llm(
            msg=self.input_prompt,
            system_message=self._sys_prompt_or_none(),
            msg_history=[],
            init_canvas_str=self.last_canvas_b64,
            seed_mode=self.seed_mode,
            gen_mode="generation",
            stop_sequences="</answer>"
        )
        all_sketch = self._normalize_listish_blocks(all_sketch)
        
        strokes_list_str, t_values_str = utils.parse_xml_string(all_sketch, res=self.res)
        strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(t_values_str)

        # In --no-grid coord mode, tokens 'xNyM' are treated as pixel coordinates
        # in a top-left origin canvas of size (res_x,res_y) (matching base_canvas).
        if self.no_grid and (not self.positions):
            W, H = self.base_canvas.size
            def _tok_to_xy(tok: str):
                m = re.search(r"x(-?\d+)y(-?\d+)", str(tok))
                if not m:
                    return None
                x = int(m.group(1))
                y = int(m.group(2))
                # Accept either 0-based [0..res] or 1-based [1..res]
                if x >= 1 and x <= self.res_x:
                    x = x - 1
                if y >= 1 and y <= self.res_y:
                    y = y - 1
                    
                px, py = self._coord_to_px_no_grid(float(x), float(y))
                return [px, py]

            all_control_points = []
            for stroke_tokens, stroke_tvals in zip(strokes_list, t_values):
                pts = []
                for tok in stroke_tokens:
                    xy = _tok_to_xy(tok)
                    if xy is not None:
                        pts.append(xy)
                if not pts:
                    continue
                cps = utils.estimate_bezier_control_points(pts, stroke_tvals)
                all_control_points.append(cps)

            sketch_text_svg = utils.format_svg(all_control_points, dim=(W, H), stroke_width=self.stroke_width)
        else:
            all_control_points = utils.get_control_points(strokes_list, t_values, self.positions)
            sketch_text_svg = utils.format_svg(all_control_points, dim=self.grid_size, stroke_width=self.stroke_width)

        with open(f"{self.path2save}/sketch.svg", "w") as svg_file:
            svg_file.write(sketch_text_svg)
            
        cairosvg.svg2png(url=f"{self.path2save}/sketch.svg", write_to=f"static/entire_sketch.png", background_color="white")
        return jsonify({"status": "success", "message": "Sketch drawn!"})

    def parse_stroke_from_canvas(self, sketch_data):
        cur_user_input_stroke = f"<s{self.stroke_counter}>\n<points>"
        cur_stroke = []
        cur_t_values = []
        stroke = sketch_data[0]
        for point_data in stroke:
            x, y, t = point_data['x'], point_data['y'], point_data['timestamp']
            x = min(self.grid_size[0] - 1, max(self.cell_size, x))
            y = min(self.grid_size[1] - 1 - self.cell_size, max(0, y))
            grid_x = int(x // self.cell_size)
            grid_y = int(self.res_y - (y // self.cell_size))
            point_str = f'x{grid_x}y{grid_y}'
            cell_center = self.positions[point_str]
            distance = math.sqrt((x - cell_center[0]) ** 2 + (y - cell_center[1]) ** 2)
            if distance <= 5:
                if (not cur_stroke) or (cur_stroke[-1] != point_str):
                    cur_stroke.append(point_str)
                    cur_t_values.append(t)
                    cur_user_input_stroke += f"'{point_str}', "
        if len(cur_t_values) == 0:
            for point_data in stroke:
                x, y, t = point_data['x'], point_data['y'], point_data['timestamp']
                x = min(self.grid_size[0] - 1, max(self.cell_size, x))
                y = min(self.grid_size[1] - 1 - self.cell_size, max(0, y))
                grid_x = int(x // self.cell_size)
                grid_y = int(self.res_y - (y // self.cell_size))
                point_str = f'x{grid_x}y{grid_y}'
                cell_center = self.positions[point_str]
                distance = math.sqrt((x - cell_center[0]) ** 2 + (y - cell_center[1]) ** 2)
                if distance <= 8:
                    if (not cur_stroke) or (cur_stroke[-1] != point_str):
                        cur_stroke.append(point_str)
                        cur_t_values.append(t)
                        cur_user_input_stroke += f"'{point_str}', "
        assert len(cur_t_values) > 0, "No values recorded from strokes!"
        cur_user_input_stroke = cur_user_input_stroke[:-2] + "</points>\n<t_values>"
        min_time = min(cur_t_values)
        max_time = max(cur_t_values)
        for t in cur_t_values:
            cur_n_t = (t - min_time) / (max_time - min_time) if max_time > min_time else 0.0
            cur_user_input_stroke += f"{cur_n_t:.2f}, "
        cur_user_input_stroke = cur_user_input_stroke[:-2] + "</t_values>"
        return cur_user_input_stroke

    import html  # top of file
    
    
    def _coord_to_px_no_grid(self, cx: float, cy: float):
        """
        Map model coordinates in virtual [0..res_x]x[0..res_y] into actual image pixels (W,H).
        Honors self.prompt_origin:
        - top_left:  y increases downward
        - bottom_left: y increases upward
        """
        W, H = self.grid_size
        if self.res_x <= 0 or self.res_y <= 0:
            raise ValueError("No-grid mode requires --res-x and --res-y > 0")

        # clamp in coord space
        cx = max(0.0, min(float(self.res_x), float(cx)))
        cy = max(0.0, min(float(self.res_y), float(cy)))

        # scale into pixel space
        px = (cx / float(self.res_x)) * float(W - 1)

        if (self.prompt_origin or "bottom_left").lower() == "top_left":
            py = (cy / float(self.res_y)) * float(H - 1)
        else:
            # bottom_left: flip y
            py = (1.0 - (cy / float(self.res_y))) * float(H - 1)

        return px, py


    def parse_model_to_svg(self, stroke_model: str):
        
        stroke_model = self._normalize_listish_blocks(stroke_model)
        
        
        # normalize one-decimal t-values like "0.5" -> "0.50"
        stroke_model = re.sub(
            r'(?<=,|\>)\s*([01])\.([0-9])(?![0-9])',
            lambda m: f"{m.group(1)}.{m.group(2)}0",
            stroke_model
        )

        # Which s-number is this block?
        m_s = re.search(r"<s(\d+)>", stroke_model)
        stroke_no = int(m_s.group(1)) if m_s else max(1, self.stroke_counter + 1)

        # Optional human-readable id
        m_id = re.search(r"<id>(.*?)</id>", stroke_model, re.S)
        stroke_label = (m_id.group(1).strip() if m_id else f"s{stroke_no}")
        stroke_label = re.sub(r"[^\w\-]", "_", stroke_label)

        # Pink/green color parity (unchanged)
        stroke_color = "green"
        if self.sketch_mode == "colab":
            if self.user_always_first:
                if stroke_no % 2 == 0:
                    stroke_color = "pink"
            else:
                if stroke_no % 2 == 1:
                    stroke_color = "pink"
                    
        


        # ----- NEW: TEXT STROKE SUPPORT (with size/color) -----
        m_text = re.search(r"<text([^>]*)>\s*'([^']+)'\s*</text>", stroke_model, re.S)
        if m_text:
            # anchor cell
            m_ptblk = re.search(r"<points>(.*?)</points>", stroke_model, re.S)
            if not m_ptblk:
                raise ValueError(f"Text stroke s{stroke_no} missing <points>")
            pts = re.findall(r"'x(\d+)y(\d+)'", m_ptblk.group(1))
            if not pts:
                raise ValueError(f"Text stroke s{stroke_no} has no valid xAyB point")
            gx, gy = pts[0]
            
            key = f"x{int(gx)}y{int(gy)}"

            if self.no_grid:
                # Treat text anchor as coord-space point scaled onto the original image
                W, H = self.grid_size
                if self.res_x <= 0 or self.res_y <= 0:
                    raise ValueError("No-grid mode requires --res-x and --res-y > 0")

                cx, cy = self._coord_to_px_no_grid(float(gx), float(gy))
            else:
                if key not in self.positions:
                    raise ValueError(f"Text stroke s{stroke_no} uses out-of-grid cell {key}")
                cx, cy = self.positions[key]


            # style overrides
            font_px_override, color_override = self._parse_text_style(stroke_model)
            default_px = self.cell_size * self.text_font_scale
            font_px = int(round(font_px_override if font_px_override is not None else default_px))

            # color priority: explicit style -> previous parity color -> 'black'
            fill_color = color_override or stroke_color or "black"

            text_val = html.escape(m_text.group(2))

            return (
                f'<g id="{stroke_label}_s{stroke_no}">'
                f'<text x="{cx:.1f}" y="{cy:.1f}" '
                f'text-anchor="middle" dominant-baseline="central" '
                f'font-family="{self.text_font_family}" '
                f'font-size="{font_px}" fill="{fill_color}">{text_val}</text>'
                f'</g>'
            )


        # ================================================================
        # No-grid coordinate strokes (top-left origin)
        # ================================================================
        # In your coord trial (--no-grid with --res-x/--res-y set), we expect:
        #   <points>(x0,y0),(x1,y1),...</points>
        # and interpret them in a 0..res_x by 0..res_y space with origin at
        # the TOP-LEFT. We scale to the current canvas size.
        if self.no_grid and self.grid_size == (self.res_x, self.res_y) and self.res_x > 0 and self.res_y > 0:
            m_ptblk = re.search(r"<points>(.*?)</points>", stroke_model, re.S | re.I)
            if not m_ptblk:
                raise ValueError(f"Stroke s{stroke_no} missing <points>")

            pts_text = m_ptblk.group(1)
            # tolerate: (500,105),(500,431)  or 500,105 500,431  or [500,105]
            coord_re = re.compile(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?")
            coords = [(float(a), float(b)) for a, b in coord_re.findall(pts_text)]
            if not coords:
                # If the model accidentally used xNyM format here, fall through
                # to the grid-based parser below.
                coords = None

            if coords:
                # t-values
                m_t = re.search(r"<t_values>(.*?)</t_values>", stroke_model, re.S | re.I)
                if m_t:
                    raw = m_t.group(1).strip().strip("[]")
                    parts = [p.strip() for p in raw.split(",") if p.strip()]
                    try:
                        t_values = [float(p) for p in parts]
                    except Exception:
                        t_values = []
                else:
                    t_values = []

                n = len(coords)
                if not t_values or len(t_values) != n:
                    if n <= 1:
                        t_values = [0.0, 1.0]
                    elif n == 2:
                        t_values = [0.0, 1.0]
                    else:
                        t_values = [i/(n-1) for i in range(n)]

                # scale coords -> pixels; clamp into the canvas.
                W, H = self.grid_size
                sx = (W - 1) / float(self.res_x) if self.res_x else 1.0
                sy = (H - 1) / float(self.res_y) if self.res_y else 1.0
                sampled_points = []
                for x, y in coords:
                    px = max(0.0, min(float(W - 1), x * sx))
                    py = max(0.0, min(float(H - 1), y * sy))
                    sampled_points.append([px, py])  # (x,y) order for SVG

                group = utils.estimate_bezier_control_points(sampled_points, t_values)
                return utils.format_svg_single_stroke(
                    group,
                    dim=self.grid_size,
                    stroke_width=self.stroke_width,
                    stroke_counter=stroke_no,
                    group_id=stroke_label,
                    stroke_color=stroke_color,
                )


        # Bounding Box Support
        # Try to detect an axis-aligned rectangle from a <points> block
        # Bounding Box Support
                
        m_ptblk = re.search(r"<points>(.*?)</points>", stroke_model, re.S)
        if m_ptblk:
            # Parse grid points like: 'x6y6','x63y6','x63y43','x6y43','x6y6'
            grid_pts = [(int(gx), int(gy)) for gx, gy in re.findall(r"x(\d+)y(\d+)", m_ptblk.group(1))]
            if len(grid_pts) >= 4:
                # Map grid tokens → pixel coords using the SAME mapping as everything else
                px_pts = []
                for gx, gy in grid_pts:
                    key = f"x{gx}y{gy}"
                    if key in self.positions:
                        px_pts.append(self.positions[key])

                # Require 4 unique corners (or 5 with explicit closure first==last)
                if len(px_pts) >= 4:
                    closed = (px_pts[0] == px_pts[-1])
                    uniq = px_pts[:-1] if closed else px_pts
                    # Unique corners
                    uniq_corners = list(dict.fromkeys(uniq))  # preserve order, de-dup
                    if len(uniq_corners) == 4:
                        xs = sorted({x for x, _ in uniq_corners})
                        ys = sorted({y for _, y in uniq_corners})
                        # Axis aligned if there are exactly two unique x’s and two unique y’s
                        if len(xs) == 2 and len(ys) == 2:
                            x0, x1 = xs[0], xs[1]
                            y0, y1 = ys[0], ys[1]
                            x, y = min(x0, x1), min(y0, y1)
                            w, h = abs(x1 - x0), abs(y1 - y0)

                            rect_svg = (
                                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                                f'stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none" '
                                f'shape-rendering="crispEdges" vector-effect="non-scaling-stroke"/>'
                            )
                            return f'<g id="{stroke_label}_s{stroke_no}">{rect_svg}</g>'


        # (no rectangle) → fall through to your EXISTING Bezier/path code unchanged
        
        
        
        # ================================================================
        # NO-GRID + xNyM TOKENS (coords in [0..res_x]x[0..res_y], scaled to image px)
        # ================================================================
        if self.no_grid:
            m_ptblk = re.search(r"<points>(.*?)</points>", stroke_model, re.S)
            if not m_ptblk:
                raise ValueError(f"Stroke s{stroke_no} missing <points>")

            pts = re.findall(r"x(\d+)y(\d+)", m_ptblk.group(1))
            if pts:
                W, H = self.grid_size  # actual image pixel size
                if self.res_x <= 0 or self.res_y <= 0:
                    raise ValueError("No-grid mode requires --res-x and --res-y > 0")

                sampled_points = []
                for gx, gy in pts:
                    # clamp in coordinate space
                    cx = max(0.0, min(float(self.res_x), float(gx)))
                    cy = max(0.0, min(float(self.res_y), float(gy)))

                    # scale into pixel space (top-left origin)
                    px, py = self._coord_to_px_no_grid(float(gx), float(gy))
                    sampled_points.append([px, py])
                    
                # t-values
                m_t = re.search(r"<t_values>(.*?)</t_values>", stroke_model, re.S)
                if m_t:
                    raw = m_t.group(1).strip().strip("[]")
                    t_values = [float(p) for p in raw.split(",") if p.strip()]
                else:
                    t_values = []

                n = len(sampled_points)
                if not t_values or len(t_values) != n:
                    t_values = [i / (n - 1) if n > 1 else 0.0 for i in range(n)]

                group = utils.estimate_bezier_control_points(sampled_points, t_values)
                return utils.format_svg_single_stroke(
                    group,
                    dim=self.grid_size,
                    stroke_width=self.stroke_width,
                    stroke_counter=stroke_no,
                    group_id=stroke_label,
                    stroke_color=stroke_color,
                )




        # ----- default: curve/path stroke (unchanged) -----
        strokes_list_str, t_values_str = utils.parse_xml_string_single_stroke(
            stroke_model, self.res, stroke_no, self.res_x, self.res_y
        )
        strokes_list = ast.literal_eval(strokes_list_str)
        t_values     = ast.literal_eval(t_values_str)
        
        strokes_list = ast.literal_eval(strokes_list_str)
        t_values     = ast.literal_eval(t_values_str)

        if len(t_values) != len(strokes_list):
            n = len(strokes_list)
            if n <= 1:
                t_values = [0.00] * n
            else:
                # evenly spaced fallback
                t_values = [round(i/(n-1), 2) for i in range(n)]


        all_control_points = utils.get_control_points_single_stroke(
            strokes_list, t_values, self.positions
        )
        return utils.format_svg_single_stroke(
            all_control_points,
            dim=self.grid_size,
            stroke_width=self.stroke_width,
            stroke_counter=stroke_no,         # use stroke_no, not self.stroke_counter
            group_id=stroke_label,
            stroke_color=stroke_color
        )


    def call_model_stroke_completion(self):
        # compute the next stroke number we expect
        expected = self.stroke_counter + 1

        # add a tiny guidance line – appended to the existing seeded prompt
        one_stroke_guard = (
            f"\n\n[Stepwise mode]\n"
            f"Return EXACTLY ONE stroke: <s{expected}> ... </s{expected}> only.\n"
            f"Do NOT output any other strokes.\n"
            f"Do NOT output <final_answer> until explicitly asked.\n"
            f"Stop immediately after </s{expected}>."
        )


        # use a dynamic stop sequence that matches the next stroke’s closing tag
        answer = self.get_response_from_llm(
            msg=(self.input_prompt + one_stroke_guard),
            system_message=self._sys_prompt_or_none(),
            msg_history=[],
            init_canvas_str=self.last_canvas_b64,
            seed_mode=self.seed_mode,
            gen_mode="completion",
            prefill_msg=self.assitant_history.strip(),
            stop_sequences=f"</s{expected}>",   # ← key change
        )

        if self.multi_stroke:
            return self._next_undrawn(answer)
        else:
            expected = self.stroke_counter + 1
            return self._next_undrawn(answer, expected_s_no=expected)



    def predict_next_stroke(self):
        blocks = self.call_model_stroke_completion()
        svgs: List[str] = []

        # expected next number based on *our* running count
        expected = self.stroke_counter + 1

        for blk in blocks:
            # parse/paint using the block’s internal <s#> for the XML parser,
            # but advance our own counter exactly once per block we actually render.
            svg = self.parse_model_to_svg(blk)
            svgs.append(svg)
            self.update_history(blk)

            # advance our global stroke count by one
            self.stroke_counter = expected
            expected += 1
            
            # maintain a human-colored running SVG alongside model-colored one
            if self.cycle_stroke_colors:
                color = self._palette[(self.stroke_counter-1) % len(self._palette)]
                colored_group = self._recolor_svg_group(svg, color)
                self._colored_svg += colored_group

        return "".join(svgs)
    
    
    '''
    def set_background_from_pil(self, img_pil: Image.Image):
        """
        Programmatic version of /upload-image: resize, overlay grid, set as base.
        """
        img = img_pil.convert("RGB").resize(self.grid_size, Image.LANCZOS)

        grid = self.init_canvas_grid.convert("RGBA")
        mask = grid.convert("L").point(lambda p: 255 if p < 200 else 0)

        img = img.convert("RGBA")
        img.paste(grid, (0, 0), mask)

        self.base_canvas = img.convert("RGB")
        self.base_canvas.save("static/init_canvas.png")
        self.base_canvas.save("static/cur_canvas_user.png")
        self.base_canvas.save("static/cur_canvas_agent.png")
        self._update_canvas_b64("static/init_canvas.png")
    '''

    def _retag_block_to_no(self, blk: str, new_no: int) -> str:
        """
        Make the model's <sN> match our current expected stroke number so
        utils.parse_xml_string_single_stroke doesn't complain.
        """
        blk = re.sub(r"<s\d+>",  f"<s{new_no}>",  blk, count=1)
        blk = re.sub(r"</s\d+>", f"</s{new_no}>", blk, count=1)
        return blk

    def _count_strokes(self, answer_xml: str, count_only_text: bool = True) -> int:
        """
        Count predicted strokes.

        Supports two forms:
        (A) Raw assistant XML with <strokes><s1>...</s1>...</strokes>
        (B) Final rendered SVG groups (<g id="..._sN"> ... <text ...> ... </text> </g>)

        If count_only_text=True, count only strokes that contain a <text ...>...</text>.
        Otherwise, count all strokes.
        """
        xml_str = str(answer_xml or "")
        # strip accidental codefences
        xml_str = re.sub(r"^```(?:xml|html)?\s*|\s*```$", "", xml_str.strip())

        # ---------- Case A: assistant XML with <sN> blocks ----------
        s_blocks = re.findall(r"(<s\d+>.*?</s\d+>)", xml_str, flags=re.S | re.I)
        if s_blocks:
            if not count_only_text:
                return len(s_blocks)
            # <text ...> 'value' | "value" | bare value </text>  (attributes tolerated)
            text_tag = re.compile(
                r"<text(?:\s+[^>]*)?>\s*(?:'[^']*'|\"[^\"]*\"|[0-9A-Za-z._\-]+)\s*</text>",
                flags=re.S | re.I
            )
            return sum(1 for b in s_blocks if text_tag.search(b))

        # ---------- Case B: rendered SVG without <sN>, but with <g id="..._sN"> ----------
        # Count groups with stroke-like ids
        g_blocks = re.findall(
            r"(<g\b[^>]*\bid\s*=\s*['\"][^'\"]*_s\d+['\"][^>]*>.*?</g>)",
            xml_str, flags=re.S | re.I
        )
        if g_blocks:
            if not count_only_text:
                return len(g_blocks)
            text_tag = re.compile(
                r"<text(?:\s+[^>]*)?>\s*(?:'[^']*'|\"[^\"]*\"|[0-9A-Za-z._\-]+)\s*</text>",
                flags=re.S | re.I
            )
            return sum(1 for g in g_blocks if text_tag.search(g))

        # ---------- Fallback: count any <text> elements in the string ----------
        # (This handles odd outputs; safer than returning 0.)
        """
        Count predicted strokes.

        Supports two forms:
        (A) Raw assistant XML with <strokes><s1>...</s1>...</strokes>
        (B) Final rendered SVG groups (<g id="..._sN"> ... <text ...> ... </text> </g>)

        If count_only_text=True, count only strokes that contain a <text ...>...</text>.
        Otherwise, count all strokes.
        """
        xml_str = str(answer_xml or "")
        # strip accidental codefences
        xml_str = re.sub(r"^```(?:xml|html)?\s*|\s*```$", "", xml_str.strip())

        # ---------- Case A: assistant XML with <sN> blocks ----------
        s_blocks = re.findall(r"(<s\d+>.*?</s\d+>)", xml_str, flags=re.S | re.I)
        if s_blocks:
            if not count_only_text:
                return len(s_blocks)
            # <text ...> 'value' | "value" | bare value </text>  (attributes tolerated)
            text_tag = re.compile(
                r"<text(?:\s+[^>]*)?>\s*(?:'[^']*'|\"[^\"]*\"|[0-9A-Za-z._\-]+)\s*</text>",
                flags=re.S | re.I
            )
            return sum(1 for b in s_blocks if text_tag.search(b))

        # ---------- Case B: rendered SVG without <sN>, but with <g id="..._sN"> ----------
        # Count groups with stroke-like ids
        g_blocks = re.findall(
            r"(<g\b[^>]*\bid\s*=\s*['\"][^'\"]*_s\d+['\"][^>]*>.*?</g>)",
            xml_str, flags=re.S | re.I
        )
        if g_blocks:
            if not count_only_text:
                return len(g_blocks)
            text_tag = re.compile(
                r"<text(?:\s+[^>]*)?>\s*(?:'[^']*'|\"[^\"]*\"|[0-9A-Za-z._\-]+)\s*</text>",
                flags=re.S | re.I
            )
            return sum(1 for g in g_blocks if text_tag.search(g))

        # ---------- Fallback: count any <text> elements in the string ----------
        # (This handles odd outputs; safer than returning 0.)
        if count_only_text:
            return len(re.findall(
                r"<text(?:\s+[^>]*)?>\s*(?:'[^']*'|\"[^\"]*\"|[0-9A-Za-z._\-]+)\s*</text>",
                xml_str, flags=re.S | re.I
            ))
        else:
            # As a last resort, count text nodes as strokes
            return len(re.findall(r"<text\b", xml_str, flags=re.I))

    def _extract_final_answer_any(self, *candidates: Optional[str]) -> Optional[str]:
        """
        Return the raw inner text from the first <final_answer>...</final_answer> found
        across the provided candidate strings. No numeric coercion—whatever is inside
        the tag is returned (trimmed). If not found, returns None.
        """
        import html as _html
        pat = re.compile(r"<final_answer\b[^>]*>(.*?)</final_answer>", re.S | re.I)

        for s in candidates:
            if not s:
                continue
            # strip accidental fences so the regex can see the tag
            t = re.sub(r"^```(?:xml|html)?\s*|\s*```$", "", str(s).strip())
            m = pat.search(t)
            if not m:
                continue
            inner = m.group(1).strip()
            # unescape any HTML entities the model might have produced
            inner = _html.unescape(inner)
            # also trim surrounding quotes if the model wrapped the answer
            if (inner.startswith(("'", '"')) and inner.endswith(("'", '"')) 
                    and len(inner) >= 2):
                inner = inner[1:-1].strip()
            return inner

        return None


    def _render_answer_xml(self, answer_xml: str, svg_out: Path, png_out: Path):
        """
        Take a full <strokes>...</strokes> answer and render *all* strokes to SVG+PNG.
        """
        # reset per-sample drawing state
        self.all_strokes_svg = self._svg_root_open()
        self.stroke_counter = 0
        self.cur_svg_to_render = "None"

        # extract blocks in order
        blocks = re.findall(r"(<s\d+>.*?</s\d+>)", answer_xml, re.S)

        # ── save the (grid + photo) background *before* compositing strokes
        png_out.parent.mkdir(parents=True, exist_ok=True)
        orig_with_grid = png_out.with_name(png_out.stem.replace("_annotated", "_grid") + png_out.suffix)
        (getattr(self, "base_canvas_clean", self.base_canvas)).save(str(orig_with_grid))


        for blk in blocks:
            self.stroke_counter += 1
            blk_fixed = self._retag_block_to_no(blk, self.stroke_counter)
            svg = self.parse_model_to_svg(blk_fixed)
            self.all_strokes_svg += svg

        self.cur_svg_to_render = f"{self.all_strokes_svg}</svg>"
        svg_out.parent.mkdir(parents=True, exist_ok=True)
        with open(svg_out, "w", encoding="utf-8") as f:
            f.write(self.cur_svg_to_render)

        # composite onto the canvas and write PNG
        self._composite_svg_on_base(self.cur_svg_to_render, str(png_out))
    
    # --- aspect-ratio aware image placement ---------------------------------
    def _fit_image_to_canvas(self, img: Image.Image, mode: str = "fit", bgcolor=(255, 255, 255)) -> Image.Image:
        """
        Place image at its natural size at the bottom-left corner of the grid.
        No scaling or stretching - just positioning the original image.
        """
        img = ImageOps.exif_transpose(img)  # respect EXIF orientation
        W, H = self.grid_size
        
        # Create canvas
        canvas = Image.new("RGB", (W, H), bgcolor)
        
        # Position image at bottom-left corner (aligned with grid)
        x = self.cell_size  # Start right after the header column
        y = H - self.cell_size - img.height  # Bottom aligned (above bottom header row)
        
        # Paste the image at its natural size
        canvas.paste(img.convert("RGB"), (x, y))
        return canvas


    def _overlay_grid(self, img_rgb: Image.Image) -> Image.Image:
        """Overlay the current grid on top of `img_rgb` (no distortion)."""
        return self.grid_manager.overlay_grid(img_rgb)


    def _set_base_canvas(self, img_rgb: Image.Image):
        """Persist as the new background everywhere + refresh b64 copy."""
        self.base_canvas = img_rgb.convert("RGB")
        self.base_canvas.save("static/init_canvas.png")
        self.base_canvas.save("static/cur_canvas_user.png")
        self.base_canvas.save("static/cur_canvas_agent.png")
        self._update_canvas_b64("static/init_canvas.png")


    def _update_grid_for_image(self, img: Image.Image):
        """Update grid size based on image dimensions if dynamic_grid is enabled."""
        if self.dynamic_grid:
            grid_changed = self.grid_manager.update_grid_for_image(img, self.show_full_grid)
            
            if grid_changed:
                # Update backward compatibility properties
                self.res_x = self.grid_manager.res_x
                self.res_y = self.grid_manager.res_y
                self.res = max(self.res_x, self.res_y)  # Keep for backward compatibility
                self.num_cells = max(self.res_x, self.res_y)  # Keep for backward compatibility
                self.grid_size = self.grid_manager.grid_size
                self.init_canvas_grid = self.grid_manager.grid_image
                self.positions = self.grid_manager.positions

    def set_background_from_pil(self, pil_img: Image.Image, mode: str = "fit", bgcolor=(255, 255, 255)):
        """Public helper used by batch eval to mimic an uploaded image."""

        # -------- no-grid coord mode --------
        # If --no-grid is enabled and the app was configured with an explicit
        # (res_x,res_y) canvas, we *do not* use the GridManager's header offsets.
        # Instead we resize the raw image to exactly fill the canvas and interpret
        # model coordinates directly in this top-left pixel space.
        if self.no_grid and self.res_x > 0 and self.res_y > 0:
            # NO-GRID coordinate mode:
            # - Keep the original image pixel size (do NOT resize).
            # - Interpret model coordinates in a virtual [0..res_x]x[0..res_y] space,
            #   and scale them onto this image elsewhere (parse_model_to_svg).
            img_rgb = ImageOps.exif_transpose(pil_img).convert("RGB")

            # Make downstream SVG/overlay use the true image pixel dimensions.
            self.grid_size = img_rgb.size

            # No grid cell centers are used in this mode.
            self.positions = {}

            self._set_base_canvas(img_rgb)
            self.base_canvas_clean = self.base_canvas.copy()
            return

        if self.dynamic_grid:
            # Dynamic-grid mode. For --no-grid coordinate trials we must NOT
            # place the image into a larger grid canvas (which changes the pixel
            # size and introduces offsets). Instead, keep the original image
            # size and render strokes proportionally onto it.
            if self.no_grid:
                composited = pil_img.convert("RGB")
                self.grid_size = composited.size
                self.init_canvas_grid = None
                self.positions = {}
                # NOTE: keep self.res_x/self.res_y as the user-provided coordinate space
            else:
                # Recompute grid, place the raw image, then overlay the grid
                self.grid_manager.update_grid_for_image(pil_img, self.show_full_grid)
                placed = self.grid_manager.place_image_at_bottom_left(pil_img, bgcolor)
                composited = self.grid_manager.overlay_grid(placed)

                # Update backward compatibility properties
                self.res_x = self.grid_manager.res_x
                self.res_y = self.grid_manager.res_y
                self.res = max(self.res_x, self.res_y)
                self.num_cells = max(self.res_x, self.res_y)
                self.grid_size = self.grid_manager.grid_size
                self.init_canvas_grid = self.grid_manager.grid_image
                self.positions = self.grid_manager.positions
        else:
            # Fall back to old method for static grids
            #
            # IMPORTANT for --no-grid coordinate trials:
            #   Do NOT resize the input image to the grid canvas. We keep the original
            #   pixel size so that strokes (in a 0..res_x / 0..res_y coordinate space)
            #   are overlaid proportionally on the original image.
            if self.no_grid:
                composited = pil_img.convert("RGB")
                # Ensure downstream SVG/PNG composition uses the true image dimensions.
                self.grid_size = composited.size
                # No grid => no grid-cell lookup map.
                self.positions = {}
            else:
                self._update_grid_for_image(pil_img)
                placed = self._fit_image_to_canvas(pil_img.convert("RGB"), mode=mode, bgcolor=bgcolor)
                composited = self._overlay_grid(placed)
        
        
        # Track raw image size + where it was placed on the canvas (used for saving annotated images without grid)
        self._raw_image_size = ImageOps.exif_transpose(pil_img).size
        if self.no_grid:
            self._raw_image_offset = (0, 0)
        else:
            # Both dynamic_grid and static_grid place the image shifted right by one cell for the left label column.
            cs = getattr(self, "cell_size", None)
            if getattr(self, "dynamic_grid", False) and hasattr(self, "grid_manager"):
                cs = getattr(self.grid_manager, "cell_size", cs)
            self._raw_image_offset = (int(cs or 0), 0)
               
        self._set_base_canvas(composited)
        self.base_canvas_clean = self.base_canvas.copy()  # pristine background for stepwise + final render
        
        

    
    def _sanitize_color(self, c):
        """Return a safe SVG color string or None."""
        if not c:
            return None
        c = str(c).strip()
        # #RGB, #RRGGBB, #RRGGBBAA
        if re.match(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$", c):
            return c
        # rgb() / rgba()
        if re.match(r"^rgba?\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}(?:\s*,\s*(?:0|1|0?\.\d+))?\s*\)$", c):
            return c
        # CSS named color (basic validation)
        if re.match(r"^[a-zA-Z]{3,20}$", c):
            return c
        return None


    def _parse_text_style(self, stroke_xml):
        """
        Parse font size + color for a text stroke.
        Returns (font_px_or_None, color_or_None).
        - size can be '2.2' (cells) or '38px' (pixels).
        - Also supports <style>…</style> or <style font_size="…" color="…"/>.
        """
        size_val = None
        color_val = None

        # <text size="…" color="…">
        m_text_tag = re.search(r"<text([^>]*)>", stroke_xml)
        if m_text_tag:
            attrs = m_text_tag.group(1)
            m_sz = re.search(r'\bsize\s*=\s*["\']([^"\']+)["\']', attrs)
            if m_sz:
                size_val = m_sz.group(1).strip()
            m_col = re.search(r'\bcolor\s*=\s*["\']([^"\']+)["\']', attrs)
            if m_col:
                color_val = m_col.group(1).strip()

        # <style> font_size / color </style>
        m_style = re.search(r"<style\b[^>]*>(.*?)</style>", stroke_xml, re.S)
        if m_style:
            inner = m_style.group(1)
            m_sz2 = re.search(r"<font_size>\s*([^<]+)\s*</font_size>", inner)
            if m_sz2:
                size_val = m_sz2.group(1).strip()
            m_col2 = re.search(r"<color>\s*([^<]+)\s*</color>", inner)
            if m_col2:
                color_val = m_col2.group(1).strip()

        # <style font_size="…" color="…" />
        m_style2 = re.search(r"<style\b([^>]*)/>", stroke_xml)
        if m_style2:
            attrs = m_style2.group(1)
            m_sz3 = re.search(r'\bfont_size\s*=\s*["\']([^"\']+)["\']', attrs)
            if m_sz3:
                size_val = m_sz3.group(1).strip()
            m_col3 = re.search(r'\bcolor\s*=\s*["\']([^"\']+)["\']', attrs)
            if m_col3:
                color_val = m_col3.group(1).strip()

        # convert size to px
        font_px = None
        if size_val:
            s = size_val.lower()
            try:
                if s.endswith("px"):
                    font_px = float(s[:-2])
                else:
                    # treat as "grid-cell multiplier"
                    mult = float(re.sub(r"[^\d\.]+", "", s))
                    font_px = mult * self.cell_size
            except Exception:
                font_px = None

        # clamp to sensible range
        if font_px is not None:
            lo = self.cell_size * 0.8
            hi = self.cell_size * 6.0
            font_px = max(lo, min(font_px, hi))

        return (font_px, self._sanitize_color(color_val))
    

    def _canon_strokes(self, s: str) -> str:
        """Coerce any provider output into a clean <strokes>…</strokes> string."""
        if not isinstance(s, str):
            s = str(s or "")
        s = s.strip()

        # Strip any code fences or duplicated prologs/chatter
        s = re.sub(r"```.*?```", "", s, flags=re.S)
        s = re.sub(r"^<\?xml[^>]*\?>", "", s)

        # If there is an explicit <strokes>…</strokes>, keep exactly that section
        m = re.search(r"<strokes\b[^>]*>(.*?)</strokes>", s, re.S)
        if m:
            return "<strokes>" + m.group(1) + "</strokes>"

        # Otherwise, collect all <sN>…</sN> blocks if any; else return empty strokes
        blocks = re.findall(r"(<s\d+>.*?</s\d+>)", s, re.S)
        return "<strokes>" + "".join(blocks) + "</strokes>"
    
    def _normalize_listish_blocks(self, xml: str) -> str:
        """
        Make <points> and <t_values> blocks canonical:
        <points>'x1y2','x3y4'</points>
        <t_values>0.00,0.33,1.00</t_values>
        Accepts variants like:
        <points>['x1y2', "x3y4"]</points>
        <t_values>[0, 0.5, 1]</t_values>
        Also strips code fences/JSON noise if any slipped in.
        """
        if not isinstance(xml, str):
            return xml

        s = xml

        # strip code fences or accidental JSON blobs around the XML
        s = re.sub(r"^```.*?```", "", s, flags=re.S)
        s = re.sub(r"^<\?xml[^>]*\?>", "", s)

        # normalize curly or backtick quotes to straight quotes
        s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("`", "'")

        def norm_points(m):
            body = m.group(1).strip()
            # drop outer []/() if present
            body = re.sub(r"^[\[\(]\s*|\s*[\]\)]$", "", body)
            # pick tokens like x12y34 (ignore quotes/commas)
            pts = re.findall(r"x\d+y\d+", body)
            return "<points>" + ",".join(f"'{p}'" for p in pts) + "</points>"

        def norm_tvals(m):
            body = m.group(1).strip()
            body = re.sub(r"^[\[\(]\s*|\s*[\]\)]$", "", body)
            nums = re.findall(r"-?\d+(?:\.\d+)?", body)
            # format to 2dp; clamp to [0,1] but don't crash if outside
            vals = []
            for x in nums:
                try:
                    v = float(x)
                except Exception:
                    continue
                v = max(min(v, 1.0), 0.0)
                vals.append(f"{v:.2f}")
            return "<t_values>" + ",".join(vals) + "</t_values>"

        s = re.sub(r"<points>(.*?)</points>", norm_points, s, flags=re.S|re.I)
        s = re.sub(r"<t_values>(.*?)</t_values>", norm_tvals, s, flags=re.S|re.I)
        return s

  

    def _sha1_bytes(self, b: bytes) -> str:
        h = hashlib.sha1(); h.update(b); return h.hexdigest()

    def _canon_strokes_context(self) -> str:
        """
        Return strokes-so-far as <strokes>...</strokes> built from all <sN>...</sN>
        blocks we’ve appended into self.assitant_history. This keeps the original
        grid-anchored XML format (xAyB points, text tags, etc.) that the model expects.
        """
        s = str(self.assitant_history or "")
        # Collect all <sN>...</sN> blocks in order
        blocks = re.findall(r"(<s\d+>.*?</s\d+>)", s, flags=re.S | re.I)
        return "<strokes>" + "".join(blocks) + "</strokes>"

        
    

    def _start_generic_session(self, user_prompt: str):
        """
        Seed a session for arbitrary prompts by appending a gated capability toolkit,
        then end at <strokes> exactly like the UI so Gemini is 'in protocol'.
        """
        
        # If user asked for raw prompt only, do exactly that.
        if self.no_system_prompt:
            self.input_prompt = user_prompt.strip()
            assistant_suffix = self.get_response_from_llm(
                msg=self.input_prompt,
                system_message=None,           # no system
                msg_history=[],
                init_canvas_str=self._canvas_b64_for_model(),
                seed_mode=self.seed_mode,
                gen_mode="generation",
                stop_sequences=None,           # don’t force a <strokes>-based stop
            )
            # No protocol tags added
            self.thinking_tags = assistant_suffix
            self.update_history(self.thinking_tags)
            return
            
        
        # Build the seeded instruction (raw prompt + gated toolkit)
        # Keep your existing system_prompt(res=...) for grid details.
        self.input_prompt = (
            f"{user_prompt.strip()}\n\n"
            "Use the following capability patterns only if relevant to the request:\n"
            f"{MIX_TOOLKIT}\n"
            # Ask the model to produce its header and stop right before strokes
            "Begin your answer now. Stop after writing the header that precedes <strokes>."
        )

        add_args = {"stop_sequences": "<strokes>"}  # dropped automatically for Gemini in get_response_from_llm
        assistant_suffix = self.get_response_from_llm(
            msg=self.input_prompt,
            system_message=self._sys_prompt_or_none(),
            msg_history=[],
            init_canvas_str=self._canvas_b64_for_model(),
            seed_mode=self.seed_mode,
            gen_mode="generation",
            **add_args
        )

        # Store the header and open the strokes section
        # ROBUSTNESS: If model ignored instruction and output strokes, extract just the header
        if "<strokes>" in assistant_suffix:
            # Model output everything - extract just what comes before <strokes>
            assistant_suffix = assistant_suffix.split("<strokes>")[0]
        self.thinking_tags = assistant_suffix + "<strokes>"
        self.update_history(self.thinking_tags)
        
        
    
        

    def evaluate_mixed_folder(
        self,
        src_dir: str = "datasets/mix",
        outdir: str = "results/mix_eval",
        stepwise: bool = False,
        max_images: int = None,
        max_turns: int = 40,
        count_only_text: bool = True,
        skip: int = 0,
        only: str = None,
        two_turn: bool = False,
        counting: bool = False,
        labeling: bool = False, 
        labels_hint: str = None,
    ):
        
        def _capture_turn_input(system_message: str, user_text: str, svg_so_far: str, step_png_path: str=None):
            return {
                "system": (system_message or ""),
                "user_text": (user_text or ""),
                "strokes_so_far": (svg_so_far or ""),
                "step_png": (str(step_png_path) if step_png_path else None),
            }

        
        src = Path(src_dir)
        assert src.exists() and src.is_dir(), f"Folder not found: {src_dir}"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(outdir) / ts
        out_root.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        images = [p for p in sorted(src.iterdir()) if p.suffix.lower() in exts]
        if max_images is not None:
            images = images[:max_images]

        results = []
        pbar = tqdm(images, desc=f"Mixed folder ({'stepwise' if stepwise else 'single-shot'})", unit="img") \
            if hasattr(tqdm, "__call__") else images
            
            
        # Simple comma-separated list -> set of ints (e.g., "5, 12, 42")
        only_set = None
        if only:
            tokens = re.split(r"[,\s]+", only.strip())
            only_set = {int(t) for t in tokens if t.isdigit()}

        for i, img_path in enumerate(pbar):
            
            if only_set is not None and i not in only_set:
                continue
            if only_set is None and i < skip:
                continue
    
            txt_path = img_path.with_suffix(".txt")
            prompt_from_txt = None
            turns = 0
            turn_trace = []  # per-turn captures
            raw_path = out_root / f"item_{i:05d}_orig.jpg"
            svg_path = out_root / f"item_{i:05d}.svg"
            png_path = out_root / f"item_{i:05d}_annotated.png"

            try:
                if not txt_path.exists():
                    raise FileNotFoundError(f"Missing prompt file: {txt_path.name}")

                img = Image.open(img_path).convert("RGB")
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt_from_txt = f.read().strip()
                    
                # Decide seed prompt (prepend counting/labeling header if requested)
                if counting:
                    if self.no_system_prompt:
                        seed_prompt = prompt_from_txt.strip()
                    else:
                        thing = re.sub(r"^[Hh]ow many\s+|\s+are there.*$", "", prompt_from_txt or "").strip() or "object"
                        seed_prompt = COUNTING_PROMPT.format(object=thing)
                elif labeling:
                    # Minimal concept inference; fall back to "object"
                    # (You can make this smarter later, but keeping this tiny on purpose.)
                    m = re.search(r"(?:parts|of)\s+(?:the|a|an)\s+([A-Za-z0-9 _\-\+]+)", (prompt_from_txt or ""), re.I)
                    concept = (m.group(1).strip() if m else "object")
                    hint = labels_hint or DEFAULT_LABELS_HINT
                    seed_prompt = GENERIC_LABEL_PROMPT.format(concept=concept, labels_hint=hint)
                else:
                    seed_prompt = prompt_from_txt

                    
                # === COUNTING wiring: match --count-dir behavior ===
                thing = None
                if counting:
                    # mimic the object extraction used in your counting flows
                    thing = re.sub(r"^[Hh]ow many\s+|\s+are there\??.*$", "", prompt_from_txt or "").strip() or "object"

                base_sys = self._sys_prompt_or_none() or ""
                sys_count = base_sys
                if counting:
                    # COUNTING_PROMPT is a system-scope template in your other flows
                    sys_count = base_sys + "\n" + COUNTING_PROMPT.format(object=thing)
                    
                    
                    
                # Build task-specific system message once
                sys_label = None
                if labeling:
                    # mirror the counting pattern by pushing the template into system scope
                    # (keeps behavior parallel to --mixed-counting)
                    hint = labels_hint or DEFAULT_LABELS_HINT
                    # concept chosen just above; ensure it's set here too:
                    try:
                        concept  # may exist from block above
                    except NameError:
                        concept = "object"
                    sys_label = (base_sys or "") + "\n" + GENERIC_LABEL_PROMPT.format(concept=concept, labels_hint=hint)

                # Background first
                self.set_background_from_pil(img, mode="fit")
                img.save(str(raw_path), quality=95)
                
                self.orig_raw_b64 = self._encode_pil_to_b64(img)
                self.orig_canvas_b64 = self._encode_pil_to_b64(self.base_canvas)


                # Header AFTER grid
                self.all_strokes_svg = self._svg_root_open()
                self.stroke_counter = 0
                self.assitant_history = ""
                self.cur_svg_to_render = "None"
                self._colored_svg = self._svg_root_open()
                self.explanations = []

                if stepwise: #MULTI-TURN MODE
                    self._start_generic_session(seed_prompt)
                    self._prompt_multi_turn = True

                    
                    base_sys = self._sys_prompt_or_none() or ""
                    sys_with_one = (sys_count if counting else sys_label if labeling else base_sys) + "\n" + ONE_STROKE_SYSTEM_GUARD
                    prev_resp_id = None  # threaded only when using GPT-5 (Responses API)
                    
                    self.multi_stroke = False

                    while turns < max_turns:
                        expected_s = self.stroke_counter + 1

                        turns += 1
                        
                        context_xml = self._canon_strokes_context()
                        context_parts = []

                        # (A) Strokes context (skip when --stepwise-vision-only)
                        if not getattr(self, "stepwise_vision_only", False):
                            if context_xml:
                                context_parts.append("\n\n[Context so far]\n" + context_xml)

                        # (B) Explanations context (include when --stepwise-explanations)
                        if getattr(self, "stepwise_explanations", False) and self.explanations:
                            context_parts.append("\n\n[Explanation so far]\n" + "\n".join(self.explanations))

                        user_msg = self.input_prompt + "".join(context_parts)



                        # Native Gemini doesn't use per-stroke stop sequences - it uses </answer> (the default)
                        # and naturally generates one stroke at a time through prefill continuation.
                        # OpenRouter Gemini should behave the same way.
                        from llm_adapters import OpenRouterAdapter
                        
                        # Check if we're using any kind of Gemini
                        is_gemini = isinstance(self.llm, GeminiAdapter)
                        is_openrouter_gemini = (isinstance(self.llm, OpenRouterAdapter) and 
                                               "gemini" in (getattr(self.llm, "model", "") or "").lower())
                        
                        # For any Gemini (native or OpenRouter), use None to get default </answer> stop
                        # For other models (GPT, Claude), use specific </sN> stop sequences
                        if is_gemini or is_openrouter_gemini:
                            stop_seq = None  # Will default to </answer>
                        else:
                            stop_seq = f"</s{expected_s}>"
                        
                        prefill = None if getattr(self, "stepwise_vision_only", False) else self.assitant_history.strip()
                        
                        answer_raw = self.get_response_from_llm(
                            msg=user_msg,
                            system_message=sys_with_one,
                            msg_history=[],
                            init_canvas_str=self._canvas_b64_for_model(),   # updated raster w/ previous overlays
                            seed_mode=self.seed_mode,
                            gen_mode="completion",
                            prefill_msg=prefill,
                            stop_sequences=stop_seq,  # Already None for Gemini, or </sN> for others
                            previous_response_id=prev_resp_id,      # <-- thread GPT-5 reasoning session
                        )

                        # Keep a lightweight record of what we sent this turn (for your JSON)
                        turn_input_dbg = {
                            "system": sys_with_one,
                            "user": user_msg,
                            "overlay_svg": (self.all_strokes_svg + "</svg>") if self.all_strokes_svg else None,
                            "previous_response_id": prev_resp_id,
                        }

                        # Extract text and isolate exactly one <sN> ... </sN>
                        full_text = getattr(self, "_last_assistant_text", None) or self.llm.extract_text(answer_raw)
                        full_text = self._normalize_listish_blocks(full_text or "")
                        m_blk = re.search(rf"(<s{expected_s}>.*?</s{expected_s}>)", full_text or "", re.S)
                        
                        #NOTE: Needed for the gemini one stroke only in multiturn image mode?
                        if not m_blk:
                            # fallback — accept any <sK>…</sK> and retag to expected_s
                            m_blk = re.search(r"(<s\d+>.*?</s\d+>)", full_text or "", re.S)
                        
                        # If the model tries to give the final answer prematurely, skip strokes phase
                        if "<final_answer" in (full_text or ""):
                            svg_chunk = ""   # force the no-stroke path below
                            
                        
                        svg_chunk = m_blk.group(1) if m_blk else ""
                        
                        
                        # --- NEW: extract natural-language explanation outside the <sN>...</sN> block
                        explanation_this_turn = ""
                        if isinstance(full_text, str) and full_text.strip():
                            if m_blk:
                                explanation_this_turn = (full_text[:m_blk.start()] + full_text[m_blk.end():]).strip()
                            else:
                                # no <sN> block found; treat the whole reply as explanation (we'll break on no stroke)
                                explanation_this_turn = full_text.strip()

                            # strip code fences or leftover XML wrappers
                            explanation_this_turn = re.sub(r"```.*?```", "", explanation_this_turn, flags=re.S).strip()

                        # persist if non-empty
                        if explanation_this_turn:
                            self.explanations.append(explanation_this_turn)


                        # Thread response id to the next turn (Responses API)
                        meta = self.llm.response_metadata(getattr(self, "_last_raw_response", None) or answer_raw)
                        #NOTE: try making this NONE to see if GPT is "cheating from seeing text in session even if we don't send it"
                        prev_resp_id = (meta or {}).get("response_id")
                        prev_resp_id = None
                        
                        
                        if not svg_chunk:
                            break

                        # Normalize tag number to expected_s (helps if the model echoes a wrong s#)
                        blk_fixed = self._retag_block_to_no(svg_chunk, expected_s)
                        
                        self.update_history(blk_fixed)

                        # Convert model XML → actual SVG <g>/<path>/<text>…
                        svg = self.parse_model_to_svg(blk_fixed)
                        
                        
                        # keep a human-colored copy in parallel
                        if self.cycle_stroke_colors:
                            color = self._palette[(expected_s-1) % len(self._palette)]
                            colored_group = self._recolor_svg_group(svg, color)
                            self._colored_svg += colored_group


                        # Append to the running SVG and advance our stroke counter
                        self.all_strokes_svg += svg
                        self.stroke_counter = expected_s
                        self.cur_svg_to_render = f"{self.all_strokes_svg}</svg>"

                        step_png = out_root / f"item_{i:05d}_step_{turns:03d}.png"
                        self._composite_svg_on_base(self.cur_svg_to_render, str(step_png))
                        last_step_png = step_png
                        
                        # Make next turn see the updated canvas (PNG)
                        with open(step_png, "rb") as fh:
                            step_bytes = fh.read()
                        self.last_canvas_b64 = base64.b64encode(step_bytes).decode("utf-8")
                        
                        
                        # ---- Human-colored step copy (do NOT update last_canvas_b64)
                        if self.save_colored_steps and self.cycle_stroke_colors:
                            step_png_color = out_root / f"item_{i:05d}_step_{turns:03d}_color.png"
                            colored_svg_full = self._colored_svg + "</svg>"
                            self._render_svg_to_png_no_b64(colored_svg_full, str(step_png_color))


                        
                        # Build the same strokes context we included in the prompt
                        ctx_preview = context_xml[:200] + ("..." if len(context_xml) > 200 else "")
                        ctx_sha1 = self._sha1_bytes(context_xml.encode("utf-8"))


                        # --- NEW: capture the exact input sent to the provider for this turn
                        turn_request = getattr(self, "_last_redacted_request", None)

                        # Capture the raw <sN>…</sN> for this turn from assistant history
                        m = re.search(rf"(<s{expected_s}>.*?</s{expected_s}>)", self.assitant_history, re.S)
                        blk_xml = m.group(1) if m else None

                        # Provider’s raw assistant text snapshot for this turn
                        turn_text = getattr(self, "_last_assistant_text", None)

                        # prove what we sent in (original vs re-render)
                        if getattr(self, "stepwise_send_original_only", False):
                            sent_file = str(raw_path)
                            with open(raw_path, "rb") as _rb:
                                sent_bytes = _rb.read()
                        else:
                            sent_file = str(step_png)
                            sent_bytes = step_bytes

                        turn_trace.append({
                            "turn": turns,
                            "s_no": expected_s,
                            "stroke_xml": blk_xml,
                            "assistant_text": turn_text,
                            "step_png": str(step_png),
                            "request_input": turn_input_dbg,
                            "sent_canvas": {
                                "file": sent_file,
                                "sha1": self._sha1_bytes(sent_bytes),
                                "bytes": len(sent_bytes),
                            },
                            "sent_svg_context": {
                                "sha1": ctx_sha1,
                                "chars": len(context_xml),
                                "preview": ctx_preview,
                            },
                            "stop_sequences": f"</s{expected_s}>",
                            "request_preview": turn_request,
                            "omitted_strokes_context": bool(getattr(self, "stepwise_vision_only", False)),
                            "explanation": explanation_this_turn,
                            "explanations_so_far": ("\n".join(self.explanations) if self.explanations else ""),

                        })



                        delay = getattr(self, "api_delay_sec", 0.0) or 0.0
                        if delay > 0:
                            time.sleep(delay)

                    # 1) Canonical <strokes>…</strokes> for JSON
                    answer_xml = re.sub(r'^.*?<svg.*?>', '<strokes>', self.cur_svg_to_render, flags=re.S)
                    answer_xml = re.sub(r'</svg>\s*$', '</strokes>', answer_xml, flags=re.S)

                    # 2) Save the full SVG directly (don’t re-render strokes; we already drew them stepwise)
                    svg_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(svg_path, "w", encoding="utf-8") as f:
                        f.write(self.cur_svg_to_render)

                    # 3) Save the background “_grid” snapshot for parity with other flows
                    orig_with_grid = png_path.with_name(png_path.stem.replace("_annotated", "_grid") + png_path.suffix)
                    (getattr(self, "base_canvas_clean", self.base_canvas)).save(str(orig_with_grid))

                    # 4) Final annotated PNG = the last visible step if we have one; otherwise composite once
                    if turns > 0 and 'last_step_png' in locals() and last_step_png.exists():
                        import shutil
                        shutil.copyfile(last_step_png, png_path)
                    else:
                        # Nothing drawn: still produce an annotated image from whatever we have
                        self._composite_svg_on_base(self.cur_svg_to_render, str(png_path))
                    
                    
                    # ---- final human-colored PNG (parallel artifact)
                    if self.save_colored_steps and self.cycle_stroke_colors:
                        final_colored_png = png_path.with_name(png_path.stem.replace("_annotated", "_annotated_color") + png_path.suffix)
                        colored_svg_full = self._colored_svg + "</svg>"
                        self._render_svg_to_png_no_b64(colored_svg_full, str(final_colored_png))


                    #  Extract final text answer
                    # 3) ANSWER PHASE (one extra call): ask ONLY for <final_answer>…</final_answer>
                    final_answer_guard = (
                        "\n\n[Answer phase]\n"
                        "All required strokes have been provided.\n"
                        "Now output ONLY the final answer in a single block:\n"
                        "<final_answer> ... </final_answer>\n"
                        "No other text, no strokes. Stop immediately after </final_answer>."
                    )

                    context_xml = self._canon_strokes_context()
                    context_parts = []

                    # (A) Strokes (respect --stepwise-vision-only)
                    if not getattr(self, "stepwise_vision_only", False):
                        if context_xml:
                            context_parts.append("\n\n[Context]\nCurrent strokes so far (machine-readable):\n" + context_xml)

                    # (B) Explanations (respect --stepwise-explanations)
                    if getattr(self, "stepwise_explanations", False) and self.explanations:
                        context_parts.append("\n\n[Explanation so far]\n" + "\n".join(self.explanations))

                    msg = self.input_prompt + "".join(context_parts) + final_answer_guard



                    use_stop = not isinstance(self.llm, GeminiAdapter)
                    
                    base_sys = self._sys_prompt_or_none() or ""
                    sys_with_ans_guard = (sys_count if counting else base_sys) + "\n" + FINAL_ANSWER_SYSTEM_GUARD
                    
                    prefill = None if getattr(self, "stepwise_vision_only", False) else self.assitant_history.strip()

                    answer_phase_raw = self.get_response_from_llm(
                        msg=msg,
                        system_message=sys_with_ans_guard,   # ← enforce answer-only at system level
                        msg_history=[],
                        init_canvas_str=self._canvas_b64_for_model(),
                        seed_mode=self.seed_mode,
                        gen_mode="completion",
                        prefill_msg=prefill,
                        stop_sequences="</final_answer>" if use_stop else None,
                        previous_response_id=prev_resp_id,      # <-- thread GPT-5 reasoning session
)


                    answer_request = getattr(self, "_last_redacted_request", None)
                    final_ans = self._extract_final_answer_any(answer_phase_raw, self.cur_svg_to_render, context_xml)

                    # reflect which image the model actually saw for the answer call
                    if getattr(self, "stepwise_send_original_only", False):
                        sent_file = str(raw_path)
                        with open(raw_path, "rb") as _rb:
                            sent_bytes = _rb.read()
                    else:
                        # when no strokes, we may not have step_bytes; fall back to raw
                        if turns > 0:
                            sent_file = str(out_root / f"item_{i:05d}_step_{turns:03d}.png")
                            sent_bytes = step_bytes
                        else:
                            sent_file = str(raw_path)
                            with open(raw_path, "rb") as _rb:
                                sent_bytes = _rb.read()

                    turn_trace.append({
                        "turn": "final_answer",
                        "s_no": None,
                        "stroke_xml": None,
                        "assistant_text": answer_phase_raw,
                        "final_answer": final_ans,
                        "step_png": str(out_root / f"item_{i:05d}_step_{turns:03d}.png") if turns > 0 else str(raw_path),
                        "sent_canvas": {
                            "file": sent_file,
                            "sha1": self._sha1_bytes(sent_bytes),
                            "bytes": len(sent_bytes),
                        },
                        "sent_svg_context": {
                            "sha1": self._sha1_bytes(context_xml.encode("utf-8")),
                            "chars": len(context_xml),
                            "preview": context_xml[:200] + ("..." if len(context_xml) > 200 else ""),
                        },
                        "stop_sequences": "</final_answer>",
                        "request_preview": answer_request,
                        "omitted_strokes_context": bool(getattr(self, "stepwise_vision_only", False)),
                        "explanations_so_far": ("\n".join(self.explanations) if self.explanations else ""),


                    })


                elif two_turn: #TWO TURN MODE
                    self._start_generic_session(seed_prompt)
                    
                    self._prompt_multi_turn = True

                    base_sys = self._sys_prompt_or_none() or ""

                    # ---------- TURN 1: strokes only ----------
                    sys_turn1 = (sys_count if counting else sys_label if labeling else base_sys) + "\n" + STROKES_ONLY_SYSTEM_GUARD
                    context_block = ""  # none yet
                    user_msg = self.input_prompt + context_block

                    use_stop = not isinstance(self.llm, GeminiAdapter)
                    resp1 = self.get_response_from_llm(
                        msg=user_msg,
                        system_message=sys_turn1,
                        msg_history=[],
                        init_canvas_str=self.last_canvas_b64,
                        seed_mode=self.seed_mode,
                        gen_mode="completion",
                        prefill_msg=self.assitant_history.strip(),
                        stop_sequences="</answer>" if use_stop else None,
                    )
                    # Extract, render
                    answer_xml = self._canon_strokes(resp1)
                    self._render_answer_xml(answer_xml, svg_out=svg_path, png_out=png_path)

                    # Debug capture
                    turn_trace.append({
                        "turn": 1,
                        "phase": "strokes_only",
                        "assistant_text": getattr(self, "_last_assistant_text", None),
                        "request_input": _capture_turn_input(sys_turn1, user_msg, ""),  # no strokes yet
                        "step_png": str(png_path),
                    })

                    # Thread previous_response_id for GPT-5
                    meta1 = self.llm.response_metadata(getattr(self, "_last_raw_response", None)) if hasattr(self, "_last_raw_response") else self.llm.response_metadata(resp1)
                    prev_resp_id = (meta1 or {}).get("response_id")

                    # ---------- TURN 2: final answer only ----------
                    sys_turn2 = (sys_count if counting else sys_label if labeling else base_sys) + "\n" + FINAL_ANSWER_SYSTEM_GUARD
                    context_block2 = f"\n\n[Context so far]\n{answer_xml}"
                    user_msg2 = self.input_prompt + context_block2

                    resp2 = self.get_response_from_llm(
                        msg=user_msg2,
                        system_message=sys_turn2,
                        msg_history=[],
                        init_canvas_str=self.last_canvas_b64,
                        seed_mode=self.seed_mode,
                        gen_mode="completion",
                        prefill_msg=self.assitant_history.strip(),
                        stop_sequences="</final_answer>" if use_stop else None,
                        previous_response_id=prev_resp_id,   # <-- key for GPT-5
                    )
                    final_ans = self._extract_final_answer_any(resp2, answer_xml, self.assitant_history)

                    turn_trace.append({
                        "turn": 2,
                        "phase": "final_answer_only",
                        "assistant_text": getattr(self, "_last_assistant_text", None),
                        "final_answer": final_ans,
                        "request_input": _capture_turn_input(sys_turn2, user_msg2, answer_xml),
                    })
                    
                    
                    

                    # Count strokes from turn 1’s XML
                    total_strokes = len(re.findall(r"(<s\d+>.*?</s\d+>)", answer_xml, re.S))
                    text_strokes  = self._count_strokes(answer_xml, count_only_text=True)

                else:  # ONE TURN MODE
                    self._prompt_multi_turn = False

                    if self.no_system_prompt:
                        use_stop = not isinstance(self.llm, GeminiAdapter)
                        # no system -> prepend counting template into the user msg (like --count-dir raw mode)
                        #format with counting prompt or labeling prompt
                        msg_once = (COUNTING_PROMPT.format(object=thing) + "\n\n" + prompt_from_txt) if counting else prompt_from_txt
                        answer = self.get_response_from_llm(
                            msg=msg_once,
                            system_message=None,
                            msg_history=[],
                            init_canvas_str=self.last_canvas_b64,
                            seed_mode=self.seed_mode,
                            gen_mode="generation",
                            stop_sequences="</answer>" if use_stop else None,
                        )
                    else:
                        self._start_generic_session(prompt_from_txt)  # keep the user prompt identical
                        use_stop = not isinstance(self.llm, GeminiAdapter)
                        # system message gets the counting template when counting=True
                        sys_msg = sys_count if counting else sys_label if labeling else (self._sys_prompt_or_none() or "")
                        answer = self.get_response_from_llm(
                            msg=self.input_prompt,
                            system_message=sys_msg,
                            msg_history=[],
                            init_canvas_str=self.last_canvas_b64,
                            seed_mode=self.seed_mode,
                            gen_mode="completion",
                            prefill_msg=self.assitant_history.strip(),
                            stop_sequences="</answer>" if use_stop else None
                        )

                    
                    
                    answer_xml = self._canon_strokes(answer)
                    self._render_answer_xml(answer_xml, svg_out=svg_path, png_out=png_path)
                    
                    final_ans = self._extract_final_answer_any(answer, answer_xml, self.assitant_history)
                
                
                # Optional: also save an annotated image without the grid background
                if getattr(self, "save_annotated_no_grid", False):
                    import shutil
                    nogrid_png = png_path.with_name(
                        png_path.stem.replace("_annotated", "_annotated_nogrid") + png_path.suffix
                    )

                    if self.no_grid:
                        # No grid was ever present; the annotated image already matches raw coords
                        shutil.copyfile(png_path, nogrid_png)
                    else:
                        # We drew strokes in grid-canvas space; composite onto the raw image by undoing the placement offset
                        # (uses self._raw_image_offset set in set_background_from_pil)
                        self._composite_svg_on_raw_image(self.cur_svg_to_render, img, str(nogrid_png))


                total_strokes = len(re.findall(r"(<s\d+>.*?</s\d+>)", answer_xml, re.S))
                text_strokes  = self._count_strokes(answer_xml, count_only_text=True)
                
                # Save generated images (Gemini image model)
                generated_image_paths = []
                for idx, img_data_url in enumerate(getattr(self, "_last_generated_images", []) or []):
                    if img_data_url.startswith("data:"):
                        header, b64_data = img_data_url.split(",", 1)
                        ext = "png" if "png" in header else "jpeg"
                        img_bytes = base64.b64decode(b64_data)
                        gen_img_path = out_root / f"item_{i:05d}_generated_{idx}.{ext}"
                        with open(gen_img_path, "wb") as f:
                            f.write(img_bytes)
                        generated_image_paths.append(str(gen_img_path))
                        print(f"[IMAGE GEN] Saved: {gen_img_path}")
                
                row = {
                    "index": i, "prompt": prompt_from_txt,
                    "mode": "stepwise" if stepwise else ("two_turn" if two_turn else "single_shot"),
                    "turns": (turns if stepwise else None),
                    "model_output": answer_xml,
                    "answer": final_ans,
                    "num_strokes_total": total_strokes,
                    "num_strokes_text": text_strokes,
                    "raw_image": str(raw_path),
                    "grid_image": str(png_path).replace("_annotated", "_grid"),
                    "annotated_image": str(png_path),
                    "svg": str(svg_path),
                    "source_image": str(img_path),
                    "source_prompt": str(txt_path),
                    "turn_trace": turn_trace,
                    "model_output_full": getattr(self, "_last_assistant_text", None),
                    "provider_debug": getattr(self, "_last_provider_debug", None),
                    "request_preview": getattr(self, "_last_redacted_request", None),
                    
                    "explanations": self.explanations,
                    
                    "grid_config": {
                        "adaptive_grid": self.grid_manager.adaptive_grid,
                        "cell_size": self.grid_manager.cell_size,
                        "res_x": self.grid_manager.res_x,
                        "res_y": self.grid_manager.res_y,
                        "grid_size_px": self.grid_manager.grid_size,  # (width, height)
                    },
                    
                    "cell_pixel_map": self.grid_manager.positions,  # already a dict: {'x1y1': (px, py), ...}
                    "generated_images": generated_image_paths,  # Gemini image model outputs
                    # Reasoning from two-turn image generation (Gemini)
                    "reasoning_image_gen": getattr(self.llm, "_last_image_gen_reasoning_turn1", None),
                    "reasoning_text_answer": getattr(self.llm, "_last_image_gen_reasoning_turn2", None),

                }
                with open(out_root / f"item_{i:05d}.json", "w", encoding="utf-8") as jf:
                    json.dump(row, jf, indent=2)

                results.append({
                    "index": i, "mode": row["mode"],
                    "num_strokes_total": total_strokes, "num_strokes_text": text_strokes,
                })

            except Exception as e:
                # build error record with full context so you can debug formatting/parsing issues
                err = {
                    "index": i,
                    "prompt": prompt_from_txt,
                    "mode": "stepwise" if stepwise else ("two_turn" if two_turn else "single_shot"),
                    "turns": (turns if stepwise else None),
                    "error": str(e),
                    "raw_image": str(raw_path),
                    "grid_image": str(png_path).replace("_annotated", "_grid"),
                    "annotated_image": str(png_path),
                    "svg": str(svg_path),
                    "source_image": str(img_path),
                    "source_prompt": str(txt_path),

                    # NEW: preserve what the model actually returned (even if it was malformed)
                    "model_output_full": getattr(self, "_last_assistant_text", None),
                    "provider_debug": getattr(self, "_last_provider_debug", None),
                    "request_preview": getattr(self, "_last_redacted_request", None),

                    # If we collected any turns before crashing, keep that too
                    "turn_trace": locals().get("turn_trace", []),
                }

                # Optional but handy: write a sidecar raw dump for fast inspection
                try:
                    raw_txt = getattr(self, "_last_assistant_text", None)
                    if raw_txt:
                        with open(out_root / f"item_{i:05d}_raw.txt", "w", encoding="utf-8") as tf:
                            tf.write(raw_txt)
                except Exception:
                    pass

                with open(out_root / f"item_{i:05d}.json", "w", encoding="utf-8") as jf:
                    json.dump(err, jf, indent=2)
                results.append(err)


        with open(out_root / "results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary = {
            "folder": str(src), "timestamp": ts, "total_items": len(images),
            "processed": len(results), "out_root": str(out_root),
            "mode": "stepwise" if stepwise else "single_shot",
            "notes": "Prompts are taken verbatim from paired .txt files.",
        }
        with open(out_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved to: {out_root}")
        print(f"Processed: {len(results)}")
        return summary


    



    def run(self, hostname, ip_address):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', 0))
        port = sock.getsockname()[1]
        sock.close()
        print(f'Server running at: http://{ip_address}:{port}')
        self.app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)


# =========================
# Entrypoint
# =========================


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SketchAgent (Claude, GPT, Gemini, or Qwen3)")
    parser.add_argument("--llm", choices=["claude", "gpt", "gemini", "qwen3", "openrouter"], required=True, help="Which provider to use.")
    parser.add_argument("--model", type=str, default=None, help="Model id (e.g., claude-3-5-sonnet-20240620, o3, or gemini-2.5-pro).")
    parser.add_argument("--deterministic", action="store_true", help="Set temperature=0 and top_k=1 (if supported).")
    parser.add_argument("--max-tokens", type=int, default=3000)
    
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="results/hf_eval")
    parser.add_argument("--count-only-text", action="store_true",
                        help="Count only strokes that contain a <text> tag")
    parser.add_argument(
        "--api-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep before every model API call (useful for Gemini rate stability)."
    )
    parser.add_argument("--label-dir", type=str, help="Directory of images to label (e.g., datasets/labeling)")
    parser.add_argument("--labels-hint", type=str, default=None, help="Comma-separated hints for part names")
    parser.add_argument("--concept-mode", choices=["filename", "constant"], default="filename",
                        help="How to set <concept>: from filename or a constant string")
    parser.add_argument("--concept", type=str, default="object",
                        help="Used when --concept-mode=constant")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--label-outdir", type=str, default="results/labeling",
                        help="Root folder for labeling outputs")
    
    parser.add_argument("--count-dir", type=str,
                        help="Folder with paired image + .txt prompt files (e.g., datasets/biased)")
    parser.add_argument("--count-outdir", type=str, default="results/biased_eval",
                        help="Output root for folder-based counting eval")
    
    parser.add_argument("--count-stepwise-dir", type=str,
                        help="Folder (e.g., datasets/biased) for stepwise counting, one stroke per turn")
    parser.add_argument("--count-stepwise-outdir", type=str, default="results/biased_eval_stepwise",
                        help="Output root for stepwise folder counting")
    parser.add_argument("--count-stepwise-max-turns", type=int, default=40,
                        help="Max turns (strokes) per image in stepwise mode")
    parser.add_argument("--eval-stepwise", action="store_true",
                    help="Run CountBench in stepwise mode (one stroke per turn)")

    parser.add_argument("--mixed-dir", type=str,
                        help="Folder with paired image + .txt prompt files (e.g., datasets/mix)")
    parser.add_argument("--mixed-outdir", type=str, default="results/mix_eval",
                        help="Output root for mixed folder eval")
    parser.add_argument("--mixed-stepwise", action="store_true",
                        help="Run one stroke per turn (stepwise). Otherwise single-shot.")
    parser.add_argument("--mixed-max-turns", type=int, default=40,
                        help="Max turns (strokes) per image in stepwise mode")
    parser.add_argument("--vg-root", type=str, default="data",
        help="Root folder containing VG_100K/ and VG_100K_2/ (default: data)")

    # --- RAW (grid-free) baselines ---

    parser.add_argument("--raw-vg-root", type=str, default="data",
        help="Root for VG_100K / VG_100K_2 for raw TallyQA.")
    parser.add_argument("--raw-mix-dir", type=str,
        help="Folder with paired image+.txt prompts for raw mixed eval.")
    parser.add_argument("--raw-outdir", type=str, default="results/raw_eval",
        help="Output root for raw baselines.")
    parser.add_argument("--raw-max-examples", type=int, default=None,
        help="Limit examples for raw baselines.")
    parser.add_argument("--label-stepwise", action="store_true",
                    help="Run labeling one stroke per turn (stepwise)")
    parser.add_argument("--label-max-turns", type=int, default=40,
                    help="Max turns for stepwise labeling")
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="OpenAI GPT-5 reasoning effort (optional)."
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N items (numbering preserved in output filenames).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of indices and/or ranges to process (e.g. '7,12,42-45'). Overrides --skip.",
    )

    # --- NEW adaptive grid controls ---
    parser.add_argument("--adaptive-grid", action="store_true",
                   help="Auto-scale grid cell size based on image resolution.")
    parser.add_argument("--target-cols", type=int, default=50,
                   help="Desired max columns when --adaptive-grid is on.")
    parser.add_argument("--target-rows", type=int, default=50,
                   help="Desired max rows when --adaptive-grid is on.")
    parser.add_argument("--min-cell-px", type=int, default=18,
                   help="Minimum pixel size per cell for readability.")
    parser.add_argument("--max-cell-px", type=int, default=64,
                   help="Upper bound for pixel size per cell.")
    
    parser.add_argument("--no-grid", action="store_true",
                       help="Do NOT draw/overlay the grid; send the raw image only.")

    # --- NEW: fixed coordinate-space size (useful for no-grid coord trials) ---
    # Example: --no-grid --res-x 1000 --res-y 1000
    # When used together with --no-grid, the input image is resized to (res_x,res_y)
    # and model points in <points> can be numeric coordinates like (500,105).
    parser.add_argument("--res-x", type=int, default=None,
                        help="Override coordinate/canvas width (e.g., 1000).")
    parser.add_argument("--res-y", type=int, default=None,
                        help="Override coordinate/canvas height (e.g., 1000).")
    
    parser.add_argument("--no-system-prompt", action="store_true",
                       help="Do NOT send the system prompt or task prompt. The model only receives the per-sample text file.")
    parser.add_argument("--two-turn", action="store_true",
        help="Turn 1: output all strokes only. Turn 2: output final answer only (for GPT-5 use previous_response_id).")
    parser.add_argument("--mixed-counting", action="store_true",
        help="When set, prepend a counting header and treat mixed items as counting tasks.")
    parser.add_argument("--mixed-labeling", action="store_true",
    help="When set, use the labeling template in mixed eval (like --mixed-counting).")
    parser.add_argument("--cycle-stroke-colors", action="store_true",
                        help="Cycle stroke hues per turn (human-view only: red→orange→yellow→green→blue→purple).")
    parser.add_argument("--save-colored-steps", action="store_true",
                        help="If set, save *_step_XXX_color.png and *_annotated_color.png with cycled colors.")
    parser.add_argument("--stepwise-original-only", action="store_true",
    help="Stepwise mode: always send the original image back to the model each turn (no re-rendered overlays).")
    parser.add_argument(
    "--stepwise-vision-only",
    action="store_true",
    help="Stepwise: send only the updated image back to the model (no text stroke history)."
)
    parser.add_argument(
    "--stepwise-explanations",
    action="store_true",
    help="In stepwise mode, capture a natural-language explanation each turn and send it back. "
         "Works with --stepwise-vision-only (sends explanation but not text strokes)."
)
    parser.add_argument(
    "--prompt-origin",
    choices=["bottom_left", "top_left"],
    default="bottom_left",
    help="Select how the system prompt describes the origin for xNyM tokens.",
)
    parser.add_argument("--save-annotated-no-grid", action="store_true",
    help="Also save an annotated image without the grid background (in addition to the normal annotated image).")




    args = parser.parse_args()
    
    # default model per provider if none given
    if not args.model:
        if args.llm == "claude":
            args.model = "claude-3-5-sonnet-20240620"
        elif args.llm == "gpt":
            args.model = "o3"
        elif args.llm == "gemini":
            args.model = "gemini-2.5-pro"  # set whatever Gemini model string you want here
            
    

    adapter = make_adapter(args.llm, args.model, cache=False, max_tokens=args.max_tokens)

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print(f'Server running at: http://{ip_address}:5000')

    user_always_first = False
    res = 50
    cell_size = 15
    grid_size = (765, 765)  # 50 * 15 + 15 for header
    stroke_width = cell_size * 0.6

    app = SketchApp(
        res=res,
        cell_size=cell_size,
        grid_size=grid_size,
        stroke_width=stroke_width,
        target_concept="sailboat",
        user_always_first=user_always_first,
        llm_adapter=adapter,
        dynamic_grid=True,
        min_grid=10,
        max_grid=100,
        # NEW: adaptive grid params
        adaptive_grid=args.adaptive_grid,
        target_cols=args.target_cols,
        target_rows=args.target_rows,
        min_cell_px=args.min_cell_px,
        max_cell_px=args.max_cell_px,
        no_grid=args.no_grid,
        no_system_prompt=args.no_system_prompt,
        prompt_origin=args.prompt_origin,
        res_x=args.res_x,
        res_y=args.res_y,
    )

    app.api_delay_sec = args.api_delay
    app.reasoning_effort = args.reasoning_effort

    if args.deterministic:
        app.seed_mode = "deterministic"
        
        
    app.cycle_stroke_colors = args.cycle_stroke_colors
    app.save_colored_steps  = args.save_colored_steps

    app.stepwise_send_original_only = args.stepwise_original_only
    app.stepwise_vision_only = args.stepwise_vision_only

    app.stepwise_explanations = args.stepwise_explanations
    
    app.save_annotated_no_grid = args.save_annotated_no_grid


        
    
    # If --mixed-dir is provided, run mixed folder eval and exit
    if args.mixed_dir:
        app.evaluate_mixed_folder(
            src_dir=args.mixed_dir,
            outdir=args.mixed_outdir,
            stepwise=args.mixed_stepwise,
            max_images=args.max_examples,
            max_turns=args.mixed_max_turns,
            count_only_text=args.count_only_text,
            only=args.only,
            skip=args.skip,
            two_turn=args.two_turn,
            counting=args.mixed_counting,
            labeling=args.mixed_labeling,
            labels_hint=args.labels_hint,
        )
        
        raise SystemExit(0)


    app.run(hostname, ip_address)
