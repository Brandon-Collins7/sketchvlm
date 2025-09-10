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


from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

import utils
from prompts import sketch_first_prompt, system_prompt, gt_example, GENERIC_LABEL_PROMPT, DEFAULT_LABELS_HINT, COUNTING_PROMPT

from PIL import Image, ImageOps

# batch eval / datasets
from pathlib import Path
from datasets import load_dataset  # pip install datasets
try:
    from tqdm import tqdm  # optional, pretty progress bar
except Exception:
    tqdm = lambda x, **k: x


# =========================
# LLM Adapters
# =========================
class BaseLLMAdapter:
    """
    Minimal interface the app relies on:
      - build_user_content(init_canvas_b64, text) -> provider-specific "content" array
      - call(system_message, messages, additional_args) -> raw response
      - extract_text(raw_response) -> str content
      - request_has_image(messages) -> bool
    """
    def __init__(self, model: str, cache: bool = False, max_tokens: int = 3000):
        self.model = model
        self.cache = cache
        self.max_tokens = max_tokens

    def build_user_content(self, init_canvas_b64: Optional[str], text: str):
        raise NotImplementedError

    def call(self, system_message, messages, additional_args):
        raise NotImplementedError

    def extract_text(self, raw_response) -> str:
        raise NotImplementedError

    def request_has_image(self, messages: List[Dict]) -> bool:
        """Return True iff any message uses an image (image or image_url)."""
        for m in messages:
            for part in m.get("content", []):
                t = part.get("type")
                if t in ("image", "image_url"):
                    return True
        return False
    
    def response_metadata(self, raw_response) -> dict:
        """Provider-specific metadata (finish_reason, usage, safety, etc.)."""
        return {}

    def debug_dump(self, raw_response) -> dict:
        """Small, JSON-serializable snapshot of the raw provider response."""
        try:
            return {"repr": repr(raw_response)[:8000]}
        except Exception:
            return {}

# --- at top of file (with other imports) ---
import base64, re
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# =========================
# Gemini Adapter (2.5 Pro/Flash)
# =========================
class GeminiAdapter(BaseLLMAdapter):
    """
    Works with Gemini 2.5 Pro (and Flash). Accepts your existing 'messages'
    shape (OpenAI/Claude style) and converts to Gemini {role, parts} format.
    If the response has no .text, we aggregate all text parts.
    """

    def __init__(self, model: str, cache: bool = False, max_tokens: int = 8192):
        super().__init__(model, cache, max_tokens)
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = None
        self._last_response = None
        self._last_sys = None

        # permissive safety so evals don't silently block
        self.safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT,         "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,        "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,             "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,  "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]

    # ---------- helpers ----------
    def _ensure_model(self, system_message: str):
        if self._model is None or self._last_sys != system_message:
            self._model = self._genai.GenerativeModel(
                self.model,
                system_instruction=system_message,
                safety_settings=self.safety_settings
            )
            self._last_sys = system_message

    @staticmethod
    def _decode_data_url(url: str):
        """
        Accepts data URLs like 'data:image/png;base64,AAAA...'
        Returns (mime_type:str, raw_bytes:bytes) or (None, None) if not a data URL.
        """
        m = re.match(r"^data:(.*?);base64,(.+)$", url)
        if not m:
            return None, None
        mime = m.group(1)
        raw = base64.b64decode(m.group(2))
        return mime, raw

    def _parts_from_our_content(self, content):
        """
        Convert your OpenAI/Claude-style 'content' list into Gemini 'parts'.
        """
        parts = []
        for item in content or []:
            # If a stray string sneaks in, treat it as text
            if isinstance(item, str):
                parts.append({"text": item})
                continue

            t = item.get("type")
            if t == "text":
                parts.append({"text": item.get("text", "")})

            elif t == "image":
                src = item.get("source", {})
                if src.get("type") == "base64":
                    mt = src.get("media_type", "image/png")
                    data_b = base64.b64decode(src.get("data", "") or b"")
                    parts.append({"inline_data": {"mime_type": mt, "data": data_b}})

            elif t == "image_url":
                # Expected to be a *data URL* for local canvas
                url = (item.get("image_url") or {}).get("url", "")
                mt, data_b = self._decode_data_url(url)
                if mt and data_b:
                    parts.append({"inline_data": {"mime_type": mt, "data": data_b}})

            # ignore unknown types quietly
        return parts

    def _to_gemini_messages(self, messages):
        """
        Turn your messages into a list of {role, parts}.
        'assistant' -> 'model' role for Gemini.
        """
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            # if content is already a string, make it a single text part
            if isinstance(content, str):
                parts = [{"text": content}]
            else:
                parts = self._parts_from_our_content(content)

            g_role = "model" if role == "assistant" else "user"
            out.append({"role": g_role, "parts": parts})
        return out

    # ---------- interface ----------
    def build_user_content(self, init_canvas_b64: Optional[str], text: str):
        """
        Keep your app's contract: returns a list of parts in OpenAI/Claude shape.
        """
        content = []
        if init_canvas_b64:
            mime = "image/jpeg" if init_canvas_b64.startswith("/9j") else "image/png"
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": init_canvas_b64}
            })
        content.append({"type": "text", "text": text})
        return content

    def call(self, system_message, messages, additional_args):
        self._ensure_model(system_message)

        gen_cfg = {"max_output_tokens": self.max_tokens}
        if "temperature" in additional_args:
            gen_cfg["temperature"] = additional_args["temperature"]
        # Gemini supports stop_sequences in generation_config
        ss = additional_args.get("stop_sequences")
        if ss:
            gen_cfg["stop_sequences"] = ss if isinstance(ss, list) else [ss]

        # Prefer structured {role, parts}. If something looks off, fallback to flat text.
        try:
            contents = self._to_gemini_messages(messages)
            resp = self._model.generate_content(contents=contents, generation_config=gen_cfg)
        except Exception:
            # Fallback: flatten all text into a single prompt string
            text_blocks = []
            for m in messages:
                c = m.get("content")
                if isinstance(c, str):
                    text_blocks.append(c)
                elif isinstance(c, list):
                    for p in c:
                        if isinstance(p, dict) and p.get("type") == "text":
                            text_blocks.append(p.get("text", ""))
            prompt = "\n\n".join(x for x in text_blocks if x)
            resp = self._model.generate_content(prompt, generation_config=gen_cfg)

        self._last_response = resp
        return resp

    def extract_text(self, raw_response) -> str:
        # 1) try the convenience accessor
        try:
            if raw_response.text:
                return raw_response.text
        except Exception:
            pass

        # 2) aggregate parts
        try:
            chunks = []
            for cand in getattr(raw_response, "candidates", []) or []:
                cnt = getattr(cand, "content", None)
                for part in getattr(cnt, "parts", []) or []:
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str) and txt:
                        chunks.append(txt)
            return "".join(chunks)
        except Exception:
            return ""

    def request_has_image(self, messages: List[Dict]) -> bool:
        for m in messages:
            for part in m.get("content", []):
                t = isinstance(part, dict) and part.get("type")
                if t in ("image", "image_url"):
                    return True
        return False

    def debug_dump(self, raw_response) -> dict:
        out = {
            "provider": "gemini",
            "model": self.model,
            "combined_text": "",
            "finish_reason": None,
            "has_candidates": False,
            "prompt_block_reason": None,
            "safety": [],
        }
        try:
            out["combined_text"] = self.extract_text(raw_response) or ""
            cands = getattr(raw_response, "candidates", None)
            out["has_candidates"] = bool(cands)
            if cands:
                fr = getattr(cands[0], "finish_reason", None)
                out["finish_reason"] = str(fr)
                sr = getattr(cands[0], "safety_ratings", None)
                if sr:
                    out["safety"] = [str(x) for x in sr]
            pf = getattr(raw_response, "prompt_feedback", None)
            if pf:
                out["prompt_block_reason"] = str(getattr(pf, "block_reason", None))
        except Exception as e:
            out["debug_error"] = str(e)
        return out



class ClaudeAdapter(BaseLLMAdapter):
    """
    Uses anthropic.Anthropic messages API.
    - Images sent as base64 parts with media_type detection.
    - stop_sequences -> stop_sequences (as list)
    """
    def __init__(self, model: str, cache: bool = False, max_tokens: int = 3000):
        super().__init__(model, cache, max_tokens)
        import anthropic
        load_dotenv()
        key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key)
        self._anthropic = anthropic

    def build_user_content(self, init_canvas_b64: Optional[str], text: str):
        content: List[Dict] = []
        if init_canvas_b64:
            media_type = "image/jpeg" if init_canvas_b64[:3] == "/9j" else "image/png"
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": init_canvas_b64
                }
            })
        content.append({"type": "text", "text": text})
        if self.cache:
            content[-1]["cache_control"] = {"type": "ephemeral"}
        return content

    def call(self, system_message, messages, additional_args):
        args = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_message,
            messages=messages,
        )
        if "stop_sequences" in additional_args:
            ss = additional_args["stop_sequences"]
            args["stop_sequences"] = ss if isinstance(ss, list) else [ss]
        if "temperature" in additional_args:
            args["temperature"] = additional_args["temperature"]
        if "top_k" in additional_args:
            args["top_k"] = additional_args["top_k"]

        if self.cache:
            return self.client.beta.prompt_caching.messages.create(**args)
        return self.client.messages.create(**args)

    def extract_text(self, raw_response) -> str:
        return raw_response.content[0].text


class OpenAIAdapter(BaseLLMAdapter):
    """
    Uses openai.chat.completions API.
    - Images must be sent as image_url (data URI) in message content.
    - stop_sequences -> stop (but we skip 'stop' if images are present).
    """
    def __init__(self, model: str, cache: bool = False, max_tokens: int = 3000):
        super().__init__(model, cache, max_tokens)
        import openai
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self._client = openai.OpenAI()

    def build_user_content(self, init_canvas_b64: Optional[str], text: str):
        content: List[Dict] = []
        if init_canvas_b64:
            hdr = "data:image/jpeg;base64," if init_canvas_b64[:3] == "/9j" else "data:image/png;base64,"
            content.append({
                "type": "image_url",
                "image_url": {"url": hdr + init_canvas_b64}
            })
        content.append({"type": "text", "text": text})
        return content

    def call(self, system_message, messages, additional_args):
        args = dict(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_tokens,
        )
        if "temperature" in additional_args:
            args["temperature"] = additional_args["temperature"]
        if "stop_sequences" in additional_args and not self.request_has_image(messages):
            ss = additional_args["stop_sequences"]
            args["stop"] = ss if isinstance(ss, list) else [ss]
        return self._client.chat.completions.create(**args)

    def extract_text(self, raw_response) -> str:
        return raw_response.choices[0].message.content


# =========================
# Flask App
# =========================
class SketchApp:
    def __init__(
        self, res, cell_size, grid_size, stroke_width, target_concept,
        user_always_first, llm_adapter: BaseLLMAdapter, show_full_grid: bool = False
    ):
        self.app = Flask(__name__)
        self.session_id = str(uuid.uuid4())

        # LLM Setup
        self.seed_mode = "stochastic"
        self.cache = llm_adapter.cache
        self.max_tokens = llm_adapter.max_tokens
        self.llm = llm_adapter

        # Grid setup
        self.res = res
        self.num_cells = res
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.show_full_grid = show_full_grid
        self.multi_stroke = True
        self.init_canvas_grid, self.positions = utils.create_grid_image(
            res=res, cell_size=cell_size, header_size=cell_size, full=self.show_full_grid
        )
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

    def _composite_svg_on_base(self, svg_text: str, out_png: str):
        png = cairosvg.svg2png(bytestring=svg_text.encode())
        over = Image.open(io.BytesIO(png)).convert("RGBA")
        self.base_canvas = self.base_canvas.convert("RGBA")
        self.base_canvas.alpha_composite(over)
        self.base_canvas.convert("RGB").save(out_png)
        self._update_canvas_b64(out_png)

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



    # ---------- routes ----------
    def toggle_grid(self):
        self.show_full_grid = not self.show_full_grid
        self.init_canvas_grid, self.positions = utils.create_grid_image(
            res=self.res, cell_size=self.cell_size, header_size=self.cell_size, full=self.show_full_grid
        )
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

        # choose "fit" (letterbox) or "fill" (cover); "fit" avoids cropping
        placed = self._fit_image_to_canvas(img, mode="fit", bgcolor=(255, 255, 255))
        composited = self._overlay_grid(placed)
        self._set_base_canvas(composited)

        return jsonify({"status": "success", "filename": fname})


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
        self.all_strokes_svg = f'<svg width="{self.grid_size[0]}" height="{self.grid_size[1]}" xmlns="http://www.w3.org/2000/svg">'
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
        self.all_strokes_svg = f'<svg width="{self.grid_size[0]}" height="{self.grid_size[1]}" xmlns="http://www.w3.org/2000/svg">'
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
        gen_mode: str = "generation"
    ):
        if msg_history is None:
            msg_history = []

        additional_args: Dict = {}
        if seed_mode == "deterministic":
            additional_args["temperature"] = 0.0
            additional_args["top_k"] = 1  # ignored by OpenAI adapter

        other_msg = self.define_input_to_llm(msg_history, init_canvas_str, msg)

        if gen_mode == "completion" and prefill_msg:
            other_msg = other_msg + [{"role": "assistant", "content": f"{prefill_msg}"}]

        additional_args["stop_sequences"] = stop_sequences if stop_sequences else "</answer>"
        
        # Gemini tends to prematurely STOP on XML-ish stop strings; disable for completion.
        '''
        if isinstance(self.llm, GeminiAdapter) and gen_mode == "completion":
            additional_args.pop("stop_sequences", None)
        '''
        # Do not use stop-sequences with Gemini (causes premature STOP and empties)
        if isinstance(self.llm, GeminiAdapter) and "stop_sequences" in additional_args:
            additional_args.pop("stop_sequences", None)
        
        
        # optional tiny throttle (helps Gemini avoid empty outputs)
        delay = getattr(self, "api_delay_sec", 0.0) or 0.0
        if delay > 0:
            time.sleep(delay)

        # ---- call provider ----
        response = self.call_llm(system_message, other_msg, additional_args)
        content  = self.llm.extract_text(response)

        if gen_mode == "completion" and prefill_msg:
            other_msg = other_msg[:-1]
            content = f"{prefill_msg}{content}"

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
            system_message=system_prompt.format(res=self.res),
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
            system_message=system_prompt.format(res=self.res),
            msg_history=[],
            init_canvas_str=self.last_canvas_b64,
            seed_mode=self.seed_mode,
            gen_mode="generation",
            stop_sequences="</answer>"
        )
        strokes_list_str, t_values_str = utils.parse_xml_string(all_sketch, res=self.res)
        strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(t_values_str)
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
            y = min(self.grid_size[0] - 1 - self.cell_size, max(0, y))
            grid_x = int(x // self.cell_size)
            grid_y = int(self.num_cells - (y // self.cell_size))
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
                y = min(self.grid_size[0] - 1 - self.cell_size, max(0, y))
                grid_x = int(x // self.cell_size)
                grid_y = int(self.num_cells - (y // self.cell_size))
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

    def parse_model_to_svg(self, stroke_model: str):
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


        # ----- default: curve/path stroke (unchanged) -----
        strokes_list_str, t_values_str = utils.parse_xml_string_single_stroke(
            stroke_model, self.res, stroke_no
        )
        strokes_list = ast.literal_eval(strokes_list_str)
        t_values     = ast.literal_eval(t_values_str)

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
        answer = self.get_response_from_llm(
            msg=self.input_prompt,
            system_message=system_prompt.format(res=self.res),
            msg_history=[],
            init_canvas_str=self.last_canvas_b64,
            seed_mode=self.seed_mode,
            gen_mode="completion",
            prefill_msg=self.assitant_history.strip(),
            stop_sequences="</answer>"
        )

        if self.multi_stroke:
            # accept all unseen blocks
            return self._next_undrawn(answer)
        else:
            # accept only the exact next s-number (e.g., s3 if we’re at s2)
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
        Count predicted strokes:
        - if count_only_text=True → count only <s*> blocks that contain <text>...</text>
        - otherwise → count all <s*> blocks inside <strokes>…</strokes>
        """
        blocks = re.findall(r"(<s\d+>.*?</s\d+>)", answer_xml, re.S)
        if count_only_text:
            blocks = [b for b in blocks if re.search(r"<text>\s*(?:'[^']*'|\"[^\"]*\")\s*</text>", b, re.S)]
        return len(blocks)

    def _render_answer_xml(self, answer_xml: str, svg_out: Path, png_out: Path):
        """
        Take a full <strokes>...</strokes> answer and render *all* strokes to SVG+PNG.
        """
        # reset per-sample drawing state
        self.all_strokes_svg = (
            f'<svg width="{self.grid_size[0]}" height="{self.grid_size[1]}" '
            f'xmlns="http://www.w3.org/2000/svg">'
        )
        self.stroke_counter = 0
        self.cur_svg_to_render = "None"

        # extract blocks in order
        blocks = re.findall(r"(<s\d+>.*?</s\d+>)", answer_xml, re.S)

        # ── save the (grid + photo) background *before* compositing strokes
        png_out.parent.mkdir(parents=True, exist_ok=True)
        orig_with_grid = png_out.with_name(png_out.stem.replace("_annotated", "_grid") + png_out.suffix)
        self.base_canvas.save(str(orig_with_grid))

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
        Place `img` onto a canvas of size self.grid_size without distortion.

        mode:
        - "fit"   : letterbox (contain) — keep all of the image, pad with bgcolor
        - "fill"  : cover — fill canvas, center-crop the overflow
        - "stretch": old behavior (distort to exactly grid size)
        - "center": no scaling, just center with possible cropping outside canvas
        """
        img = ImageOps.exif_transpose(img)  # respect EXIF orientation
        W, H = self.grid_size
        iw, ih = img.size

        if mode == "stretch":
            return img.resize((W, H), Image.LANCZOS).convert("RGB")

        if mode == "center":
            canvas = Image.new("RGB", (W, H), bgcolor)
            x = (W - iw) // 2
            y = (H - ih) // 2
            canvas.paste(img.convert("RGB"), (x, y))
            return canvas

        if mode == "fill":
            s = max(W / iw, H / ih)          # cover
            nw, nh = int(round(iw * s)), int(round(ih * s))
            im = img.resize((nw, nh), Image.LANCZOS)
            x0 = (nw - W) // 2
            y0 = (nh - H) // 2
            return im.crop((x0, y0, x0 + W, y0 + H)).convert("RGB")

        # default "fit" (letterbox)
        s = min(W / iw, H / ih)              # contain
        nw, nh = int(round(iw * s)), int(round(ih * s))
        im = img.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), bgcolor)
        x = (W - nw) // 2
        y = (H - nh) // 2
        canvas.paste(im.convert("RGB"), (x, y))
        return canvas


    def _overlay_grid(self, img_rgb: Image.Image) -> Image.Image:
        """Overlay the current grid on top of `img_rgb` (no distortion)."""
        grid = self.init_canvas_grid.convert("RGBA")
        base = img_rgb.convert("RGBA")
        # Keep only the grid lines (white/transparent background)
        mask = grid.convert("L").point(lambda p: 255 if p < 200 else 0)
        base.paste(grid, (0, 0), mask)
        return base.convert("RGB")


    def _set_base_canvas(self, img_rgb: Image.Image):
        """Persist as the new background everywhere + refresh b64 copy."""
        self.base_canvas = img_rgb.convert("RGB")
        self.base_canvas.save("static/init_canvas.png")
        self.base_canvas.save("static/cur_canvas_user.png")
        self.base_canvas.save("static/cur_canvas_agent.png")
        self._update_canvas_b64("static/init_canvas.png")


    def set_background_from_pil(self, pil_img: Image.Image, mode: str = "fit", bgcolor=(255, 255, 255)):
        """Public helper used by batch eval to mimic an uploaded image."""
        placed = self._fit_image_to_canvas(pil_img.convert("RGB"), mode=mode, bgcolor=bgcolor)
        composited = self._overlay_grid(placed)
        self._set_base_canvas(composited)
        
        
    def _extract_label_legend(self, answer_xml: str):
        legend = []
        for blk in re.findall(r"(<s\d+>.*?</s\d+>)", answer_xml, re.S):
            m_text = re.search(r"<text[^>]*>\s*'([^']+)'\s*</text>", blk, re.S)
            m_id   = re.search(r"<id>\s*(.*?)\s*</id>", blk, re.S)
            if m_text:
                label = m_text.group(1).strip()
                font_px, color = self._parse_text_style(blk)
                legend.append({
                    "text": label,
                    "id": (m_id.group(1).strip() if m_id else label),
                    "font_px": (int(round(font_px)) if font_px else None),
                    "color": color
                })
        return legend


        
    def label_parts(self, concept: str, labels_hint: str = None):
        """
        Ask the LLM to emit grid-anchored text labels for visible parts.
        Returns the raw XML and writes labels.svg / labels_annotated.png.
        """
        hint = labels_hint or DEFAULT_LABELS_HINT
        prompt = GENERIC_LABEL_PROMPT.format(concept=concept, labels_hint=hint)

        answer = self.get_response_from_llm(
            msg=prompt,
            system_message=system_prompt.format(res=self.res),
            msg_history=[],
            init_canvas_str=self.last_canvas_b64,
            seed_mode=self.seed_mode,
            gen_mode="generation",
            stop_sequences=None,  # important for Gemini stability
        )

        out_root = Path(self.path2save)
        svg_path = out_root / "labels.svg"
        png_path = out_root / "labels_annotated.png"
        self._render_answer_xml(answer, svg_out=svg_path, png_out=png_path)
        return answer, str(svg_path), str(png_path)
    
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

    
    def evaluate_labeling_folder(
        self,
        src_dir: str = "datasets/labeling",
        outdir: str = "results/labeling",
        concept_mode: str = "filename",     # "filename" or "constant"
        constant_concept: str = "object",   # used if concept_mode="constant"
        labels_hint: str = None,
        max_images: int = None,
    ):
        """
        Batch-label all images under `src_dir` (non-recursive).
        Saves per-item: *_orig.jpg, *_grid.png, *_annotated.png, *.svg, *.json
        Also writes results.jsonl and summary.json in out root.
        """
        src = Path(src_dir)
        assert src.exists() and src.is_dir(), f"Folder not found: {src_dir}"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(outdir) / ts
        out_root.mkdir(parents=True, exist_ok=True)

        # collect images
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        items = [p for p in sorted(src.iterdir()) if p.suffix.lower() in exts]
        if max_images is not None:
            items = items[:max_images]

        results = []
        total = 0
        total_labels = 0

        pbar = tqdm(items, desc="Labeling images", unit="img") if hasattr(tqdm, "__call__") else items

        for idx, path in enumerate(pbar):
            try:
                # 1) load + set background (keeps aspect with letterbox padding)
                img = Image.open(path).convert("RGB")
                self.set_background_from_pil(img, mode="fit")

                # 2) decide concept (for the <concept> line in the prompt/output)
                if concept_mode == "constant":
                    concept = constant_concept.strip()
                else:
                    # derive from filename: strip extension, swap _/- for spaces
                    stem = path.stem
                    concept = re.sub(r"[_\-]+", " ", stem).strip() or "object"

                # 3) build prompt (grid-only text labels)
                hint = labels_hint or DEFAULT_LABELS_HINT
                prompt = GENERIC_LABEL_PROMPT.format(concept=concept, labels_hint=hint)

                # 4) call the model (no stop_sequences for Gemini)
                answer = self.get_response_from_llm(
                    msg=prompt,
                    system_message=system_prompt.format(res=self.res),
                    msg_history=[],
                    init_canvas_str=self.last_canvas_b64,
                    seed_mode=self.seed_mode,
                    gen_mode="generation",
                    stop_sequences=None,
                )

                # 5) render SVG + annotated PNG
                raw_path = out_root / f"item_{idx:05d}_orig.jpg"
                img.save(str(raw_path), quality=95)

                svg_path = out_root / f"item_{idx:05d}.svg"
                png_path = out_root / f"item_{idx:05d}_annotated.png"
                self._render_answer_xml(answer, svg_out=svg_path, png_out=png_path)

                # 6) extract labels + counts
                legend = self._extract_label_legend(answer)
                n_labels = len(legend)
                total_labels += n_labels

                # 7) per-item JSON
                row = {
                    "index": idx,
                    "filename": str(path),
                    "concept": concept,
                    "labels_hint": hint,
                    "model_output": answer,
                    "labels": legend,                     # [{"text": "...", "id": "..."}]
                    "num_labels": n_labels,
                    "raw_image": str(raw_path),
                    "grid_image": str(png_path).replace("_annotated", "_grid"),
                    "annotated_image": str(png_path),
                    "svg": str(svg_path),
                }
                with open(out_root / f"item_{idx:05d}.json", "w", encoding="utf-8") as jf:
                    json.dump(row, jf, indent=2)
                results.append(row)
                total += 1

                # live progress bar info
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(labels=f"{total_labels} total")

            except Exception as e:
                total += 1
                err = {"index": idx, "filename": str(path), "error": str(e)}
                with open(out_root / f"item_{idx:05d}.json", "w", encoding="utf-8") as jf:
                    json.dump(err, jf, indent=2)
                results.append(err)

        # results.jsonl + summary
        with open(out_root / "results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary = {
            "folder": str(src),
            "timestamp": ts,
            "total_images": len(items),
            "processed": total,
            "total_labels": total_labels,
            "out_root": str(out_root),
        }
        with open(out_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved to: {out_root}")
        print(f"Processed: {total}   Total labels: {total_labels}")
        return summary


    def evaluate_dataset(
        self,
        dataset_id: str = "vikhyatk/CountBenchQA",
        split: str = "test",
        outdir: str = "results/hf_eval",
        max_examples: int = None,
        count_only_text: bool = True,
    ):
        out_root = Path(outdir) / f"{dataset_id.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_root.mkdir(parents=True, exist_ok=True)

        ds = load_dataset(dataset_id, split=split)
        total = 0
        correct = 0
        results = []

        # NEW: keep a handle to tqdm so we can set a postfix
        pbar = tqdm(ds, desc="Evaluating", unit="item")
        use_postfix = hasattr(pbar, "set_postfix")
        
        

        for i, row in enumerate(pbar):
            if max_examples is not None and i >= max_examples:
                break

            question = None
            gold = None

            try:
                img = row["image"]                  # PIL Image
                question = str(row["question"]).rstrip(" ?").strip()
                gold = int(row["number"])

                # 1) background
                self.set_background_from_pil(img, mode="fit")

                # 2) prompt TODO
                #prompt = sketch_first_prompt.format(concept=question, gt_sketches_str=gt_example)
                
                thing = re.sub(r"^[Hh]ow many\s+|\s+are there.*$", "", question).strip() or "object"
                prompt = COUNTING_PROMPT.format(thing=thing)



                use_stop = not isinstance(self.llm, GeminiAdapter)
                
                # 3) single LLM call
                answer = self.get_response_from_llm(
                    msg=prompt,
                    system_message=system_prompt.format(res=self.res),
                    msg_history=[],
                    init_canvas_str=self.last_canvas_b64,
                    seed_mode=self.seed_mode,
                    gen_mode="generation",
                    stop_sequences="</answer>" if use_stop else None,
                )

                

                # 4) count strokes
                pred = self._count_strokes(answer, count_only_text=count_only_text)

                # 5) save artifacts
                raw_path = out_root / f"item_{i:05d}_orig.jpg"
                row["image"].convert("RGB").save(str(raw_path), quality=95)

                svg_path = out_root / f"item_{i:05d}.svg"
                png_path = out_root / f"item_{i:05d}_annotated.png"
                self._render_answer_xml(answer, svg_out=svg_path, png_out=png_path)

                # 6) per-row JSON
                per_row = {
                    "index": i,
                    "prompt": question,
                    "ground_truth": gold,
                    "model_output": answer,
                    "model_answer": pred,
                    "correct": (pred == gold),
                    "raw_image": str(raw_path),
                    "grid_image": str(png_path).replace("_annotated", "_grid"),
                    "annotated_image": str(png_path),
                    "svg": str(svg_path),
                }
                with open(out_root / f"item_{i:05d}.json", "w", encoding="utf-8") as jf:
                    json.dump(per_row, jf, indent=2)

                # 7) tally + results.jsonl row
                is_correct = int(pred == gold)
                correct += is_correct
                total += 1

                results.append({
                    "index": i,
                    "question": question,
                    "gold_number": gold,
                    "pred_number": pred,
                    "correct": bool(is_correct),
                    "raw_image": str(raw_path),
                    "grid_image": str(png_path).replace("_annotated", "_grid"),
                    "annotated_image": str(png_path),
                    "svg": str(svg_path)
                })

            except Exception as e:
                total += 1
                err_row = {
                    "index": i,
                    "prompt": question,
                    "ground_truth": gold,
                    "error": str(e),
                }
                with open(out_root / f"item_{i:05d}.json", "w", encoding="utf-8") as jf:
                    json.dump(err_row, jf, indent=2)

                results.append({"index": i, "error": str(e)})

            finally:
                # NEW: live accuracy update after each row
                if total:
                    running_acc = correct / total
                    if use_postfix:
                        pbar.set_postfix(acc=f"{running_acc:.3%}", correct=f"{correct}/{total}")
                    else:
                        print(f"[{i:05d}] Running accuracy: {running_acc:.3%} ({correct}/{total})", flush=True)

        # Write results & summary
        with open(out_root / "results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        accuracy = (correct / total) if total else 0.0
        summary = {"dataset": dataset_id, "split": split, "total": total, "correct": correct, "accuracy": accuracy}
        with open(out_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nFinal accuracy: {accuracy:.3%}  ({correct}/{total})")
        return accuracy




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
def make_adapter(llm_name: str, model: str, cache: bool, max_tokens: int) -> BaseLLMAdapter:
    name = (llm_name or "").lower()
    if name in ("claude", "anthropic"):
        return ClaudeAdapter(model=model, cache=cache, max_tokens=max_tokens)
    if name in ("gpt", "openai"):
        return OpenAIAdapter(model=model, cache=cache, max_tokens=max_tokens)
    if name in ("gemini", "google"):
        return GeminiAdapter(model=model, cache=cache, max_tokens=max_tokens)
    raise ValueError("Unknown --llm. Use 'claude', 'gpt', or 'gemini'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SketchAgent (Claude, GPT, or Gemini)")
    parser.add_argument("--llm", choices=["claude", "gpt", "gemini"], required=True, help="Which provider to use.")
    parser.add_argument("--model", type=str, default=None, help="Model id (e.g., claude-3-5-sonnet-20240620, o3, or gemini-2.5-pro).")
    parser.add_argument("--deterministic", action="store_true", help="Set temperature=0 and top_k=1 (if supported).")
    parser.add_argument("--max-tokens", type=int, default=3000)
    
    parser.add_argument("--eval-dataset", type=str, help="HF dataset id, e.g. vikhyatk/CountBenchQA")
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
    cell_size = 12
    grid_size = (612, 612)
    stroke_width = cell_size * 0.6

    app = SketchApp(
        res=res,
        cell_size=cell_size,
        grid_size=grid_size,
        stroke_width=stroke_width,
        target_concept="sailboat",
        user_always_first=user_always_first,
        llm_adapter=adapter
    )

    app.api_delay_sec = args.api_delay

    if args.deterministic:
        app.seed_mode = "deterministic"
        
        
    # If --eval-dataset is provided, run batch mode and exit
    if args.eval_dataset:
        app.evaluate_dataset(
            dataset_id=args.eval_dataset,
            split=args.eval_split,
            outdir=args.outdir,
            max_examples=args.max_examples,
            count_only_text=args.count_only_text
        )
        raise SystemExit(0)  # do not start Flask in batch mode
    
    # If --label-dir is provided, run folder labeling and exit
    if args.label_dir:
        app.evaluate_labeling_folder(
            src_dir=args.label_dir,
            outdir=args.label_outdir,
            concept_mode=args.concept_mode,
            constant_concept=args.concept,
            labels_hint=args.labels_hint,
            max_images=args.max_images,
        )
        raise SystemExit(0)



    app.run(hostname, ip_address)
