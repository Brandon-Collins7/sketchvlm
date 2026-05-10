#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-provider Connect-the-Dots (mixed_connect) baseline
Supports OpenAI GPT-5 and Qwen3-VL via OpenRouter.
(Standalone; no llm_adapters dependency.)

See usage examples at bottom of file.
"""

import os, io, re, json, base64, argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from dotenv import load_dotenv

# ---------------------- OpenAI Responses API adapter ----------------------

def _resp_to_dict(resp: Any) -> Dict[str, Any]:
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(resp, attr, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    try:
        js = getattr(resp, "model_dump_json", lambda: "")()
        if js:
            return json.loads(js)
    except Exception:
        pass
    return {}

def _extract_output_text(resp: Any) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    out = []
    output = getattr(resp, "output", None)
    if isinstance(output, list):
        for blk in output:
            parts = getattr(blk, "content", None)
            if isinstance(parts, list):
                for p in parts:
                    typ = getattr(p, "type", None)
                    if typ in ("output_text", "text"):
                        val = getattr(p, "text", None) or getattr(p, "value", None)
                        if isinstance(val, str):
                            out.append(val)
                    elif typ == "refusal":
                        val = getattr(p, "text", None) or getattr(p, "refusal", None) or getattr(p, "value", None)
                        if isinstance(val, str):
                            out.append(val)
        if out:
            return "".join(out).strip()

    d = _resp_to_dict(resp)
    output = d.get("output") or d.get("outputs") or []
    for blk in output:
        for p in blk.get("content", []):
            typ = p.get("type")
            if typ in ("output_text", "text", "refusal"):
                val = p.get("text") or p.get("refusal") or p.get("value")
                if isinstance(val, str):
                    out.append(val)
    return "".join(out).strip() if out else ""

class OpenAIAdapter:
    def __init__(self, model: str = "gpt-5", reasoning_effort: Optional[str] = "medium",
                 temperature: Optional[float] = None, max_output_tokens: int = 2048,
                 api_key_env: str = "OPENAI_API_KEY", debug_dir: Optional[Path] = None,
                 verbose: bool = False):
        load_dotenv()
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"Missing API key in env var {api_key_env}")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.debug_dir = debug_dir
        self.verbose = verbose
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
        import io, base64
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/{fmt.lower()};base64,{b64}"

    def call(self, system: str, user_text: str, img: Image.Image) -> str:
        data_url = self.pil_to_data_url(img, fmt="PNG")
        payload = dict(
            model=self.model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": data_url},
                ]},
            ],
            max_output_tokens=self.max_output_tokens,
        )
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": str(self.reasoning_effort)}
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)

        if self.debug_dir:
            try:
                (self.debug_dir / "requests").mkdir(parents=True, exist_ok=True)
                idx = len(list((self.debug_dir / "requests").glob("*.json")))
                (self.debug_dir / "requests" / f"{idx:06d}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
            except Exception:
                pass

        resp = self.client.responses.create(**payload)

        if self.debug_dir:
            try:
                (self.debug_dir / "responses").mkdir(parents=True, exist_ok=True)
                idx = len(list((self.debug_dir / "responses").glob("*.json")))
                (self.debug_dir / "responses" / f"{idx:06d}.json").write_text(json.dumps(_resp_to_dict(resp), ensure_ascii=False, indent=2), "utf-8")
            except Exception:
                pass

        return _extract_output_text(resp)


# ---------------------- OpenRouter Adapter (Qwen3) ----------------------

class OpenRouterAdapter:
    """
    Uses OpenRouter API with Alibaba provider for Qwen3-VL models.
    Interface matches OpenAIAdapter for drop-in compatibility.
    """
    def __init__(self, model: str = "alibaba/qwen-3-vl-72b",
                 reasoning_effort: Optional[str] = None,  # Ignored for Qwen (OpenAI-only feature)
                 temperature: Optional[float] = None, max_output_tokens: int = 2048,
                 api_key_env: str = "OPENROUTER_API_KEY", debug_dir: Optional[Path] = None,
                 verbose: bool = False):
        load_dotenv()
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"Missing API key in env var {api_key_env}")

        import openai
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        self.model = model
        self.reasoning_effort = reasoning_effort  # Store but don't use (for compatibility)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.debug_dir = debug_dir
        self.verbose = verbose
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
        import io, base64
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/{fmt.lower()};base64,{b64}"

    def call(self, system: str, user_text: str, img: Image.Image) -> str:
        data_url = self.pil_to_data_url(img, fmt="PNG")

        # Build messages with system message and user content
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ]},
        ]

        args = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_output_tokens,
            extra_body={
                "provider": {
                    "only": ["alibaba"],      # enforce single provider
                    "allow_fallbacks": False  # never fall back
                }
            }
        )
        if self.temperature is not None:
            args["temperature"] = float(self.temperature)

        if self.debug_dir:
            try:
                (self.debug_dir / "requests").mkdir(parents=True, exist_ok=True)
                idx = len(list((self.debug_dir / "requests").glob("*.json")))
                # Convert messages to JSON-serializable format for debugging
                debug_msgs = []
                for m in messages:
                    if isinstance(m["content"], list):
                        debug_content = []
                        for item in m["content"]:
                            if item.get("type") == "image_url":
                                debug_content.append({"type": "image_url", "image_url": {"url": "[IMAGE_DATA]"}})
                            else:
                                debug_content.append(item)
                        debug_msgs.append({"role": m["role"], "content": debug_content})
                    else:
                        debug_msgs.append(m)
                (self.debug_dir / "requests" / f"{idx:06d}.json").write_text(
                    json.dumps({"messages": debug_msgs, "model": self.model}, ensure_ascii=False, indent=2),
                    "utf-8"
                )
            except Exception:
                pass

        resp = self.client.chat.completions.create(**args)

        if self.debug_dir:
            try:
                (self.debug_dir / "responses").mkdir(parents=True, exist_ok=True)
                idx = len(list((self.debug_dir / "responses").glob("*.json")))
                resp_dict = {
                    "id": resp.id,
                    "model": resp.model,
                    "choices": [{"message": {"content": resp.choices[0].message.content}}] if resp.choices else []
                }
                (self.debug_dir / "responses" / f"{idx:06d}.json").write_text(
                    json.dumps(resp_dict, ensure_ascii=False, indent=2),
                    "utf-8"
                )
            except Exception:
                pass

        # Extract text from response
        if hasattr(resp, "choices") and resp.choices:
            return resp.choices[0].message.content.strip()
        return ""


# ---------------------- Connect-the-dots helpers ----------------------

CONNECT_SYS = "You are a helpful vision assistant. Return ONLY plain text line segments."

def build_connect_prompt(user_prompt: str, coord_max: int = 1000, origin: str = "top-left") -> str:
    origin_note = (
        "The origin (0,0) is the TOP-LEFT corner."
        if origin.lower() == "top-left"
        else "The origin (0,0) is the BOTTOM-LEFT corner."
    )
    return (
        f"{(user_prompt or 'Connect the dots in numeric order.').strip()}\n\n"
        "Return ONLY a list of straight line segments connecting the dots, as pairs:\n"
        "(x1,y1)->(x2,y2)\n\n"
        f"Coordinates are integers normalized from 0 to {coord_max} on both axes. {origin_note}\n"
        "Do not output SVG, paths, code blocks, or prose—only one segment per line in the exact format above."
    )

_SEG_RE_1 = re.compile(
    r"\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)\s*(?:->|to|[-–—])\s*\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)",
    re.I,
)
_SEG_RE_2 = re.compile(
    r"\(\s*X\s*[:=]\s*(\d{1,4})\s*,\s*Y\s*[:=]\s*(\d{1,4})\s*\)\s*(?:->|to|[-–—])\s*"
    r"\(\s*X\s*[:=]\s*(\d{1,4})\s*,\s*Y\s*[:=]\s*(\d{1,4})\s*\)", re.I,
)
_PATH_POINT_RE = re.compile(r"[ML]\s*(\d{1,4})\s*[, ]\s*(\d{1,4})", re.I)

def parse_segments_norm(text: str) -> List[Dict[str, int]]:
    s = (text or "").strip()
    segs: List[Dict[str, int]] = []
    for m in _SEG_RE_1.finditer(s):
        x1, y1, x2, y2 = map(int, m.groups())
        segs.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    if not segs:
        for m in _SEG_RE_2.finditer(s):
            x1, y1, x2, y2 = map(int, m.groups())
            segs.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    if not segs:
        pts = [(int(a), int(b)) for a, b in _PATH_POINT_RE.findall(s)]
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                (x1, y1), (x2, y2) = pts[i], pts[i + 1]
                segs.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    for seg in segs:
        for k in ("x1", "y1", "x2", "y2"):
            seg[k] = max(0, min(1000, int(seg[k])))
    return segs

def segs_norm_to_px(segs: List[Dict[str, int]], w: int, h: int, coord_max: int = 1000, origin: str = "top-left") -> List[Dict[str, float]]:
    def clamp(v: float) -> float:
        return max(0.0, min(float(coord_max), float(v)))
    def sx(x: float) -> float:
        return clamp(x) * (w / float(coord_max))
    def sy(y: float) -> float:
        y = clamp(y)
        if origin.lower() == "bottom-left":
            y = coord_max - y
        return y * (h / float(coord_max))
    out = []
    for s in segs:
        x1, y1, x2, y2 = s.get("x1"), s.get("y1"), s.get("x2"), s.get("y2")
        if None in (x1, y1, x2, y2):
            continue
        out.append({**s, "x1_px": sx(x1), "y1_px": sy(y1), "x2_px": sx(x2), "y2_px": sy(y2)})
    return out

def make_svg_lines(w: int, h: int, segs_px: List[Dict[str, float]]) -> str:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    for s in segs_px:
        parts.append(
            f'<line x1="{s["x1_px"]}" y1="{s["y1_px"]}" '
            f'x2="{s["x2_px"]}" y2="{s["y2_px"]}" stroke="red" stroke-width="3" />'
        )
    parts.append("</svg>")
    return "\n".join(parts)

def draw_lines(img: Image.Image, segs_px: List[Dict[str, float]]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for s in segs_px:
        draw.line((s["x1_px"], s["y1_px"], s["x2_px"], s["y2_px"]), fill=(255, 0, 0), width=3)
    return out

# ---------------------- Main evaluation ----------------------

def evaluate_mixed_connect(
    src_dir: str = "datasets/mixed_connect",
    outdir: str = "results/mix_connect_gpt5",
    provider: str = "openai",
    model: str = "gpt-5",
    reasoning_effort: str = "medium",
    temperature: Optional[float] = None,
    max_output_tokens: int = 20000,
    coord_max: int = 1000,
    origin: str = "top-left",
    max_images: Optional[int] = None,
    debug_dir: Optional[str] = None,
    progress: bool = False,
    verbose: bool = False,
) -> int:
    src = Path(src_dir)
    assert src.exists() and src.is_dir(), f"Folder not found: {src}"

    out_root = Path(outdir) / src.name
    out_root.mkdir(parents=True, exist_ok=True)

    debug_root = Path(debug_dir) if debug_dir else None
    if debug_root:
        debug_root.mkdir(parents=True, exist_ok=True)

    # Select adapter based on provider
    provider_lower = provider.lower()
    if provider_lower in ("openai", "gpt"):
        adapter = OpenAIAdapter(
            model=model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            debug_dir=debug_root,
            verbose=verbose,
        )
    elif provider_lower in ("openrouter", "qwen", "qwen3"):
        adapter = OpenRouterAdapter(
            model=model,
            reasoning_effort=reasoning_effort,  # Passed but ignored (for compatibility)
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            debug_dir=debug_root,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'openrouter'.")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    images = [p for p in sorted(src.iterdir()) if p.suffix.lower() in exts]
    if max_images is not None:
        images = images[:max_images]

    if progress:
        try:
            from tqdm import tqdm
            iterator = enumerate(tqdm(images, total=len(images)))
            log = tqdm.write
        except Exception:
            iterator = enumerate(images)
            log = print
    else:
        iterator = enumerate(images)
        log = print if verbose else (lambda *a, **k: None)

    rows: List[Dict] = []

    for i, img_path in iterator:
        txt_path = img_path.with_suffix(".txt")
        raw_path = out_root / f"item_{i:05d}_orig.jpg"
        svg_path = out_root / f"item_{i:05d}_overlay.svg"
        ann_path = out_root / f"item_{i:05d}_annotated.jpg"
        json_path_i = out_root / f"item_{i:05d}.json"

        rec: Dict[str, Any] = {"index": i, "image": str(img_path)}
        try:
            img = Image.open(img_path).convert("RGB")
            img.save(raw_path, quality=95)
            w, h = img.size

            if txt_path.exists():
                user_prompt = txt_path.read_text(encoding="utf-8").strip()
            else:
                user_prompt = "Connect the dots in numeric order."

            prompt = build_connect_prompt(user_prompt, coord_max=coord_max, origin=origin)

            raw_text = adapter.call(CONNECT_SYS, prompt, img).strip()

            segs_norm = parse_segments_norm(raw_text)
            segs_px = segs_norm_to_px(segs_norm, w, h, coord_max=coord_max, origin=origin)

            svg = make_svg_lines(w, h, segs_px)
            svg_path.write_text(svg, encoding="utf-8")
            draw_lines(img, segs_px).save(ann_path, quality=95)

            rec.update({
                "prompt": user_prompt,
                "coord_max": coord_max,
                "origin": origin,
                "image_w": w, "image_h": h,
                "model_raw_text": raw_text,
                "lines_norm": segs_norm,
                "lines_px": segs_px,
                "num_lines": len(segs_px),
                "files": {
                    "raw_image": str(raw_path),
                    "svg_overlay": str(svg_path),
                    "annotated_image": str(ann_path),
                },
            })
            rows.append(rec)
            log(f"[{i+1}/{len(images)}] {img_path.name}: {len(segs_px)} segments")

        except Exception as e:
            rec.update({"error": str(e), "prompt_file": str(txt_path)})
            rows.append(rec)

        json_path_i.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out_root / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (out_root / "summary.json").write_text(
        json.dumps({"folder": str(src), "total": len(images), "provider": provider, "model": model}, indent=2),
        encoding="utf-8")
    print(f"[MIX CONNECT/{provider.upper()}] Processed: {len(images)} images with {model}  ->  {out_root}")
    return len(images)

def main():
    ap = argparse.ArgumentParser(description="Multi-provider connect-the-dots baseline (GPT-5, Qwen3, etc.)")
    ap.add_argument("--src-dir", default="datasets/mixed_connect")
    ap.add_argument("--outdir", default="results/mix_connect_gpt5")
    ap.add_argument("--provider", choices=["openai", "gpt", "openrouter", "qwen", "qwen3"], default="openai",
                    help="LLM provider: 'openai'/'gpt' for GPT-5, 'openrouter'/'qwen'/'qwen3' for Qwen3-VL")
    ap.add_argument("--model", default="gpt-5",
                    help="Model name (e.g., 'gpt-5' for OpenAI, 'alibaba/qwen-3-vl-72b' for Qwen3)")
    ap.add_argument("--reasoning-effort", choices=["minimal","low","medium","high"], default="medium",
                    help="Reasoning effort (OpenAI GPT-5 only)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--coord-max", type=int, default=1000)
    ap.add_argument("--origin", choices=["top-left","bottom-left"], default="top-left")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--debug-dir", type=str, default="")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    evaluate_mixed_connect(
        src_dir=args.src_dir,
        outdir=args.outdir,
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        coord_max=args.coord_max,
        origin=args.origin,
        max_images=args.max_images,
        debug_dir=(args.debug_dir or None),
        progress=args.progress,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()


'''

GPT-5 medium

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/random_source --outdir results/gpt5_dots_random --provider openai --model gpt-5 --reasoning-effort medium --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/mix_connect_gpt5/raw_dots --progress --verbose --max-output-tokens 20000

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/worksheets_source --outdir results/gpt5_dots_worksheets --provider openai --model gpt-5 --reasoning-effort medium --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/mix_connect_gpt5/raw_worksheets --progress --verbose --max-output-tokens 20000

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/outlines_source --outdir results/gpt5_dots_outlines --provider openai --model gpt-5 --reasoning-effort medium --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/mix_connect_gpt5/raw_outlines --progress --verbose --max-output-tokens 20000


GPT-5 low

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/random_source --outdir results/gpt5_low_dots_random --provider openai --model gpt-5 --reasoning-effort low --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/mix_connect_gpt5/raw_outlines_low --progress --verbose --max-output-tokens 20000

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/worksheets_source --outdir results/gpt5_low_dots_worksheets --provider openai --model gpt-5 --reasoning-effort low --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/mix_connect_gpt5/raw_worksheets_low --progress --verbose --max-output-tokens 20000

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/outlines_source --outdir results/gpt5_low_dots_outlines --provider openai --model gpt-5 --reasoning-effort low --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/mix_connect_gpt5/raw_outlines_low --progress --verbose --max-output-tokens 20000


GPT-5 high

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/random_source --outdir results/gpt5_high_dots_random --provider openai --model gpt-5 --reasoning-effort high --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/mix_connect_gpt5/raw_outlines_high --progress --verbose --max-output-tokens 20000

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/worksheets_source --outdir results/gpt5_high_dots_worksheets --provider openai --model gpt-5 --reasoning-effort high --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/mix_connect_gpt5/raw_worksheets_high --progress --verbose --max-output-tokens 20000

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/outlines_source --outdir results/gpt5_high_dots_outlines --provider openai --model gpt-5 --reasoning-effort high --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/mix_connect_gpt5/raw_outlines_high --progress --verbose --max-output-tokens 20000


Qwen3-VL (via OpenRouter)

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/random_source --outdir results/qwen3_dots_random --provider qwen --model alibaba/qwen-3-vl-72b --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/qwen3/raw_dots --progress --verbose --max-output-tokens 20000

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/worksheets_source --outdir results/qwen3_dots_worksheets --provider qwen --model alibaba/qwen-3-vl-72b --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/qwen3/raw_worksheets --progress --verbose --max-output-tokens 20000

python gpt_baseline_connect.py --src-dir connecting_dots_dataset/outlines_source --outdir results/qwen3_dots_outlines --provider qwen --model alibaba/qwen-3-vl-72b --coord-max 1000 --origin top-left --max-images 50 --debug-dir results/qwen3/raw_outlines --progress --verbose --max-output-tokens 20000


'''