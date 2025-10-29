#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bucket Physics Evaluation (boxed answers) — OpenAI/GPT variant

PATCH 8:
- Uses model="gpt-5" with reasoning={"effort": "<level>"} (default: medium)
- Robust response extraction across SDK shapes
- Debug mode that dumps BOTH the exact request payload (sanitized) and raw response
  per image to a folder you specify: --debug-dir path/
- CSV/JSONL schema mirrors your Gemini eval script
"""

import os
import re
import io
import csv
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List, Set

from PIL import Image
from dotenv import load_dotenv

DEFAULT_PROMPT: str = r"""You are given the start frame of a physics simulation. A ball is dropped from the top of the screen and falls due to gravity. The ball can roll off the lines or the walls in the image. The bouncing of the ball is relatively minor and realistic for normal gravity. Nothing in the image will move besides the ball. Predict which bucket will eventually catch the ball. There are 4 different buckets called bucket 1, bucket 2, bucket 3, and bucket 4. Please respond with what bucket the ball will fall into. Your final answer must be formatted as \"$\\boxed{bucket number}$\". For example, if the ball will fall into bucket 2, respond with \"$\\boxed{2}$\"."""
#DEFAULT_PROMPT: str = r"""You are given the start frame of a physics simulation. A ball is dropped from the top of the screen and falls due to gravity. The ball can roll off the lines or the walls in the image. The bouncing of the ball is relatively minor and realistic for normal gravity. Nothing in the image will move besides the ball. Predict which bucket will eventually catch the ball. There are 3 different buckets called bucket 1, bucket 2, bucket 3. Please respond with what bucket the ball will fall into. Your final answer must be formatted as \"$\\boxed{bucket number}$\". For example, if the ball will fall into bucket 2, respond with \"$\\boxed{2}$\"."""
SYSTEM_FORCE_BOXED = (
    "Return ONLY a single LaTeX-style boxed answer in the exact form "
    "$\\boxed{1}$, $\\boxed{2}$, $\\boxed{3}$, or $\\boxed{4}$."
    "Do not include any other words, symbols, or explanation."
)

_BOXED_RE = re.compile(r"\$\\boxed\{\s*(1|2|3|4|none)\s*\}\$", re.IGNORECASE)
_LOOSE_RE = re.compile(r"\b(1|2|3|4|none)\b", re.IGNORECASE)

BucketVal = Literal["1", "2", "3", "4", "none"]

def parse_bucket(text: Optional[str], *, none_as_zero: bool = False) -> Optional[str]:
    if not text:
        return None
    m = _BOXED_RE.search(text)
    if not m:
        return None
    val = m.group(1).lower()
    if none_as_zero and val == "none":
        return "0"
    return val

def parse_bucket_loose(text: Optional[str], *, none_as_zero: bool = False) -> Optional[str]:
    if not text:
        return None
    m = _BOXED_RE.search(text)
    if not m:
        m = _LOOSE_RE.search(text)
        if not m:
            return None
        val = m.group(1).lower()
    else:
        val = m.group(1).lower()
    if none_as_zero and val == "none":
        return "0"
    return val

def label_to_int(label: BucketVal) -> int:
    return 0 if label == "none" else int(label)

def int_to_label(x: int) -> BucketVal:
    if x == 0:
        return "none"
    if x in (1,2,3,4):
        return str(x)  # type: ignore[return-value]
    raise ValueError("Bucket must be 0 or 1..4")

def _as_dict(resp: Any) -> Dict[str, Any]:
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
    try:
        if isinstance(resp, str):
            return json.loads(resp)
    except Exception:
        pass
    return {}

def _extract_output_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out = []
    output_list = getattr(resp, "output", None)
    if isinstance(output_list, list):
        for blk in output_list:
            parts = getattr(blk, "content", None)
            if isinstance(parts, list):
                for p in parts:
                    t = getattr(p, "type", None)
                    if t in ("output_text", "text"):
                        tval = getattr(p, "text", None) or getattr(p, "value", None)
                        if isinstance(tval, str):
                            out.append(tval)
                    elif t == "refusal":
                        tval = getattr(p, "text", None) or getattr(p, "refusal", None) or getattr(p, "value", None)
                        if isinstance(tval, str):
                            out.append(tval)
        if out:
            return "".join(out).strip()

    d = _as_dict(resp)
    try:
        outputs = d.get("output") or d.get("outputs") or []
        for blk in outputs:
            parts = blk.get("content", [])
            for p in parts:
                t = p.get("type")
                if t in ("output_text", "text", "refusal"):
                    tv = p.get("text") or p.get("refusal") or p.get("value")
                    if isinstance(tv, str):
                        out.append(tv)
        if out:
            return "".join(out).strip()
    except Exception:
        pass

    return ""

class OpenAIAdapter:
    def __init__(self, model_name: str = "gpt-5", api_key_env: str = "OPENAI_API_KEY",
                 temperature: Optional[float] = None, max_output_tokens: int = 32,
                 reasoning_effort: Optional[str] = None, verbose: bool = False,
                 debug_dir: Optional[Path] = None):
        load_dotenv()
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"Missing API key in env var {api_key_env}")
        from openai import OpenAI  # lazy import
        self._client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.verbose = verbose
        self.debug_dir = debug_dir
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pil_to_data_url(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def predict(self, img_path: Path, prompt: str, timeout: int = 80) -> str:
        img = Image.open(img_path).convert("RGB")
        try:
            data_url = self._pil_to_data_url(img)
            req_payload: Dict[str, Any] = dict(
                model=self.model_name,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_FORCE_BOXED}]},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ]},
                ],
                max_output_tokens=self.max_output_tokens,
            )
            if self.reasoning_effort:
                req_payload["reasoning"] = {"effort": str(self.reasoning_effort)}
            if self.temperature is not None:
                req_payload["temperature"] = float(self.temperature)

            if self.debug_dir:
                try:
                    req_path = self.debug_dir / (img_path.stem + ".request.json")
                    with open(req_path, "w", encoding="utf-8") as f:
                        json.dump(req_payload, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            resp = self._client.responses.create(**req_payload)

            if self.debug_dir:
                try:
                    rawd = _as_dict(resp)
                    dump_path = self.debug_dir / (img_path.stem + ".response.json")
                    with open(dump_path, "w", encoding="utf-8") as f:
                        json.dump(rawd, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            return _extract_output_text(resp)
        finally:
            img.close()

def load_labels_any(path: Optional[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path or not path.is_file():
        return mapping
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = (row.get("image") or row.get("file") or row.get("filename") or "").strip()
                lab = (row.get("label") or row.get("bucket") or "").strip().lower()
                if not img or not lab:
                    continue
                if lab in {"0", "none"}:
                    mapping[Path(img).name] = "none"
                elif lab in {"1", "2", "3", "4"}:
                    mapping[Path(img).name] = lab
    elif path.suffix.lower() == ".json":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for k, v in obj.items():
                img = Path(str(k)).name
                lab = str(v).lower().strip()
                if lab in {"0", "none"}:
                    mapping[img] = "none"
                elif lab in {"1", "2", "3", "4"}:
                    mapping[img] = lab
    else:
        raise ValueError("Unsupported labels file (use .csv or .json)")
    return mapping

def evaluate(
    images_dir: Path,
    model_kind: str = "openai",
    openai_model: str = "gpt-5",
    out_jsonl: Path = Path("logs_buckets_boxed_gpt.jsonl"),
    out_csv: Path = Path("results_buckets_boxed_gpt.csv"),
    labels_path: Optional[Path] = None,
    none_as_zero: bool = False,
    strict_parse: bool = True,
    shuffle: bool = False,
    limit: Optional[int] = None,
    temperature: Optional[float] = None,
    sleep: float = 0.0,
    verbose: bool = False,
    flush_files: bool = False,
    use_progress: bool = False,
    only_indices: Optional[Set[int]] = None,
    max_output_tokens: int = 32,
    reasoning_effort: Optional[str] = "medium",
    debug_dir: Optional[Path] = None,
) -> None:
    imgs: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        imgs.extend(images_dir.rglob(ext))
    imgs = sorted(set(imgs), key=lambda p: p.as_posix())
    if shuffle:
        import random
        random.shuffle(imgs)
    if limit is not None:
        imgs = imgs[:int(limit)]
    if not imgs:
        raise RuntimeError(f"No images found under: {images_dir}")

    if model_kind.lower() in {"openai", "gpt"}:
        adapter = OpenAIAdapter(
            model_name=openai_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            verbose=verbose,
            debug_dir=debug_dir,
        )
        model_suffix = f"-{reasoning_effort}" if reasoning_effort else ""
        model_descriptor = f"openai::{openai_model}{model_suffix}"
    else:
        raise ValueError("Only --model openai|gpt is implemented in this script.")

    labels = load_labels_any(labels_path)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_parsed = 0
    n_correct = 0

    iterator = enumerate(imgs)
    tqdm_write = None
    if use_progress:
        try:
            from tqdm import tqdm
            t = tqdm(imgs, total=len(imgs))
            iterator = enumerate(t)
            tqdm_write = t.write
        except ImportError:
            print("[warn] tqdm not installed; running without progress bar.")
            use_progress = False

    with open(out_jsonl, "w", encoding="utf-8") as fj, open(out_csv, "w", newline="", encoding="utf-8") as fc:
        csv_writer = csv.writer(fc)
        csv_writer.writerow(["image","model","raw_text","parsed_label","parsed_int","gold","gold_int","correct"])

        for i, img_path in iterator:
            n_total += 1
            fname = img_path.name

            if only_indices is not None and i not in only_indices:
                continue

            try:
                text = adapter.predict(img_path, DEFAULT_PROMPT)
            except Exception as e:
                text = f"[ERROR] {type(e).__name__}: {e}"

            parsed = parse_bucket(text, none_as_zero=none_as_zero) if strict_parse else parse_bucket_loose(text, none_as_zero=none_as_zero)
            if parsed is None:
                parsed = parse_bucket_loose(text, none_as_zero=none_as_zero)

            parsed_int: Optional[int] = None
            if parsed is not None:
                n_parsed += 1
                if parsed == "0":
                    parsed_int = 0
                elif parsed in {"1","2","3","4"}:
                    parsed_int = int(parsed)
                elif parsed == "none":
                    parsed_int = 0

            g_label = labels.get(fname, "")
            g_int: Optional[int] = None
            if g_label:
                if g_label == "none":
                    g_int = 0
                else:
                    try:
                        g_int = int(g_label)
                    except Exception:
                        g_int = None

            correct = (parsed_int is not None and g_int is not None and parsed_int == g_int)
            if correct:
                n_correct += 1

            rec = {
                "index": i,
                "file": str(img_path),
                "prompt": "Bucket Physics (boxed)",
                "model": model_descriptor,
                "raw_text": text,
                "parsed_label": parsed,
                "parsed_int": parsed_int,
                "gold": g_label if g_label else None,
                "gold_int": g_int,
                "correct": correct if g_label else None,
            }
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")

            csv_writer.writerow([fname, model_descriptor, text, parsed, parsed_int, g_label, g_int, int(bool(correct)) if g_label else ""])

            if verbose:
                msg = f"[{i+1}/{len(imgs)}] {fname} -> {parsed or '∅'}"
                if g_label:
                    msg += f" (gold={g_label}, {'✓' if correct else '✗'})"
                if tqdm_write:
                    tqdm_write(msg)
                else:
                    print(msg)

            if flush_files:
                fj.flush()
                fc.flush()

            if sleep > 0:
                time.sleep(sleep)

    print(f"Total images: {n_total}")
    print(f"Parsed answers: {n_parsed} ({(100.0*n_parsed/max(1,n_total)):.1f}%)")
    if labels:
        print(f"Accuracy (on labeled subset): {n_correct}/{len(labels)} = {(100.0*n_correct/max(1,len(labels))):.2f}%")
    print(f"Wrote JSONL: {out_jsonl}")
    print(f"Wrote CSV:   {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate bucket-physics predictions with boxed outputs (OpenAI/GPT)." )
    ap.add_argument("--images", type=str, required=True, help="Folder of input images.")
    ap.add_argument("--model", type=str, default="openai", help="Model family: 'openai' or 'gpt'.")
    ap.add_argument("--openai-model", type=str, default="gpt-5", help="OpenAI model id (e.g., gpt-5)." )
    ap.add_argument("--out", type=str, default="logs_buckets_boxed_gpt.jsonl", help="Output JSONL path.")
    ap.add_argument("--csv", type=str, default="results_buckets_boxed_gpt.csv", help="Output CSV path.")
    ap.add_argument("--labels", type=str, default="", help="Optional labels file (.csv or .json)." )
    ap.add_argument("--none-as-zero", action="store_true", help="Map 'none' to '0' in the parser output (for legacy evals)." )
    ap.add_argument("--loose", action="store_true", help="Use loose parsing first (otherwise strict boxed first)." )
    ap.add_argument("--shuffle", action="store_true", help="Shuffle image order." )
    ap.add_argument("--limit", type=int, default=None, help="Limit number of images." )
    ap.add_argument("--temperature", type=float, default=None, help="Sampling temperature (omitted by default)." )
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests (rate limiting)." )
    ap.add_argument("--max-output-tokens", type=int, default=32, help="Max output tokens for the model." )
    ap.add_argument("--verbose", action="store_true", help="Print per-sample result to stdout." )
    ap.add_argument("--flush", dest="flush_files", action="store_true", help="Flush files after each write." )
    ap.add_argument("--progress", dest="use_progress", action="store_true", help="Show a tqdm progress bar." )
    ap.add_argument("--only", type=str, default="", help="Comma-separated list of zero-based item indices to (re)run." )
    ap.add_argument("--reasoning-effort", choices=["minimal","low","medium","high"], default="medium",
                    help="GPT‑5 reasoning effort level." )
    ap.add_argument("--debug-dir", type=str, default="", help="Optional folder to dump raw JSON Responses AND request payloads." )
    args = ap.parse_args()

    images_dir = Path(args.images)
    out_jsonl = Path(args.out)
    out_csv = Path(args.csv)
    labels_path = Path(args.labels) if args.labels else None
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    only_indices = None
    if args.only.strip():
        only_indices = {int(tok) for tok in re.split(r"[,\s]+", args.only.strip()) if tok.isdigit()}

    evaluate(
        images_dir=images_dir,
        model_kind=args.model,
        openai_model=args.openai_model,
        out_jsonl=out_jsonl,
        out_csv=out_csv,
        labels_path=labels_path,
        none_as_zero=args.none_as_zero,
        strict_parse=(not args.loose),
        shuffle=args.shuffle,
        limit=args.limit,
        temperature=args.temperature,
        sleep=args.sleep,
        verbose=args.verbose,
        flush_files=args.flush_files,
        use_progress=args.use_progress,
        only_indices=only_indices,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        debug_dir=debug_dir,
    )

if __name__ == "__main__":
    main()

'''


python gpt_baseline_ball_drop.py --images datasets/ball_number --openai-model gpt-5 --reasoning-effort medium --max-output-tokens 10000 --out results/mix_eval/ball_number_gpt5_medium.jsonl --csv results/mix_eval/ball_number_gpt5_medium.csv --debug-dir results/mix_eval/raw_responses --progress --verbose

python gpt_baseline_ball_drop.py --images datasets/vpct_ball_drop --openai-model gpt-5 --reasoning-effort high --max-output-tokens 20000 --out results/mix_eval/vpct_ball_gpt5_high_2.jsonl --csv results/mix_eval/vpct_ball_gpt5_high_2.csv --debug-dir results/mix_eval/raw_responses_2 --progress --verbose 


  '''