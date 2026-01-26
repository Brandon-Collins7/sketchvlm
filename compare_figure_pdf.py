#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_figure_pdf_unified2.py

Unified PDF comparison figure generator for:
- VPCT bucket prediction (--task vpct)
- Path Navigation validity (--task pathnav)

PLUS support for additional column "kinds" with different artifact layouts:
- standard    : item_00000_annotated.png (or _annotated_color / _orig fallback) + item_*.json for preds
- nanobanana  : item_00000_generated_0.png (or other generated_N) + item_*.json for preds (if present)
- thinkmorph  : sample_*_<SOURCE_STEM>/images/image_0.png (or highest image_N) + optional json/txt for preds
- vilasr      : sample_* (no source tracked) => align by ORDER with thinkmorph sample folders, then use images/image_<highest>.png
               + optional json/txt for preds

Why vilasr needs a reference:
- It doesn't include the source image name, so we use ordering to match.
- If you include at least one thinkmorph column in the same figure, vilasr will align to it automatically.
- If no thinkmorph column is provided, vilasr falls back to dataset row order (best-effort).

Usage (new flags):
  --col-kind standard|nanobanana|thinkmorph|vilasr   (repeat once per --col-dir, defaults to standard)
"""

import argparse
from email.mime import text
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth


SIM_IMG_RE = re.compile(r"sim_(\d+)_initial\.png$", re.I)

# -----------------------------
# parsing helpers
# -----------------------------

def to_int_123(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        s = str(v).strip()
        m = re.search(r"\b([123])\b", s)
        if m:
            iv = int(m.group(1))
            return iv if iv in (1, 2, 3) else None
        iv = int(s)
        return iv if iv in (1, 2, 3) else None
    except Exception:
        return None

def extract_last_int_token(text: str) -> Optional[int]:
    if not text:
        return None
    vals: List[int] = []
    for m in re.finditer(r"-?\d+", text):
        s, e = m.span()
        prev = text[s - 1] if s > 0 else ""
        prevprev = text[s - 2] if s > 1 else ""
        nxt = text[e] if e < len(text) else ""
        nxtnxt = text[e + 1] if e + 1 < len(text) else ""
        # reject decimals like 21.5
        if nxt == "." and nxtnxt.isdigit():
            continue
        if prev == "." and prevprev.isdigit():
            continue
        try:
            vals.append(int(m.group()))
        except Exception:
            pass
    return vals[-1] if vals else None

def extract_last_bucket_from_output(text: str) -> Optional[int]:
    iv = extract_last_int_token(text or "")
    return iv if iv in (1, 2, 3) else None

def normalize_pathnav_label(s: str) -> Optional[str]:
    if not s:
        return None
    t = str(s).lower()
    # strip common LaTeX wrappers
    t = t.replace("$", " ")
    t = re.sub(r"\\boxed\s*\{", " ", t)
    t = t.replace("}", " ")
    # IMPORTANT: check "invalid" first (since "valid" is a substring)
    if re.search(r"\binvalid\b", t):
        return "invalid"
    if re.search(r"\bvalid\b", t):
        return "valid"
    return None

def parse_rows_spec(spec: Optional[str]) -> Optional[List[int]]:
    if not spec or not spec.strip():
        return None
    s = spec.replace(" ", ",").strip()
    out: List[int] = []
    seen = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a = a.strip()
            b = b.strip()
            if a.isdigit() and b.isdigit():
                lo = int(a); hi = int(b)
                step = 1 if hi >= lo else -1
                for x in range(lo, hi + step, step):
                    if x not in seen:
                        seen.add(x); out.append(x)
            continue
        if tok.isdigit():
            x = int(tok)
            if x not in seen:
                seen.add(x); out.append(x)
    return out

# -----------------------------
# column kinds / run examples
# -----------------------------

COL_KINDS = ("standard", "nanobanana", "thinkmorph", "vilasr")

@dataclass
class ColumnSpec:
    title: str
    run_dir: Path
    kind: str

@dataclass
class RunExample:
    pred_int: Optional[int] = None
    pred_label: Optional[str] = None
    img_path: Optional[Path] = None
    item_idx: Optional[str] = None
    source_image_raw: Optional[str] = None
    source_basename: Optional[str] = None
    sample_dir: Optional[Path] = None

# -----------------------------
# GT loaders
# -----------------------------

def load_gt_vpct(gt_root: Path) -> Tuple[List[str], Dict[str, int]]:
    gt_map: Dict[str, int] = {}
    for jf in sorted(gt_root.glob("sim_*_results.json")):
        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        gt = to_int_123(j.get("finalBucket"))
        if gt is None:
            continue
        m = re.search(r"sim_(\d+)_results\.json$", jf.name, re.I)
        if not m:
            continue
        img = f"sim_{m.group(1)}_initial.png"
        gt_map[img] = gt

    def sort_key(name: str):
        m = SIM_IMG_RE.search(name)
        return (int(m.group(1)) if m else 10**9, name)

    ordered = sorted(gt_map.keys(), key=sort_key)
    return ordered, gt_map

# -----------------------------
# image pickers
# -----------------------------

def pick_standard_item_image(run_dir: Path, item_idx: str) -> Optional[Path]:
    cand_color = run_dir / f"item_{item_idx}_annotated_color.png"
    if cand_color.exists():
        return cand_color
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = run_dir / f"item_{item_idx}_annotated{ext}"
        if cand.exists():
            return cand
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = run_dir / f"item_{item_idx}_orig{ext}"
        if cand.exists():
            return cand
    return None

def pick_nanobanana_item_image(run_dir: Path, item_idx: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand0 = run_dir / f"item_{item_idx}_generated_0{ext}"
        if cand0.exists():
            return cand0
    gen = []
    for p in run_dir.glob(f"item_{item_idx}_generated_*.*"):
        m = re.search(r"_generated_(\d+)\.", p.name)
        if m:
            gen.append((int(m.group(1)), p))
    if gen:
        gen.sort(key=lambda t: t[0])
        return gen[0][1]
    return pick_standard_item_image(run_dir, item_idx)

def pick_sample_folder_image(sample_dir: Path, prefer_idx: int = 0, highest: bool = False) -> Optional[Path]:
    img_dir = sample_dir / "images"
    if not img_dir.exists():
        return None
    if highest:
        best = None
        best_i = -1
        for p in img_dir.glob("image_*.png"):
            m = re.match(r"image_(\d+)\.png$", p.name, re.I)
            if not m:
                continue
            i = int(m.group(1))
            if i > best_i:
                best_i = i
                best = p
        return best
    cand = img_dir / f"image_{prefer_idx}.png"
    if cand.exists():
        return cand
    imgs = []
    for p in img_dir.glob("image_*.png"):
        m = re.match(r"image_(\d+)\.png$", p.name, re.I)
        if m:
            imgs.append((int(m.group(1)), p))
    if imgs:
        imgs.sort(key=lambda t: t[0])
        return imgs[0][1]
    return None

# -----------------------------
# prediction extraction (sample-based)
# -----------------------------

TAG_RE_CACHE = {}

def extract_last_tag(text: str, tag: str) -> Optional[str]:
    """Return the LAST <tag>...</tag> content (case-insensitive), or None."""
    if not text:
        return None
    key = tag.lower()
    if key not in TAG_RE_CACHE:
        TAG_RE_CACHE[key] = re.compile(rf"<\s*{re.escape(tag)}\s*>\s*(.*?)\s*<\s*/\s*{re.escape(tag)}\s*>",
                                       re.IGNORECASE | re.DOTALL)
    matches = TAG_RE_CACHE[key].findall(text)
    if not matches:
        return None
    val = matches[-1].strip()
    return val if val else None

def extract_answer_like(text: str) -> Optional[str]:
    """Prefer <final_answer> then <answer>. If neither, return None."""
    v = extract_last_tag(text, "final_answer")
    if v is not None:
        return v
    return extract_last_tag(text, "answer")


def parse_pred_from_text(task: str, text: str) -> Tuple[Optional[int], Optional[str]]:
    # 1) Prefer explicit XML-ish tags if present
    tagged = extract_answer_like(text or "")
    if tagged:
        if task == "vpct":
            pred = extract_last_bucket_from_output(tagged) or to_int_123(tagged)
            return pred, None
        if task == "pathnav":
            lab = normalize_pathnav_label(tagged)
            return None, lab

    # 2) Otherwise fall back to previous behavior
    if task == "vpct":
        pred = extract_last_bucket_from_output(text or "") or to_int_123(text)
        return pred, None
    if task == "pathnav":
        lab = normalize_pathnav_label(text or "")
        return None, lab
    return None, None

def parse_pred_from_any_file(task: str, folder: Path) -> Tuple[Optional[int], Optional[str]]:
    
    # --- Prefer text_data.json if present (ThinkMorph / VilaSR) ---
    td = folder / "text_data.json"
    if td.exists():
        try:
            j = json.loads(td.read_text(encoding="utf-8"))
            # ViLaSR: {"response": "<answer>3</answer>"}
            if isinstance(j.get("response"), str):
                return parse_pred_from_text(task, j["response"])

            # ThinkMorph: {"text_outputs": [ ... "<answer>2</answer>" ] }
            if isinstance(j.get("text_outputs"), list) and j["text_outputs"]:
                last = j["text_outputs"][-1]
                if isinstance(last, str):
                    return parse_pred_from_text(task, last)
        except Exception:
            pass
        
        
    for jf in sorted(folder.glob("*.json")):
        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("answer", "final_answer", "prediction", "pred", "output", "response"):
            if key in j:
                val = j.get(key)
                if isinstance(val, (dict, list)):
                    val = json.dumps(val)
                pred_i, pred_lab = parse_pred_from_text(task, str(val))
                if pred_i is not None or pred_lab is not None:
                    return pred_i, pred_lab
        pred_i, pred_lab = parse_pred_from_text(task, jf.read_text(encoding="utf-8", errors="ignore"))
        if pred_i is not None or pred_lab is not None:
            return pred_i, pred_lab

    for tf in sorted(folder.glob("*.txt")):
        try:
            t = tf.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        pred_i, pred_lab = parse_pred_from_text(task, t)
        if pred_i is not None or pred_lab is not None:
            return pred_i, pred_lab

    return None, None

# -----------------------------
# thinkmorph/vilasr folder indexing
# -----------------------------

def extract_source_stem_from_sample_folder(folder_name: str) -> Optional[str]:
    i_sim = folder_name.find("sim_")
    i_maze = folder_name.find("maze_")
    i = -1
    if i_sim != -1 and i_maze != -1:
        i = min(i_sim, i_maze)
    elif i_sim != -1:
        i = i_sim
    elif i_maze != -1:
        i = i_maze
    if i == -1:
        return None
    return folder_name[i:]

def list_sample_folders(run_dir: Path) -> List[Path]:
    dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")]
    dirs.sort(key=lambda p: p.name)
    return dirs

def build_thinkmorph_index(run_dir: Path) -> Tuple[Dict[str, Path], Dict[str, int]]:
    stem_to_dir: Dict[str, Path] = {}
    stem_to_rank: Dict[str, int] = {}
    folders = list_sample_folders(run_dir)
    for i, d in enumerate(folders):
        stem = extract_source_stem_from_sample_folder(d.name)
        if stem:
            stem_to_dir[stem] = d
            stem_to_rank[stem] = i
    return stem_to_dir, stem_to_rank

# -----------------------------
# loaders
# -----------------------------


def _is_nanobanana_merged_list(obj) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    e0 = obj[0]
    return isinstance(e0, dict) and \
           "index" in e0 and "image_path" in e0 and "consistency_check_response" in e0

def find_nanobanana_merged_json(run_dir: Path) -> Optional[Path]:
    # Search run_dir then parent for a merged file
    for base in [run_dir, run_dir.parent]:
        if not base or not base.exists():
            continue
        for p in sorted(base.glob("*.json")):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if _is_nanobanana_merged_list(obj):
                return p
    return None

def load_run_dir_nanobanana_from_merged_json(task: str, run_dir: Path,
                                            vpct_ordered_imgs: Optional[List[str]] = None
                                            ) -> Dict[str, RunExample]:
    """
    NanoBanana merged format:
      - annotated image: entry["image_path"] (basename exists under run_dir)
      - answer source: entry["consistency_check_response"]
      - index aligns to dataset order
    Returns map keyed like the rest of the script:
      - VPCT: key = sim_XXX_initial.png basename (from vpct_ordered_imgs[index])
      - PathNav: key = item_idx (5-digit string)
    """
    merged = find_nanobanana_merged_json(run_dir)
    if merged is None:
        print(f"[nanobanana] No merged json found near {run_dir}")
        return {}

    try:
        rows = json.loads(merged.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[nanobanana] Failed reading {merged}: {e}")
        return {}

    out: Dict[str, RunExample] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        idx = r.get("index")
        if not isinstance(idx, int) or idx < 0:
            continue

        resp = r.get("consistency_check_response", "")
        if not isinstance(resp, str):
            resp = "" if resp is None else str(resp)

        pred_i, pred_lab = parse_pred_from_text(task, resp)

        img_path_raw = r.get("image_path", "")
        img_path = None
        if isinstance(img_path_raw, str) and img_path_raw:
            bname = Path(img_path_raw).name
            cand = run_dir / bname
            if cand.exists():
                img_path = cand
            elif Path(img_path_raw).exists():
                img_path = Path(img_path_raw)

        if task == "vpct":
            if not vpct_ordered_imgs or idx >= len(vpct_ordered_imgs):
                continue
            key = vpct_ordered_imgs[idx]          # e.g. "sim_100_initial.png"
            out[key] = RunExample(pred_int=pred_i, img_path=img_path, item_idx=f"{idx:05d}")
        else:
            key = f"{idx:05d}"                    # match pathnav’s item_idx keys
            out[key] = RunExample(pred_label=pred_lab, img_path=img_path, item_idx=key)

    print(f"[nanobanana] Loaded {len(out)} examples from {merged.name}")
    return out


def load_run_dir_vpct_standard_or_nanobanana(run_dir: Path, kind: str) -> Dict[str, RunExample]:
    out: Dict[str, RunExample] = {}
    for jf in sorted(run_dir.glob("item_*.json")):
        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        src = str(j.get("source_image", "") or j.get("image", "") or j.get("file", "")).replace("\\", "/")
        basename = Path(src).name
        if not SIM_IMG_RE.search(basename):
            continue

        pred = to_int_123(j.get("answer"))
        mo_full_any = j.get("model_output_full") or j.get("model_out_full") or ""
        mo_full_text = mo_full_any if isinstance(mo_full_any, str) else str(mo_full_any or "")
        if pred is None and mo_full_text.strip():
            pred = to_int_123(mo_full_text) or extract_last_bucket_from_output(mo_full_text)

        item_idx = None
        m = re.match(r"item_(\d+)\.json$", jf.name, re.I)
        if m:
            item_idx = m.group(1)

        img_path = None
        if item_idx:
            img_path = pick_nanobanana_item_image(run_dir, item_idx) if kind == "nanobanana" else pick_standard_item_image(run_dir, item_idx)

        out[basename] = RunExample(pred_int=pred, img_path=img_path, item_idx=item_idx)
    return out

def load_run_dir_pathnav_standard_or_nanobanana(run_dir: Path, kind: str) -> Dict[str, RunExample]:
    out: Dict[str, RunExample] = {}
    for jf in sorted(run_dir.glob("item_*.json")):
        m = re.match(r"item_(\d+)\.json$", jf.name, re.I)
        if not m:
            continue
        item_idx = m.group(1)

        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        src_raw = str(j.get("source_image", "") or "").replace("\\", "/")
        src_base = Path(src_raw).name if src_raw else None

        ans = j.get("answer")
        pred = normalize_pathnav_label(ans if isinstance(ans, str) else str(ans or ""))
        if pred is None:
            mo_full = j.get("model_output_full") or ""
            mo_full_text = mo_full if isinstance(mo_full, str) else str(mo_full or "")
            pred = normalize_pathnav_label(mo_full_text)

        img_path = pick_nanobanana_item_image(run_dir, item_idx) if kind == "nanobanana" else pick_standard_item_image(run_dir, item_idx)

        out[item_idx] = RunExample(
            pred_label=pred,
            img_path=img_path,
            item_idx=item_idx,
            source_image_raw=(src_raw or None),
            source_basename=(src_base or None),
        )
    return out

def load_run_dir_thinkmorph(task: str, run_dir: Path) -> Dict[str, RunExample]:
    out: Dict[str, RunExample] = {}
    stem_to_dir, _ = build_thinkmorph_index(run_dir)
    for stem, d in stem_to_dir.items():
        img_path = pick_sample_folder_image(d, prefer_idx=0, highest=False)
        if img_path is None:
            img_path = pick_sample_folder_image(d, prefer_idx=0, highest=True)
        pred_i, pred_lab = parse_pred_from_any_file(task, d)
        out[stem] = RunExample(pred_int=pred_i, pred_label=pred_lab, img_path=img_path, sample_dir=d)
    return out

def load_run_dir_vilasr(task: str, run_dir: Path, ref_stem_to_rank: Optional[Dict[str, int]] = None) -> Dict[str, RunExample]:
    folders = list_sample_folders(run_dir)
    out: Dict[str, RunExample] = {}
    if ref_stem_to_rank:
        for stem, rank in ref_stem_to_rank.items():
            if 0 <= rank < len(folders):
                d = folders[rank]
                img_path = pick_sample_folder_image(d, highest=True)
                if img_path is None:
                    img_path = pick_sample_folder_image(d, prefer_idx=0, highest=False)
                pred_i, pred_lab = parse_pred_from_any_file(task, d)
                out[stem] = RunExample(pred_int=pred_i, pred_label=pred_lab, img_path=img_path, sample_dir=d)
    return out

def resolve_source_image_path_from_ex(ex: Optional[RunExample], gt_root: Path) -> Optional[Path]:
    if ex and ex.source_image_raw:
        p = Path(ex.source_image_raw)
        if p.exists():
            return p
        p2 = Path(str(ex.source_image_raw).replace("\\", "/"))
        if p2.exists():
            return p2
        if ex.source_basename:
            p3 = gt_root / ex.source_basename
            if p3.exists():
                return p3
    return None


#----

def infer_vpct_order_from_item_jsons(run_dir: Path) -> List[str]:
    """
    Returns basenames like ['sim_100_initial.png', ...] ordered by item_XXXXX.json index.
    """
    pairs = []
    for jf in sorted(run_dir.glob("item_*.json")):
        m = re.match(r"item_(\d+)\.json$", jf.name, re.I)
        if not m:
            continue
        idx = int(m.group(1))
        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        src = str(j.get("source_image", "") or j.get("image", "") or j.get("file", "")).replace("\\", "/")
        base = Path(src).name
        if SIM_IMG_RE.search(base):
            pairs.append((idx, base))
    pairs.sort(key=lambda t: t[0])
    return [b for _, b in pairs]


# -----------------------------
# PDF drawing
# -----------------------------

def draw_image_fit(c: canvas.Canvas, img_path: Optional[Path], x: float, y: float, w: float, h: float, mode: str = "contain") -> None:
    """Draw image into (x,y,w,h).
    mode:
      - contain: preserve aspect ratio, fully visible (may letterbox)
      - fill   : preserve aspect ratio, fill box (may crop)
    """
    if not img_path or not img_path.exists():
        c.rect(x, y, w, h, stroke=1, fill=0)
        c.setFont("Times-Roman", 7)
        c.drawCentredString(x + w / 2, y + h / 2 - 3, "missing")
        return
    try:
        with Image.open(img_path) as im:
            iw, ih = im.size
        if iw <= 0 or ih <= 0:
            raise ValueError("bad image size")
    except Exception:
        c.rect(x, y, w, h, stroke=1, fill=0)
        c.setFont("Times-Roman", 7)
        c.drawCentredString(x + w / 2, y + h / 2 - 3, "missing")
        return

    mode = (mode or "contain").lower().strip()
    if mode not in ("contain", "fill"):
        mode = "contain"

    if mode == "contain":
        scale = min(w / iw, h / ih)
    else:
        scale = max(w / iw, h / ih)

    dw = iw * scale
    dh = ih * scale
    dx = x + (w - dw) / 2
    dy = y + (h - dh) / 2
    # NOTE: reportlab will happily draw outside the box; for fill-mode we rely on clipping.
    if mode == "fill":
        c.saveState()
        p = c.beginPath()
        p.rect(x, y, w, h)
        c.clipPath(p, stroke=0, fill=0)
        c.drawImage(
            ImageReader(str(img_path)),
            dx, dy,
            width=dw, height=dh,
            preserveAspectRatio=True,
            mask="auto",
        )
        c.restoreState()
    else:
        c.drawImage(ImageReader(str(img_path)), dx, dy, width=dw, height=dh, preserveAspectRatio=True, mask="auto")

def draw_label(c: canvas.Canvas, x_center: float, y: float, text: str, font_size: int) -> None:
    c.setFillColor(colors.black)
    c.setFont("Times-Roman", font_size)
    c.drawCentredString(x_center, y, text)


def wrap_header_text(title: str, cell_w: float, font_name: str, font_size: int, max_lines: int = 2) -> List[str]:
    """Wrap header text to fit within a column width.
    - If title contains '\n', respect manual line breaks.
    - Otherwise, greedily wrap by spaces up to max_lines.
    """
    if title is None:
        return [""]
    t = str(title)
    if "\n" in t:
        lines = [ln.strip() for ln in t.split("\n")]
        return [ln for ln in lines if ln] or [""]
    words = t.split()
    if not words:
        return [""]

    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        cand = cur + " " + w
        if stringWidth(cand, font_name, font_size) <= (cell_w * 0.98):
            cur = cand
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines - 1:
                break
    lines.append(cur)
    # If we still have leftover words, append ellipsis to last line.
    used = sum(len(ln.split()) for ln in lines)
    if used < len(words):
        last = lines[-1]
        if not last.endswith("…"):
            lines[-1] = (last + " …")
    return lines

def draw_header_cell(c: canvas.Canvas, x0: float, y_top: float, cell_w: float, header_h: float,
                     title: str, font_name: str, font_size: int, max_lines: int = 2, leading: Optional[float] = None) -> None:
    """Draw a (possibly multi-line) header centered inside the header band."""
    lines = wrap_header_text(title, cell_w, font_name, font_size, max_lines=max_lines)
    lead = leading if leading is not None else (font_size * 1.05)
    total_h = lead * len(lines)
    # vertical centering inside [y_top-header_h, y_top]
    y_start = (y_top - header_h) + (header_h - total_h) / 2 + (len(lines) - 1) * lead
    c.setFont(font_name, font_size)
    for i, ln in enumerate(lines):
        c.drawCentredString(x0 + cell_w / 2, y_start - i * lead, ln)
# -----------------------------
# figure construction
# -----------------------------

def make_pdf_vpct(out_pdf: Path, gt_root: Path, ordered_imgs: List[str], gt_map: Dict[str, int],
                  cols: List[Tuple[ColumnSpec, Dict[str, RunExample]]], picked_rows: List[int],
                  include_original: bool, show_ids: bool, original_title: str,
                  page: str, landscape_mode: bool, pt_per_inch: float,
                  margin_in: float, header_h_in: float, label_h_in: float, row_gap_in: float,
                  font_size: int, header_font_size: int, header_max_lines: int, header_leading: Optional[float], img_h_in: Optional[float], image_fit_mode: str) -> None:

    pagesize = letter if page.lower() == "letter" else A4
    pw, ph = pagesize
    if landscape_mode:
        pw, ph = ph, pw

    inch = float(pt_per_inch)
    margin = margin_in * inch
    header_h = header_h_in * inch
    label_h = label_h_in * inch
    row_gap = row_gap_in * inch

    col_titles = ([original_title] if include_original else []) + [cs.title for cs, _ in cols]
    ncols = len(col_titles)
    usable_w = pw - 2 * margin
    cell_w = usable_w / ncols

    target_img_h = (img_h_in * inch) if (img_h_in is not None) else (2.2 * inch)

    def rows_per_page() -> int:
        usable_h = ph - 2 * margin - header_h
        per_row = target_img_h + label_h + row_gap
        return max(1, int(usable_h // per_row)) if per_row > 0 else 1

    rpp = rows_per_page()
    c = canvas.Canvas(str(out_pdf), pagesize=(pw, ph))
    c.setTitle(out_pdf.stem)
    def draw_header():

        font_name = "Times-Bold"

        y_top = ph - margin

        for i, title in enumerate(col_titles):

            x0 = margin + i * cell_w

            draw_header_cell(

                c, x0=x0, y_top=y_top, cell_w=cell_w, header_h=header_h,

                title=title, font_name=font_name, font_size=header_font_size,

                max_lines=header_max_lines, leading=header_leading

            )



    def draw_page(page_rows: List[int]):
        draw_header()
        y_cursor = ph - margin - header_h

        for ridx in page_rows:
            img_name = ordered_imgs[ridx]
            stem = Path(img_name).stem
            gt = gt_map.get(img_name)
            gt_str = str(gt) if gt in (1, 2, 3) else "?"

            y_top = y_cursor
            y_img = y_top - target_img_h
            y_label = y_img - label_h

            col_i = 0
            if include_original:
                x = margin + col_i * cell_w
                draw_image_fit(c, gt_root / img_name, x, y_img, cell_w, target_img_h, mode=image_fit_mode)
                label = f"GT={gt_str}"
                if show_ids:
                    label = f"[{ridx}]  GT={gt_str}"
                draw_label(c, x + cell_w / 2, y_label + 0.02 * inch, label[:140], font_size)
                col_i += 1

            for cs, run_map in cols:
                x = margin + col_i * cell_w
                key = img_name if cs.kind in ("standard", "nanobanana") else stem
                ex = run_map.get(key)

                pred = ex.pred_int if ex else None
                pred_str = str(pred) if pred in (1, 2, 3) else "N/A"
                ok = (pred == gt) if (pred in (1, 2, 3) and gt in (1, 2, 3)) else None
                mark = "OK" if ok is True else ("X" if ok is False else "")
                pred_label = f"Pred={pred_str}" if not mark else f"Pred={pred_str} {mark}"

                draw_image_fit(c, ex.img_path if ex else None, x, y_img, cell_w, target_img_h, mode=image_fit_mode)
                draw_label(c, x + cell_w / 2, y_label + 0.02 * inch, pred_label, font_size)
                col_i += 1

            y_cursor = y_label - row_gap

    all_rows = [r for r in picked_rows if 0 <= r < len(ordered_imgs)]
    for start in range(0, len(all_rows), rpp):
        draw_page(all_rows[start:start + rpp])
        if start + rpp < len(all_rows):
            c.showPage()
    c.save()

def make_pdf_pathnav(out_pdf: Path, gt_root: Path, gt_fixed: str,
                     cols: List[Tuple[ColumnSpec, Dict[str, RunExample]]], picked_rows: List[int],
                     include_original: bool, show_ids: bool, original_title: str,
                     page: str, landscape_mode: bool, pt_per_inch: float,
                     margin_in: float, header_h_in: float, label_h_in: float, row_gap_in: float,
                     font_size: int, header_font_size: int, header_max_lines: int, header_leading: Optional[float], img_h_in: Optional[float], square_cells: bool, image_fit_mode: str) -> None:

    pagesize = letter if page.lower() == "letter" else A4
    pw, ph = pagesize
    if landscape_mode:
        pw, ph = ph, pw

    inch = float(pt_per_inch)
    margin = margin_in * inch
    header_h = header_h_in * inch
    label_h = label_h_in * inch
    row_gap = row_gap_in * inch

    col_titles = ([original_title] if include_original else []) + [cs.title for cs, _ in cols]
    ncols = len(col_titles)
    usable_w = pw - 2 * margin
    cell_w = usable_w / ncols

    if img_h_in is not None:
        target_img_h = img_h_in * inch
    else:
        target_img_h = cell_w if square_cells else (2.2 * inch)

    def rows_per_page() -> int:
        usable_h = ph - 2 * margin - header_h
        per_row = target_img_h + label_h + row_gap
        return max(1, int(usable_h // per_row)) if per_row > 0 else 1

    rpp = rows_per_page()
    c = canvas.Canvas(str(out_pdf), pagesize=(pw, ph))
    c.setTitle(out_pdf.stem)
    def draw_header():

        font_name = "Times-Bold"

        y_top = ph - margin

        for i, title in enumerate(col_titles):

            x0 = margin + i * cell_w

            draw_header_cell(

                c, x0=x0, y_top=y_top, cell_w=cell_w, header_h=header_h,

                title=title, font_name=font_name, font_size=header_font_size,

                max_lines=header_max_lines, leading=header_leading

            )



    def draw_page(rows: List[int]):
        draw_header()
        y_cursor = ph - margin - header_h

        ref_map = None
        for cs, m in cols:
            if cs.kind in ("standard", "nanobanana"):
                ref_map = m
                break

        for ridx in rows:
            item_idx = f"{ridx:05d}"
            y_top = y_cursor
            y_img = y_top - target_img_h
            y_label = y_img - label_h

            col_i = 0
            if include_original:
                x = margin + col_i * cell_w
                src_path = None
                if ref_map is not None:
                    ref_ex = ref_map.get(item_idx)
                    src_path = resolve_source_image_path_from_ex(ref_ex, gt_root)
                draw_image_fit(c, src_path, x, y_img, cell_w, target_img_h, mode=image_fit_mode)

                label = f"GT={gt_fixed}"
                if show_ids:
                    label = f"[{ridx}]  GT={gt_fixed}"
                draw_label(c, x + cell_w / 2, y_label + 0.02 * inch, label[:140], font_size)
                col_i += 1

            for cs, run_map in cols:
                x = margin + col_i * cell_w

                ex = None
                if cs.kind in ("standard", "nanobanana"):
                    ex = run_map.get(item_idx)
                elif cs.kind in ("thinkmorph", "vilasr") and ref_map is not None:
                    ref_ex = ref_map.get(item_idx)
                    if ref_ex and ref_ex.source_basename:
                        stem = Path(ref_ex.source_basename).stem
                        ex = run_map.get(stem)

                pred = ex.pred_label if (ex and ex.pred_label) else None
                pred_str = pred if pred in ("valid", "invalid") else "N/A"
                ok = (pred_str == gt_fixed) if pred_str in ("valid", "invalid") else None
                mark = "OK" if ok is True else ("X" if ok is False else "")
                pred_label = f"Pred={pred_str}" if not mark else f"Pred={pred_str} {mark}"

                draw_image_fit(c, ex.img_path if ex else None, x, y_img, cell_w, target_img_h, mode=image_fit_mode)
                draw_label(c, x + cell_w / 2, y_label + 0.02 * inch, pred_label, font_size)
                col_i += 1

            y_cursor = y_label - row_gap

    all_rows = [r for r in picked_rows if r >= 0]
    for start in range(0, len(all_rows), rpp):
        draw_page(all_rows[start:start + rpp])
        if start + rpp < len(all_rows):
            c.showPage()
    c.save()

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate a standalone PDF comparison figure (VPCT + PathNav + ThinkMorph/VilaSR/NanoBanana).")
    ap.add_argument("--task", choices=["vpct", "pathnav"], default="vpct")
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--gt-fixed", choices=["valid", "invalid"], default=None)

    ap.add_argument("--col-dir", type=Path, action="append", default=[])
    ap.add_argument("--col-title", type=str, action="append", default=[])
    ap.add_argument("--col-kind", type=str, action="append", default=[],
                    help=f"Repeatable. One of: {', '.join(COL_KINDS)}. Default: standard")

    ap.add_argument("--rows", type=str, default="")
    ap.add_argument("--max-rows", type=int, default=12)
    ap.add_argument("--out-pdf", type=Path, required=True)

    ap.add_argument("--no-original", action="store_true")
    ap.add_argument("--no-ids", action="store_true")

    ap.add_argument("--page", choices=["letter", "a4"], default="letter")
    ap.add_argument("--landscape", action="store_true")

    ap.add_argument("--margin-in", type=float, default=0.15)
    ap.add_argument("--header-h-in", type=float, default=0.16)
    ap.add_argument("--label-h-in", type=float, default=0.10)
    ap.add_argument("--row-gap-in", type=float, default=0.00)

    ap.add_argument("--font-size", type=int, default=7)
    ap.add_argument("--header-font-size", type=int, default=9)
    ap.add_argument("--header-max-lines", type=int, default=2, help="Max header lines (supports manual \\n in titles).")
    ap.add_argument("--header-leading", type=float, default=None, help="Header line spacing in points; default ~1.05*font size.")
    ap.add_argument("--image-fit", choices=["contain", "fill"], default="contain", help="How to fit images inside cells.")
    ap.add_argument("--pt-per-inch", type=float, default=45.0)

    ap.add_argument("--img-h-in", type=float, default=None)
    ap.add_argument("--no-square-cells", action="store_true")

    args = ap.parse_args()

    if not args.col_dir:
        raise SystemExit("Need at least one --col-dir")

    titles = list(args.col_title or [])
    while len(titles) < len(args.col_dir):
        titles.append(args.col_dir[len(titles)].name)
    titles = titles[:len(args.col_dir)]

    kinds = list(args.col_kind or [])
    while len(kinds) < len(args.col_dir):
        kinds.append("standard")
    kinds = [k.strip().lower() for k in kinds[:len(args.col_dir)]]
    for k in kinds:
        if k not in COL_KINDS:
            raise SystemExit(f"Unknown --col-kind {k}. Must be one of: {', '.join(COL_KINDS)}")

    picked = parse_rows_spec(args.rows)
    if picked is None:
        picked = list(range(args.max_rows))

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    col_specs = [ColumnSpec(title=t, run_dir=d, kind=k) for t, d, k in zip(titles, args.col_dir, kinds)]

    thinkmorph_rank = None
    for cs in col_specs:
        if cs.kind == "thinkmorph":
            _, thinkmorph_rank = build_thinkmorph_index(cs.run_dir)
            break

    if args.task == "vpct":
        ordered, gt_map = load_gt_vpct(args.gt_root)
        
        # Prefer dataset/item order (from item_*.json) so row indices match item_00000, item_00001, ...
        for cs in col_specs:
            if cs.kind == "standard":
                inferred = infer_vpct_order_from_item_jsons(cs.run_dir)
                if inferred:
                    ordered = inferred
                break

        if not ordered:
            raise SystemExit(f"No VPCT GT found in {args.gt_root} (expected sim_*_results.json)")

        cols_loaded = []
        for cs in col_specs:
            if cs.kind == "nanobanana":
                cols_loaded.append((cs, load_run_dir_nanobanana_from_merged_json("vpct", cs.run_dir, vpct_ordered_imgs=ordered)))
            elif cs.kind == "standard":
                cols_loaded.append((cs, load_run_dir_vpct_standard_or_nanobanana(cs.run_dir, cs.kind)))
            elif cs.kind == "thinkmorph":
                cols_loaded.append((cs, load_run_dir_thinkmorph("vpct", cs.run_dir)))
                if thinkmorph_rank is None:
                    _, thinkmorph_rank = build_thinkmorph_index(cs.run_dir)
            elif cs.kind == "vilasr":
                cols_loaded.append((cs, load_run_dir_vilasr("vpct", cs.run_dir, ref_stem_to_rank=thinkmorph_rank)))
            else:
                raise SystemExit(f"Unhandled kind: {cs.kind}")


        make_pdf_vpct(args.out_pdf, args.gt_root, ordered, gt_map, cols_loaded, picked,
              include_original=(not args.no_original),
              show_ids=(not args.no_ids),
              original_title="Source Image",
              page=args.page,
              landscape_mode=args.landscape,
              pt_per_inch=float(args.pt_per_inch),
              margin_in=float(args.margin_in),
              header_h_in=float(args.header_h_in),
              label_h_in=float(args.label_h_in),
              row_gap_in=float(args.row_gap_in),
              font_size=int(args.font_size),
              header_font_size=int(args.header_font_size),
              header_max_lines=int(args.header_max_lines),
              header_leading=args.header_leading,
              img_h_in=args.img_h_in,
              image_fit_mode=args.image_fit)

        print(f"Wrote PDF: {args.out_pdf}")
        return

    if not args.gt_fixed:
        raise SystemExit("--task pathnav requires --gt-fixed valid|invalid")

    cols_loaded2 = []
    for cs in col_specs:
        if cs.kind == "nanobanana":
            cols_loaded2.append((cs, load_run_dir_nanobanana_from_merged_json("pathnav", cs.run_dir)))
        elif cs.kind == "standard":
            cols_loaded2.append((cs, load_run_dir_pathnav_standard_or_nanobanana(cs.run_dir, cs.kind)))
        elif cs.kind == "thinkmorph":
            cols_loaded2.append((cs, load_run_dir_thinkmorph("pathnav", cs.run_dir)))
            if thinkmorph_rank is None:
                _, thinkmorph_rank = build_thinkmorph_index(cs.run_dir)
        elif cs.kind == "vilasr":
            cols_loaded2.append((cs, load_run_dir_vilasr("pathnav", cs.run_dir, ref_stem_to_rank=thinkmorph_rank)))
        else:
            raise SystemExit(f"Unhandled kind: {cs.kind}")


    make_pdf_pathnav(args.out_pdf, args.gt_root, args.gt_fixed, cols_loaded2, picked,
                 include_original=(not args.no_original),
                 show_ids=(not args.no_ids),
                 original_title="Source Image",
                 page=args.page,
                 landscape_mode=args.landscape,
                 pt_per_inch=float(args.pt_per_inch),
                 margin_in=float(args.margin_in),
                 header_h_in=float(args.header_h_in),
                 label_h_in=float(args.label_h_in),
                 row_gap_in=float(args.row_gap_in),
                 font_size=int(args.font_size),
                 header_font_size=int(args.header_font_size),
                 header_max_lines=int(args.header_max_lines),
                 header_leading=args.header_leading,
                 img_h_in=args.img_h_in,
                 square_cells=(not args.no_square_cells),
                 image_fit_mode=args.image_fit)

    print(f"Wrote PDF: {args.out_pdf}")

if __name__ == "__main__":
    main()


    
'''

##VPCT - working

python compare_figure_pdf.py --task vpct --gt-root vpct-1 --col-dir results/mix_eval/gemini3pro_vpct_no_grid_0_to_1000 --col-title "Gemini-3-Pro No Grid" --col-dir results/mix_eval/gem3pro_vpct_multi_withtextstrokes --col-title "Gemini-3-Pro Multi-turn" --col-dir results/mix_eval/vpct_ball_gpt5low --col-title "GPT-5 (low) With Grid" --col-dir results/mix_eval/gpt5low_vpct_multiturn --col-title "GPT-5 (low) Multi-turn" --rows "0,3,5,8,14,17" --no-ids --margin-in 0.15 --header-h-in 0.16 --label-h-in 0.10 --row-gap-in 0 --pt-per-inch 45 --out-pdf fig_vpct_compare.pdf 



python compare_figure_pdf.py --gt-root vpct-1 `
  --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro No Grid" `
  --col-dir results/mix_eval/gem3pro_vpct_multi_withtextstrokes --col-title "Gemini-3-Pro Multi-turn" `
  --col-dir results/mix_eval/vpct_ball_gpt5low --col-title "GPT-5 (low) With Grid" `
  --rows "0,3,5,9" --no-ids `
  --margin-in 0.15 --header-h-in 0.16 --label-h-in 0.10 --row-gap-in 0 `
  --pt-per-inch 55 `
  --out-pdf fig_vpct_compare.pdf



## Path Navigation 

python compare_figure_pdf.py --task pathnav --gt-root datasets/maze_v2/sketch_valid_flattened --gt-fixed valid --col-dir results/mix_eval/gemini3pro_gridworld_validpaths_0_to_1000 --col-title "Gemini-3-Pro Single Turn" --col-dir results/mix_eval/20260115_131741_gem3_multi_valid_validity_answers --col-title "Gemini-3-Pro Multi-turn" --col-dir results/mix_eval/20260123_232940_gpt_maze_valid_validity_answers --col-title "GPT-5 (low) Multi-turn" --rows "12,13,15,17" --no-ids --margin-in 0.15 --header-h-in 0.16 --label-h-in 0.10 --row-gap-in 0 --pt-per-inch 80 --out-pdf fig_pathnav_compare.pdf

python compare_figure_pdf.py `
  --task pathnav `
  --gt-root datasets/maze_v2/sketch_valid_flattened `
  --gt-fixed valid `
  --col-dir results/mix_eval/gemini3pro_gridworld_validpaths_0_to_1000 --col-title "Gemini-3-Pro Single Turn" `
  --col-dir results/mix_eval/20260115_131741_gem3_multi_valid_validity_answers --col-title "GPT-5 Multi-turn" `
  --rows "10,11,12,14" `
  --no-ids `
  --pt-per-inch 45 `
  --out-pdf fig_pathnav_compare.pdf




--margin-in 0.15 --header-h-in 0.16 --label-h-in 0.10 --row-gap-in 0 --pt-per-inch 45

python compare_figure_pdf.py --task vpct --gt-root vpct-1 --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro" --col-kind standard --col-dir results/mix_eval/vpct_nanobanana_sketch --col-title "NanoBanana" --col-kind nanobanana --col-dir results/mix_eval/vpct_thinkmorph --col-title "ThinkMorph" --col-kind thinkmorph --col-dir results/mix_eval/vpct_vilasr --col-title "VilaSR" --col-kind vilasr --rows "0,3,5,9" --no-ids --margin-in 0.15 --header-h-in 0.16 --label-h-in 0.10 --row-gap-in 0 --pt-per-inch 45 --out-pdf fig_vpct_compare_2.pdf

## USE BELOW -- VPCT &&&&&
python compare_figure_pdf.py --task vpct --gt-root vpct-1 --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro" --col-kind standard --col-dir results/mix_eval/gem3pro_vpct_multi_withtextstrokes --col-title "Gemini-3-Pro Multi-turn" --col-kind standard --col-dir results/mix_eval/vpct_ball_gpt5low --col-title "GPT-5 (low) With Grid" --col-kind standard --col-dir results/mix_eval/gpt5low_vpct_multiturn --col-title "GPT-5 (low) Multi-turn" --col-kind standard --col-dir results/mix_eval/vpct_nanobanana_sketch --col-title "NanoBanana" --col-kind nanobanana --col-dir results/mix_eval/vpct_thinkmorph --col-title "ThinkMorph" --col-kind thinkmorph --col-dir results/mix_eval/vpct_vilasr --col-title "ViLaSR" --col-kind vilasr --rows "0,2,4,5,6,7,8,11,12,13,14" --no-ids --margin-in 0.15 --header-h-in 0.36 --label-h-in 0.10 --row-gap-in 0 --pt-per-inch 30 --header-font-size 8 --header-max-lines 2 --out-pdf fig_vpct_compare_3.pdf

--col-dir results/mix_eval/gem3pro_vpct_multi_withtextstrokes --col-title "Gemini-3-Pro Multi-turn" --col-kind standard --col-dir results/mix_eval/vpct_ball_gpt5low --col-title "GPT-5 (low) With Grid" --col-kind standard --col-dir results/mix_eval/gpt5low_vpct_multiturn --col-title "GPT-5 (low) Multi-turn" --col-kind standard


python compare_figure_pdf.py --task vpct \
  --gt-root vpct-1 \
  --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro" --col-kind standard \
  --col-dir results/mix_eval/vpct_nanobanana_sketch --col-title "NanoBanana" --col-kind nanobanana \
  --col-dir results/mix_eval/vpct_thinkmorph --col-title "ThinkMorph" --col-kind thinkmorph \
  --col-dir results/mix_eval/vpct_vilasr --col-title "VilaSR" --col-kind vilasr \
  --rows "0,3,5,9" --no-ids --margin-in 0.15 --header-h-in 0.16 --label-h-in 0.10 --row-gap-in 0 --pt-per-inch 45 \
  --out-pdf fig_vpct_compare_2.pdf



python compare_figure_pdf.py --task pathnav --gt-root datasets/maze_v2/sketch_valid_flattened --gt-fixed valid --col-dir results/mix_eval/gemini3pro_gridworld_validpaths_0_to_1000 --col-title "Gemini-3-Pro" --col-kind standard --col-dir results/mix_eval/20260123_232940_gpt_maze_valid_validity_answers --col-title "GPT-5 (low)" --col-kind standard --col-dir results/mix_eval/mazev2_other_models/nano_banana/nanob_maze_valid --col-title "NanoBanana" --col-kind nanobanana --col-dir results/mix_eval/mazev2_other_models/thinkmorph/thinkmorph_valid --col-title "ThinkMorph" --col-kind thinkmorph --col-dir results/mix_eval/mazev2_other_models/vilasr/vilasr_valid --col-title "ViLaSR" --col-kind vilasr --rows "10,11,12,13" --no-ids --pt-per-inch 70 --out-pdf fig_pathnav_compare.pdf

## USE BELOW -- Path Navigation &&&&&
python compare_figure_pdf.py --task pathnav --gt-root datasets/maze_v2/sketch_valid_flattened --gt-fixed valid --col-dir results/mix_eval/gemini3pro_gridworld_validpaths_0_to_1000 --col-title "Gemini-3-Pro" --col-kind standard --col-dir results/mix_eval/20260123_232940_gpt_maze_valid_validity_answers --col-title "GPT-5 (low)" --col-kind standard --col-dir results/mix_eval/mazev2_other_models/nano_banana/nanob_maze_valid --col-title "NanoBanana" --col-kind nanobanana --col-dir results/mix_eval/mazev2_other_models/thinkmorph/thinkmorph_valid --col-title "ThinkMorph" --col-kind thinkmorph --col-dir results/mix_eval/mazev2_other_models/vilasr/vilasr_valid --col-title "ViLaSR" --col-kind vilasr --rows "110,111,112,113" --no-ids --pt-per-inch 70 --out-pdf fig_pathnav_compare.pdf

python compare_figure_pdf.py --task pathnav \
  --gt-root datasets/maze_v2/sketch_valid_flattened --gt-fixed valid \
  --col-dir results/mix_eval/gemini3pro_gridworld_validpaths_0_to_1000 --col-title "Gemini-3-Pro" --col-kind standard \
  --col-dir results/mix_eval/20260123_232940_gpt_maze_valid_validity_answers --col-title "GPT-5 (low)" --col-kind standard \
  --col-dir results/mix_eval/mazev2_other_models/nano_banana/nanob_maze_valid --col-title "NanoBanana" --col-kind nanobanana \
  --col-dir results/mix_eval/mazev2_other_models/thinkmorph/thinkmorph_valid --col-title "ThinkMorph" --col-kind thinkmorph \
  --col-dir results/mix_eval/mazev2_other_models/vilasr/vilasr_valid --col-title "ViLaSR" --col-kind vilasr \
  --rows "10,11,12,13" --no-ids --pt-per-inch 45 \
  --out-pdf fig_pathnav_compare.pdf





How to use the new options
1) Remove the row whitespace

Add this:

--image-fit fill


Example (your VPCT-style run):

python compare_figure_pdf_unified3.py --task vpct --gt-root vpct-1 \
  ... \
  --pt-per-inch 45 \
  --image-fit fill \
  --out-pdf fig_vpct_compare.pdf

2) Two-line headers / smaller headers

Option A (auto wrap):

--header-font-size 8 --header-max-lines 2


Option B (manual split):

--col-title "Gemini-3-Pro\nMulti-turn"


If you want tighter/looser spacing between the header lines:

--header-leading 8.5

'''
