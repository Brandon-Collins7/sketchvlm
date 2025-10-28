#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ball Drop Task: Compare Ground Truth with Grid Sketch results (one or two),
optionally also including a Raw Baseline.

Backwards compatible:
- Legacy single-grid mode with raw baseline still works.
New features:
- Dual-grid mode: --grid-dir (A) and --grid-dir-b (B) shown side by side.
- --grid-a-name/--grid-b-name let you label columns.
- --no-raw hides the Raw column and removes raw from the summary.
- raw jsonl becomes optional if --no-raw or --grid-dir-b is provided.

Row alignment:
- Rows are keyed by actual run basename like run_001.png or run_001_2.png.
- Grid json item_* must include source_image (or similar) pointing to that file.

Examples
--------
# Legacy (single grid + raw)
python ball_drop_compare_report.py \
  --gt-root datasets/ball_physics_orig \
  --raw-jsonl logs_buckets_boxed.jsonl \
  --grid-dir results/mix_eval/meta_ball_drop \
  --out ball_drop_report.html \
  --thumb-width 480

# Dual grid (no raw)
python ball_drop_compare_report.py \
  --gt-root large_run_split \
  --grid-dir results/mix_eval/straight_lines \
  --grid-dir-b results/mix_eval/curves_and_lines \
  --grid-a-name "Straight lines only" \
  --grid-b-name "Curves + lines" \
  --no-raw \
  --out meta_ball_better.html \
  --thumb-width 480
  
  
  
  python ball_drop_compare_report.py --gt-root large_run_split --raw-jsonl results/mix_eval/ball_number_gpt5_medium.jsonl --grid-dir results/mix_eval/gpt_ball_drop_curves --out gpt_ball_drop_report.html --thumb-width 480

  python ball_drop_compare_report.py --gt-root large_run_split --raw-jsonl results/mix_eval/vpct_ball_gpt5_high.jsonl --grid-dir results/mix_eval/vpct_ball_gpt5high --out gpt_high_ball_drop_report.html --thumb-width 480

"""

import re
import io
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from collections import defaultdict


from PIL import Image

# --------- Prompts to display if not available in files ---------
GRID_SKETCH_PROMPT_FALLBACK: str = (
    "Draw the path that the ball will take. It will either land in container 1, 2, 3, or 4, "
    "or it will get stuck before then. The ball is released from rest, the only force it is subject to is gravity. "
    "The ball cannot be compressed so it must not be too big to fit between gaps. The black lines are walls and platforms. "
    "The path cannot pass through the black lines, and the ball cannot pass through the black lines. "
    "After emitting the strokes, put your answer of which container it will land in, 1, 2, 3, or 4, "
    "or 0 if it gets stuck before reaching any."
)

RAW_BASELINE_PROMPT_FALLBACK: str = (
    "The ball is released from rest, the only force it is subject to is gravity. "
    "The ball cannot be compressed so it must not be too big to fit between gaps. "
    "The dark lines are walls and platforms. The path cannot pass through the lines, "
    "and the ball cannot pass through the lines. "
    "Put your answer of which container it will land in, 1, 2, 3, or 4, or 0 if it gets stuck before reaching any. "
    "Only return the number answer."
)

# Accept optional suffix like _1, _2 after the 3-digit id
RUN_DIR_RE  = re.compile(r"^run_(\d{3})(?:_(\d+))?$", re.I)
RUN_PNG_RE  = re.compile(r"(run_(\d{3})(?:_(\d+))?)\.(?:png|jpg|jpeg|webp|bmp)$", re.I)
RUN_ANY_RE  = re.compile(r"(run_(\d{3})(?:_(\d+))?)", re.I)

def _esc(s: Optional[str]) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _label_to_int(v: Any) -> Optional[int]:
    """Map 1..4 or '1'..'4' to int; map 'none'/0/null -> 0 for GT. For preds, None if unparsable."""
    if v is None:
        return 0  # GT null -> 0 ('none')
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            iv = int(v)
            if iv in (0,1,2,3,4):
                return iv
        except Exception:
            return None
    s = str(v).strip().lower()
    if s in {"none", "null"}:
        return 0
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    iv = int(m.group(0))
    return iv if iv in (0,1,2,3,4) else None

# ----------------- Loaders -----------------
def load_gt(gt_root: Path) -> Dict[str, Dict[str, int]]:
    """
    Return mapping:
      'run_###[_v].png' -> {'gt': int(0..4), 'num_lines': int or 0}

    Scans run folders and reads random_scene_metadata.json, pulling:
      - simulation.bucket_hit -> gt (0..4 where 0 means 'none')
      - simulation.num_lines  -> number of lines in the scene (expected 1,2,3)
    """
    out: Dict[str, Dict[str, int]] = {}
    for d in sorted(gt_root.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if not m:
            continue
        meta = d / "random_scene_metadata.json"
        if not meta.exists():
            continue
        try:
            j = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue

        sim = j.get("simulation") if isinstance(j, dict) else None
        bucket = sim.get("bucket_hit") if isinstance(sim, dict) else None
        iv = _label_to_int(bucket)
        if iv is None:
            iv = 0  # default to 'none'

        nl = 0
        try:
            nl_val = (sim or {}).get("num_lines", 0)
            nl = int(nl_val) if nl_val is not None else 0
        except Exception:
            nl = 0

        out[f"{d.name}.png"] = {"gt": iv, "num_lines": nl}
    return out


def load_raw_jsonl(raw_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    """
    Return mapping keyed by basename 'run_###[_v].png' -> {'pred': int|None, 'raw_text': str, 'prompt': str}
    We extract run from rec['file'] allowing optional variant suffix.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not raw_jsonl or not raw_jsonl.exists():
        return out
    with raw_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            file_field = str(rec.get("file", "")).replace("\\","/")
            base = Path(file_field).name
            m = RUN_PNG_RE.match(base)
            if not m:
                mm = RUN_ANY_RE.search(file_field)
                if not mm:
                    continue
                base = f"{mm.group(1)}.png"
            else:
                base = m.group(1) + ".png"
            pred = rec.get("parsed_int", None)
            pred = _label_to_int(pred)
            if pred is None:
                pred = _label_to_int(rec.get("parsed_label"))
            out[base] = {
                "pred": pred,
                "raw_text": rec.get("raw_text", ""),
                "prompt": rec.get("prompt", ""),
            }
    return out

def load_grid(grid_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Return mapping keyed by basename 'run_###[_v].png' -> {
        'pred': int|None, 'annot_path': Path|None, 'orig_path': Path|None, 'prompt': str
    }
    We derive the run from item_#####.json -> "source_image" path (preferred).
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not grid_dir or not grid_dir.exists():
        return out
    for jf in sorted(grid_dir.glob("item_*.json")):
        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            j = {}
        key = None
        src = (j.get("source_image") or j.get("raw_image") or j.get("source_prompt") or "")
        src = str(src).replace("\\","/")
        if src:
            b = Path(src).name
            m = RUN_PNG_RE.match(b)
            if m:
                key = m.group(1) + ".png"
            else:
                mm = RUN_ANY_RE.search(src)
                if mm:
                    key = f"{mm.group(1)}.png"
        if key is None:
            m = re.match(r"item_(\d{5})\.json$", jf.name, re.I)
            if m:
                idx5 = m.group(1)
                key = f"run_{int(idx5)+1:03d}.png"
        if key is None:
            continue

        ans = j.get("answer", None)
        pred = _label_to_int(ans)

        annot = orig = None
        m2 = re.match(r"item_(\d{5})\.json$", jf.name, re.I)
        if m2:
            idx5 = m2.group(1)
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                cand = grid_dir / f"item_{idx5}_annotated{ext}"
                if cand.exists():
                    annot = cand; break
            for ext in (".jpg", ".png", ".jpeg", ".webp"):
                cand = grid_dir / f"item_{idx5}_orig{ext}"
                if cand.exists():
                    orig = cand; break

        out[key] = {
            "pred": pred,
            "annot_path": annot,
            "orig_path": orig,
            "prompt": j.get("prompt", ""),
        }
    return out

# ----------------- HTML helpers -----------------
def _img_to_data_uri(path: Path, max_width: int = 700) -> Optional[str]:
    try:
        with Image.open(path) as im:
            is_png = path.suffix.lower() == ".png"
            im = im.convert("RGBA" if is_png else "RGB")
            if max_width and im.width > max_width:
                nh = max(1, int(im.height * (max_width / float(im.width))))
                im = im.resize((max_width, nh), Image.LANCZOS)
            buf = io.BytesIO()
            if is_png:
                im.save(buf, format="PNG"); mime = "image/png"
            else:
                im = im.convert("RGB"); im.save(buf, format="JPEG", quality=92, subsampling=0); mime = "image/jpeg"
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def _sort_key_runpng(name: str) -> Tuple[int, int, str]:
    nm = Path(name).name
    m = RUN_PNG_RE.match(nm)
    if m:
        rid = int(m.group(2))
        var = int(m.group(3)) if m.group(3) is not None else 0
        return (rid, var, nm)
    mm = RUN_ANY_RE.search(nm)
    if mm:
        rid = int(mm.group(2))
        var = int(mm.group(3)) if mm.group(3) is not None else 0
        return (rid, var, nm)
    return (10**9, 10**9, nm)

# ----------------- Report builder -----------------
def build_report(
    gt_root: Path,
    raw_jsonl: Optional[Path],
    grid_dir_a: Path,
    out_html: Path,
    thumb_width: int = 480,
    *,
    grid_dir_b: Optional[Path] = None,
    grid_a_name: str = "Grid A",
    grid_b_name: str = "Grid B",
    no_raw: bool = False,
) -> None:
    gt_map = load_gt(gt_root)  # now: run_png -> {'gt': int, 'num_lines': int}
    raw  = load_raw_jsonl(raw_jsonl) if (raw_jsonl and not no_raw) else {}
    gridA = load_grid(grid_dir_a)
    gridB = load_grid(grid_dir_b) if grid_dir_b else {}

    runs = sorted(gt_map.keys(), key=_sort_key_runpng)

    # Accumulators (overall)
    raw_tot = raw_correct = 0
    a_tot = a_correct = 0
    b_tot = b_correct = 0

    # Per-line accumulators for A and B
    a_tot_by_nl = defaultdict(int)
    a_cor_by_nl = defaultdict(int)
    b_tot_by_nl = defaultdict(int)
    b_cor_by_nl = defaultdict(int)

    rows: List[str] = []

    for run_png in runs:
        info = gt_map.get(run_png, {"gt": 0, "num_lines": 0})
        gt_val = info.get("gt", 0)
        num_lines = int(info.get("num_lines", 0))

        raw_info  = raw.get(run_png, {})
        a_info    = gridA.get(run_png, {})
        b_info    = gridB.get(run_png, {}) if gridB else {}

        raw_pred  = raw_info.get("pred", None)
        raw_text  = raw_info.get("raw_text", "")
        raw_prompt= raw_info.get("prompt", "") or RAW_BASELINE_PROMPT_FALLBACK

        a_pred    = a_info.get("pred", None)
        a_annot   = a_info.get("annot_path", None)
        a_orig    = a_info.get("orig_path", None)
        a_prompt  = a_info.get("prompt", "") or GRID_SKETCH_PROMPT_FALLBACK

        b_pred    = b_info.get("pred", None)
        b_annot   = b_info.get("annot_path", None)
        b_orig    = b_info.get("orig_path", None)
        b_prompt  = b_info.get("prompt", "") or GRID_SKETCH_PROMPT_FALLBACK

        # Overall accuracies
        if not no_raw and raw_pred is not None:
            raw_tot += 1
            if raw_pred == gt_val:
                raw_correct += 1
        if a_pred is not None:
            a_tot += 1
            if a_pred == gt_val:
                a_correct += 1
            # Per-line tracking only for 1,2,3 as requested
            if num_lines in (1, 2, 3):
                a_tot_by_nl[num_lines] += 1
                if a_pred == gt_val:
                    a_cor_by_nl[num_lines] += 1
        if gridB and b_pred is not None:
            b_tot += 1
            if b_pred == gt_val:
                b_correct += 1
            if num_lines in (1, 2, 3):
                b_tot_by_nl[num_lines] += 1
                if b_pred == gt_val:
                    b_cor_by_nl[num_lines] += 1

        # --- build row (unchanged except gt access) ---
        gt_cell = f'<div class="metric"><span class="label">GT:</span> <span class="val">{gt_val}</span></div>'

        # ORIGINAL (prefer A's orig, else B's, else GT fallback)
        orig_path = a_orig or b_orig
        if isinstance(orig_path, Path) and orig_path.exists():
            orig_uri = _img_to_data_uri(orig_path, max_width=thumb_width)
            orig_cell = f'<img src="{orig_uri}" />' if orig_uri else '<div class="missing">failed to embed</div>'
        else:
            base_noext = Path(run_png).stem
            candidate = gt_root / base_noext / f"{base_noext}.png"
            if candidate.exists():
                furi = _img_to_data_uri(candidate, max_width=thumb_width)
                orig_cell = f'<img src="{furi}" />' if furi else '<div class="missing">failed to embed</div>'
            else:
                candidate2 = gt_root / (base_noext + ".png")
                if candidate2.exists():
                    furi = _img_to_data_uri(candidate2, max_width=thumb_width)
                    orig_cell = f'<img src="{furi}" />' if furi else '<div class="missing">failed to embed</div>'
                else:
                    orig_cell = '<div class="missing">missing original</div>'

        # RAW cell (optional)
        if no_raw:
            raw_cell = None
        else:
            if raw_pred is None:
                raw_cell = '<div class="metric"><span class="label">Raw:</span> <span class="val">–</span> <span class="na">–</span></div>'
            else:
                badge = '<span class="ok">✓</span>' if raw_pred == gt_val else '<span class="bad">✗</span>'
                raw_cell = f'<div class="metric"><span class="label">Raw:</span> <span class="val">{raw_pred}</span> {badge}</div>'
            if raw_text:
                raw_cell += f'<details class="rawtxt"><summary>raw_text</summary><pre>{_esc(raw_text)}</pre></details>'

        # GRID A cell
        if isinstance(a_annot, Path) and a_annot.exists():
            auri = _img_to_data_uri(a_annot, max_width=thumb_width)
            a_img = f'<img src="{auri}" />' if auri else '<div class="missing">failed to embed</div>'
        else:
            a_img = '<div class="missing">missing annotated</div>'
        if a_pred is None:
            a_cell = a_img + f'<div class="metric"><span class="label">{_esc(grid_a_name)}:</span> <span class="val">–</span> <span class="na">–</span></div>'
        else:
            abadge = '<span class="ok">✓</span>' if a_pred == gt_val else '<span class="bad">✗</span>'
            a_cell = a_img + f'<div class="metric"><span class="label">{_esc(grid_a_name)}:</span> <span class="val">{a_pred}</span> {abadge}</div>'

        # GRID B cell (if provided)
        if gridB:
            if isinstance(b_annot, Path) and b_annot.exists():
                buri = _img_to_data_uri(b_annot, max_width=thumb_width)
                b_img = f'<img src="{buri}" />' if buri else '<div class="missing">failed to embed</div>'
            else:
                b_img = '<div class="missing">missing annotated</div>'
            if b_pred is None:
                b_cell = b_img + f'<div class="metric"><span class="label">{_esc(grid_b_name)}:</span> <span class="val">–</span> <span class="na">–</span></div>'
            else:
                bbadge = '<span class="ok">✓</span>' if b_pred == gt_val else '<span class="bad">✗</span>'
                b_cell = b_img + f'<div class="metric"><span class="label">{_esc(grid_b_name)}:</span> <span class="val">{b_pred}</span> {bbadge}</div>'
        else:
            b_cell = None

        # Prompts cell (collapsible)
        prompt_bits = []
        if not no_raw:
            prompt_bits.append(f'<details><summary>show raw baseline prompt</summary><pre>{_esc(raw_prompt)}</pre></details>')
        prompt_bits.append(f'<details><summary>show {_esc(grid_a_name)} prompt</summary><pre>{_esc(a_prompt)}</pre></details>')
        if gridB:
            prompt_bits.append(f'<details><summary>show {_esc(grid_b_name)} prompt</summary><pre>{_esc(b_prompt)}</pre></details>')
        prompts_cell = "".join(prompt_bits)

        idx_label = Path(run_png).stem

        # Build row with dynamic columns
        tds = [
            f'<td class="idx">{idx_label}.png</td>',
            f'<td class="prompt">{prompts_cell}</td>',
            f'<td class="gt">{gt_cell}</td>',
        ]
        if not no_raw:
            tds.append(f'<td class="raw">{raw_cell}</td>')
        tds.append(f'<td class="orig">{orig_cell}</td>')
        tds.append(f'<td class="grid">{a_cell}</td>')
        if b_cell is not None:
            tds.append(f'<td class="grid">{b_cell}</td>')

        rows.append("<tr>\n  " + "\n  ".join(tds) + "\n</tr>\n")

    # ---- Summary (overall + per-line) ----
    parts = []
    if not no_raw:
        raw_acc = (raw_correct / raw_tot * 100.0) if raw_tot else 0.0
        parts.append(f"<div><b>Raw baseline:</b> {raw_acc:.1f}% ({raw_correct}/{raw_tot})</div>")

    a_acc = (a_correct / a_tot * 100.0) if a_tot else 0.0
    parts.append(f"<div><b>{_esc(grid_a_name)}:</b> {a_acc:.1f}% ({a_correct}/{a_tot})</div>")

    if gridB:
        b_acc = (b_correct / b_tot * 100.0) if b_tot else 0.0
        parts.append(f"<div><b>{_esc(grid_b_name)}:</b> {b_acc:.1f}% ({b_correct}/{b_tot})</div>")

    # Per-line table (only 1,2,3)
    def fmt_cell(c, t):
        return "—" if t == 0 else f"{(c/t*100):.1f}% ({c}/{t})"

    per_line_rows = []
    for nl in (1, 2, 3):
        a_row = fmt_cell(a_cor_by_nl[nl], a_tot_by_nl[nl])
        if gridB:
            b_row = fmt_cell(b_cor_by_nl[nl], b_tot_by_nl[nl])
            per_line_rows.append(f"<tr><td>{nl}</td><td>{a_row}</td><td>{b_row}</td></tr>")
        else:
            per_line_rows.append(f"<tr><td>{nl}</td><td>{a_row}</td></tr>")

    if gridB:
        per_line_table = f"""
        <table class="small">
          <thead><tr><th>Num Lines</th><th>{_esc(grid_a_name)}</th><th>{_esc(grid_b_name)}</th></tr></thead>
          <tbody>
            {''.join(per_line_rows)}
          </tbody>
        </table>
        """
    else:
        per_line_table = f"""
        <table class="small">
          <thead><tr><th>Num Lines</th><th>{_esc(grid_a_name)}</th></tr></thead>
          <tbody>
            {''.join(per_line_rows)}
          </tbody>
        </table>
        """

    summary = f'<div class="summary">{"".join(parts)}<div class="subhead">Per-line accuracy</div>{per_line_table}</div>'

    # Table header
    headers = ["Run", "Prompts", "Ground Truth"]
    if not no_raw:
        headers.append("Raw Baseline")
    headers.extend(["Original Image", grid_a_name])
    if gridB:
        headers.append(grid_b_name)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Ball Drop: GT • {' • '.join([_esc(h) for h in headers[3:]])}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }}
  h1 {{ margin: 0 0 8px 0; font-size: 20px; }}
  .paths {{ font-size: 12px; color: #555; margin-bottom: 10px; }}
  .summary {{ margin: 10px 0 16px 0; font-size: 14px; }}
  .subhead {{ margin-top: 8px; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
  table.small {{ width: auto; table-layout: auto; margin-top: 8px; font-size: 13px; }}
  table.small th, table.small td {{ border: 1px solid #ddd; padding: 6px 8px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
  th {{ background: #f8f8f8; }}
  td.idx {{ width: 160px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: #555; }}
  td.prompt {{ width: 26%; }}
  td.gt {{ width: 7%; }}
  td.raw {{ width: 16%; }}
  td.orig {{ width: 21%; }}
  td.grid {{ width: 30%; }}
  img {{ max-width: 100%; height: auto; display: block; }}
  .missing {{ padding: 12px; text-align: center; background: #fafafa; color: #999; border: 1px dashed #ddd; }}
  .metric {{ margin-top: 6px; font-size: 13px; color: #333; }}
  .label {{ color: #666; }}
  .val {{ font-weight: 600; }}
  .ok {{ color: #1e8e3e; font-weight: 700; }}
  .bad {{ color: #c62828; font-weight: 700; }}
  .na {{ color: #888; }}
  details > summary {{ cursor: pointer; color: #0b57d0; }}
  pre {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.35; }}
  .legend {{ font-size: 12px; color: #555; margin: 8px 0 16px; }}
</style>
</head>
<body>
  <h1>Ball Drop: Ground Truth • {('Raw • ' if not no_raw else '')}{_esc(grid_a_name)}{(' • ' + _esc(grid_b_name)) if gridB else ''}</h1>
  <div class="paths">
    <div><b>GT root:</b> {gt_root.as_posix()}</div>
    {'' if no_raw else f'<div><b>RAW jsonl:</b> {raw_jsonl.as_posix() if raw_jsonl else "(none)"} </div>'}
    <div><b>GRID A:</b> {grid_dir_a.as_posix()} ({_esc(grid_a_name)})</div>
    {f'<div><b>GRID B:</b> {grid_dir_b.as_posix()} ({_esc(grid_b_name)})</div>' if grid_dir_b else ''}
  </div>
  {summary}
  <div class="legend">Columns: Run • Prompts • Ground Truth (0=none){'' if no_raw else ' • Raw (value + raw_text)'} • Original Image • {_esc(grid_a_name)}{(' • ' + _esc(grid_b_name)) if grid_dir_b else ''}</div>
  <table>
    <thead>
      <tr>
        {''.join(f'<th>{_esc(h)}</th>' for h in headers)}
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote: {out_html}")


def main():
    ap = argparse.ArgumentParser(description="Compare GT with one or two Grid Sketch runs (optionally Raw baseline).")
    ap.add_argument("--gt-root", type=Path, required=True, help="Root of ground truth with run_###[_v] folders.")
    ap.add_argument("--raw-jsonl", type=Path, required=False, help="JSONL with raw baseline outputs (optional if --no-raw or --grid-dir-b).")
    ap.add_argument("--grid-dir", type=Path, required=True, help="Grid A folder with item_*.json and *_annotated/_orig images.")
    ap.add_argument("--grid-dir-b", type=Path, required=False, help="Grid B folder to compare side-by-side (optional).")
    ap.add_argument("--grid-a-name", type=str, default="Grid A", help="Label for Grid A column.")
    ap.add_argument("--grid-b-name", type=str, default="Grid B", help="Label for Grid B column.")
    ap.add_argument("--no-raw", action="store_true", help="Hide raw baseline column and summary.")
    ap.add_argument("--out", type=Path, required=True, help="Output HTML path.")
    ap.add_argument("--thumb-width", type=int, default=480, help="Embed images resized to this width (px).")
    args = ap.parse_args()

    if not args.raw_jsonl and not args.no_raw and not args.grid_dir_b:
        # Preserve legacy expectation: if you didn't pass --no-raw and didn't supply Grid B,
        # we expect a raw jsonl (like the old script).
        ap.error("Either provide --raw-jsonl for legacy single-grid mode, or pass --no-raw and/or --grid-dir-b.")

    build_report(
        gt_root=args.gt_root,
        raw_jsonl=args.raw_jsonl,
        grid_dir_a=args.grid_dir,
        grid_dir_b=args.grid_dir_b,
        grid_a_name=args.grid_a_name,
        grid_b_name=args.grid_b_name,
        no_raw=args.no_raw,
        out_html=args.out,
        thumb_width=args.thumb_width,
    )

if __name__ == "__main__":
    main()


'''


python ball_drop_compare_report.py --gt-root large_run_split --grid-dir results/mix_eval/meta_ball_better --grid-dir-b results/mix_eval/meta_ball_line --grid-a-name "Curves + lines" --grid-b-name "Straight lines only" --no-raw --out meta_ball_better.html --thumb-width 480
python ball_drop_compare_report.py --gt-root large_run_split --grid-dir results/mix_eval/meta_flash --grid-dir-b results/mix_eval/meta_ball_line --grid-a-name "flash" --grid-b-name "Straight lines only" --no-raw --out meta_ball_better.html --thumb-width 480
python ball_drop_compare_report.py --gt-root large_run_split --grid-dir results/mix_eval/gpt_ball_drop --grid-dir-b results/mix_eval/meta_ball_line --grid-a-name "flash" --grid-b-name "Straight lines only" --no-raw --out meta_ball_better.html --thumb-width 480


python ball_drop_compare_report.py --gt-root large_run_split --grid-dir results/mix_eval/gpt_ball_drop_curves --grid-dir-b results/mix_eval/gpt_ball_drop_lines --grid-a-name "Curves + lines" --grid-b-name "Straight lines only" --no-raw --out meta_ball_better.html --thumb-width 480



  '''
  