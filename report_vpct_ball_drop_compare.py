#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPCT Ball Drop: Compare Ground Truth vs Raw Baseline vs Grid Sketch
Now supports raw results from a JSONL file via --raw-jsonl.
"""
import re, io, os, json, base64, argparse
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from PIL import Image

PROMPT_DISPLAY: str = (
    "Draw the path that the ball will take. It will either land in container 1, 2, 3 (numbered left to right) "
    "The ball is released from rest, the only force it is subject to is gravity. "
    "The black lines are walls and platforms and the ball cannot pass through them."
)
SIM_IMG_RE = re.compile(r"sim_(\d+)_initial\.png$", re.I)


def _init_cm() -> List[List[int]]:
    """Create an empty 3×3 confusion matrix for buckets 1,2,3."""
    return [[0, 0, 0] for _ in range(3)]


def _render_cm_table(cm: List[List[int]], title: str) -> str:
    """
    Render a 3×3 confusion matrix as HTML.
    Convention: rows = GT bucket, columns = predicted bucket.
    """
    total = sum(sum(r) for r in cm)
    if total == 0:
        return f'''
<div class="cm-block">
  <div class="cm-title"><b>{_esc(title)} confusion matrix</b> (no examples with both GT and prediction)</div>
</div>
'''
    header = "<tr><th class=\"label\">GT \\ Pred</th>" + "".join(
        f"<th>{i}</th>" for i in (1, 2, 3)
    ) + "</tr>"

    body_rows = []
    for gt_bucket in (1, 2, 3):
        row = cm[gt_bucket - 1]
        cells = "".join(f"<td>{v}</td>" for v in row)
        body_rows.append(f"<tr><th>{gt_bucket}</th>{cells}</tr>")

    body_html = "\n".join(body_rows)
    return f'''
<div class="cm-block">
  <div class="cm-title"><b>{_esc(title)} confusion matrix</b> (rows = ground-truth bucket, columns = predicted bucket)</div>
  <table class="cm">
    <thead>{header}</thead>
    <tbody>
      {body_html}
    </tbody>
  </table>
</div>
'''


def _esc(s: Optional[str]) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _to_int_123(v: Any) -> Optional[int]:
    if v is None: return None
    try:
        iv = int(str(v).strip())
        return iv if iv in (1,2,3) else None
    except Exception:
        return None

def _img_to_data_uri(path: Path, max_width: int = 520) -> Optional[str]:
    try:
        with Image.open(path) as im:
            is_png = path.suffix.lower() == ".png"
            im = im.convert("RGBA" if is_png else "RGB")
            if max_width and im.width > max_width:
                nh = max(1, int(im.height * (max_width/float(im.width))))
                im = im.resize((max_width, nh), Image.LANCZOS)
            buf = io.BytesIO()
            if is_png:
                im.save(buf, format="PNG"); mime="image/png"
            else:
                im = im.convert("RGB"); im.save(buf, format="JPEG", quality=92, subsampling=0); mime="image/jpeg"
            import base64 as b64
            return f"data:{mime};base64,{b64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception:
        return None

def load_gt(gt_root: Path) -> Dict[str,int]:
    """
    Read ground-truth bucket from files like sim_###_results.json and map to sim_###_initial.png basename.
    """
    out: Dict[str,int] = {}
    for jf in sorted(gt_root.glob("sim_*_results.json")):
        try:
            j=json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        gt = _to_int_123(j.get("finalBucket"))
        if gt is None: 
            continue
        m = re.search(r"sim_(\d+)_results\.json$", jf.name, re.I)
        if not m: 
            continue
        img = f"sim_{m.group(1)}_initial.png"
        out[img] = gt
    return out

def load_raw_dir(raw_dir: Path, raw_fallback_last_number: bool = False) -> Dict[str,Dict[str,Any]]:
    """
    Raw loader: scan a directory of item_*.json files.

    Supports both:
      - legacy raw-eval json (model_raw_text + image)
      - VQA-style json from mix_eval (answer + model_output_full + source_image)
    """
    out: Dict[str,Dict[str,Any]] = {}
    for jf in sorted(raw_dir.glob("item_*.json")):
        try:
            j=json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Try to locate the VPCT source image (sim_###_initial.png)
        img_path = str(j.get("source_image","") or j.get("image","") or j.get("file","")).replace("\\","/")
        basename = Path(img_path).name
        if not SIM_IMG_RE.search(basename):
            # Some legacy formats may store it elsewhere
            alt = str(j.get("image","") or j.get("source_image","") or "").replace("\\","/")
            basename = Path(alt).name
            if not SIM_IMG_RE.search(basename):
                continue

        # Prefer the unified "answer" field (same as grid sketch parsing starts with)
        pred = _to_int_123(j.get("answer"))

        # Fallback: legacy raw format stored the model output under model_raw_text.text
        raw_text = ""
        mr = j.get("model_raw_text", {})
        if isinstance(mr, dict):
            raw_text = str(mr.get("text","") or "")
        if pred is None and raw_text.strip():
            pred = _to_int_123(raw_text)

        # Fallback: some VQA outputs store the full text in model_output_full
        mo_full = j.get("model_output_full") or j.get("model_out_full") or ""
        if pred is None and raw_fallback_last_number and isinstance(mo_full, str) and mo_full.strip():
            last_int = _extract_last_int_token(mo_full)
            pred = _to_int_123(last_int)

        # What to show in the HTML "raw_text" details panel
        if not raw_text.strip() and isinstance(mo_full, str) and mo_full.strip():
            raw_text = mo_full

        idxm = re.match(r"item_(\d+)\.json$", jf.name, re.I)
        orig_path=None
        if idxm:
            idx=idxm.group(1)
            for ext in (".jpg",".png",".jpeg",".webp"):
                cand = raw_dir / f"item_{idx}_orig{ext}"
                if cand.exists():
                    orig_path=cand; break

        out[basename]={"pred":pred,"orig":orig_path,"raw_text":raw_text}
    return out


def load_raw_jsonl(jsonl_path: Path, raw_fallback_last_number: bool = False) -> Dict[str,Dict[str,Any]]:
    """
    JSONL raw loader.

    We map by the basename of a file field (file/image/source_image) and try:
      1) "answer" (if present)
      2) parsed_int / parsed_label / raw_text (legacy jsonl schema)
      3) (optional) last integer in model_output_full if answer is null
    """
    out: Dict[str,Dict[str,Any]] = {}
    for ln in jsonl_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            j = json.loads(ln)
        except Exception:
            continue

        f = (j.get("file") or j.get("image") or j.get("source_image") or "").replace("\\","/")
        basename = Path(f).name
        if not SIM_IMG_RE.search(basename):
            continue

        pred = _to_int_123(j.get("answer"))

        if pred is None:
            pred0 = j.get("parsed_int")
            if pred0 is None:
                pred = _to_int_123(j.get("parsed_label")) or _to_int_123(j.get("raw_text"))
            else:
                pred = _to_int_123(pred0)

        raw_text = str(j.get("raw_text") or "")
        mo_full = j.get("model_output_full") or j.get("model_out_full") or ""
        if pred is None and raw_fallback_last_number and isinstance(mo_full, str) and mo_full.strip():
            last_int = _extract_last_int_token(mo_full)
            pred = _to_int_123(last_int)

        if not raw_text.strip() and isinstance(mo_full, str) and mo_full.strip():
            raw_text = mo_full

        out[basename] = {"pred": pred, "raw_text": raw_text, "orig": None}
    return out



def _extract_last_int_token(text: str) -> Optional[int]:
    """
    Extract the last integer token from text.
    Accepts punctuation like '1.' or '(10)', but rejects decimals like '21.5'.
    """
    if not text:
        return None

    vals = []
    for m in re.finditer(r"-?\d+", text):
        s, e = m.span()

        prev = text[s - 1] if s > 0 else ""
        prevprev = text[s - 2] if s > 1 else ""
        nxt = text[e] if e < len(text) else ""
        nxtnxt = text[e + 1] if e + 1 < len(text) else ""

        # Reject decimal cases:
        #  - this token is the left side of a decimal: "21.5" -> "21"
        if nxt == "." and nxtnxt.isdigit():
            continue
        #  - this token is the right side of a decimal: "21.5" -> "5"
        if prev == "." and prevprev.isdigit():
            continue

        try:
            vals.append(int(m.group()))
        except Exception:
            pass

    return vals[-1] if vals else None



def _extract_last_bucket_from_output(text: str) -> Optional[int]:
    """
    VPCT helper: from a long model_output_full string, extract the last
    standalone integer token, then keep it only if it's in {1,2,3}.
    """
    iv = _extract_last_int_token(text)
    return iv if iv in (1, 2, 3) else None


def load_grid_dir(grid_dir: Path) -> Dict[str,Dict[str,Any]]:
    out: Dict[str,Dict[str,Any]] = {}
    for jf in sorted(grid_dir.glob("item_*.json")):
        try:
            j=json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        img_path = str(j.get("source_image","")).replace("\\","/")
        basename = Path(img_path).name
        if not SIM_IMG_RE.search(basename):
            continue

        # --- NEW: smarter parsing of grid prediction ---
        pred = _to_int_123(j.get("answer"))
        if pred is None:
            mo_full = j.get("model_output_full")
            if isinstance(mo_full, str) and mo_full.strip():
                # Case 1: model_output_full is just "2" or "3" etc.
                pred = _to_int_123(mo_full)
                # Case 2: long explanation – use last standalone 1/2/3
                if pred is None:
                    pred = _extract_last_bucket_from_output(mo_full)
        # --- END NEW ---

        idxm = re.match(r"item_(\d+)\.json$", jf.name, re.I)
        annot_path = orig_path = None
        if idxm:
            idx = idxm.group(1)
            for ext in (".png",".jpg",".jpeg",".webp"):
                cand = grid_dir / f"item_{idx}_annotated{ext}"
                if cand.exists():
                    annot_path=cand; break
            for ext in (".jpg",".png",".jpeg",".webp"):
                cand = grid_dir / f"item_{idx}_orig{ext}"
                if cand.exists():
                    orig_path=cand; break
        out[basename]={"pred":pred,"annot":annot_path,"orig":orig_path}
    return out


def build_report(gt_root: Path, raw_source: Dict[str,Dict[str,Any]], grid_dir: Path, out_html: Path, thumb_width: int = 520) -> None:
    gt_map = load_gt(gt_root)
    raw_map = raw_source
    grid_map = load_grid_dir(grid_dir)

    def sort_key(name: str):
        m = SIM_IMG_RE.search(name)
        return (int(m.group(1)) if m else 10**9, name)

    all_imgs = sorted(gt_map.keys(), key=sort_key)

    raw_tot=raw_correct=grid_tot=grid_correct=0
    rows: List[str]=[]
    
    # Confusion matrices: rows = GT bucket, cols = predicted bucket
    raw_cm = _init_cm()
    grid_cm = _init_cm()


    for img_name in all_imgs:
        gt = gt_map.get(img_name)
        rinfo = raw_map.get(img_name, {})
        ginfo = grid_map.get(img_name, {})

        rpred = rinfo.get("pred"); rtext=rinfo.get("raw_text",""); rorig=rinfo.get("orig")
        gpred = ginfo.get("pred"); gannt=ginfo.get("annot"); gorig=ginfo.get("orig")

        if rpred is not None:
            raw_tot+=1
            if rpred==gt: raw_correct+=1
        if gpred is not None:
            grid_tot+=1
            if gpred==gt: grid_correct+=1
            
            
        # Update confusion matrices when both GT and prediction are valid buckets
        if gt in (1, 2, 3) and rpred in (1, 2, 3):
            raw_cm[gt - 1][rpred - 1] += 1
        if gt in (1, 2, 3) and gpred in (1, 2, 3):
            grid_cm[gt - 1][gpred - 1] += 1


        gt_cell = f'<div class="metric"><span class="label">GT:</span> <span class="val">{gt}</span></div>'

        if rpred is None:
            raw_cell = '<div class="metric"><span class="label">Raw:</span> <span class="val">–</span> <span class="na">–</span></div>'
        else:
            badge = '<span class="ok">✓</span>' if rpred==gt else '<span class="bad">✗</span>'
            raw_cell = f'<div class="metric"><span class="label">Raw:</span> <span class="val">{rpred}</span> {badge}</div>'
        if rtext:
            raw_cell += f'<details class="rawtxt"><summary>raw_text</summary><pre>{_esc(rtext)}</pre></details>'

        orig_path = rorig or gorig
        if isinstance(orig_path, Path) and orig_path.exists():
            orig_uri = _img_to_data_uri(orig_path, max_width=thumb_width)
            orig_cell = f'<img src="{orig_uri}" />' if orig_uri else '<div class="missing">failed to embed</div>'
        else:
            fallback = gt_root / img_name
            if fallback.exists():
                furi = _img_to_data_uri(fallback, max_width=thumb_width)
                orig_cell = f'<img src="{furi}" />' if furi else '<div class="missing">failed to embed</div>'
            else:
                orig_cell = '<div class="missing">missing original</div>'

        if isinstance(gannt, Path) and gannt.exists():
            auri = _img_to_data_uri(gannt, max_width=thumb_width)
            img_html = f'<img src="{auri}" />' if auri else '<div class="missing">failed to embed</div>'
        else:
            img_html = '<div class="missing">missing annotated</div>'

        if gpred is None:
            grid_cell = img_html + '<div class="metric"><span class="label">Grid:</span> <span class="val">–</span> <span class="na">–</span></div>'
        else:
            gbadge = '<span class="ok">✓</span>' if gpred==gt else '<span class="bad">✗</span>'
            grid_cell = img_html + f'<div class="metric"><span class="label">Grid:</span> <span class="val">{gpred}</span> {gbadge}</div>'

        prompts_cell = f'<details><summary>show prompt</summary><pre>{_esc(PROMPT_DISPLAY)}</pre></details>'

        rows.append(f"""
<tr>
  <td class="idx">{_esc(img_name)}</td>
  <td class="prompt">{prompts_cell}</td>
  <td class="gt">{gt_cell}</td>
  <td class="raw">{raw_cell}</td>
  <td class="orig">{orig_cell}</td>
  <td class="grid">{grid_cell}</td>
</tr>
""")

    raw_acc = (raw_correct/raw_tot*100.0) if raw_tot else 0.0
    grid_acc = (grid_correct/grid_tot*100.0) if grid_tot else 0.0
    cm_html = (
        _render_cm_table(raw_cm, "Raw baseline") +
        _render_cm_table(grid_cm, "Grid sketch")
    )

    summary = f"""
<div class="summary">
  <div><b>Raw baseline:</b> {raw_acc:.1f}% ({raw_correct}/{raw_tot})</div>
  <div><b>Grid sketch:</b> {grid_acc:.1f}% ({grid_correct}/{grid_tot})</div>
</div>
{cm_html}
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>VPCT Ball Drop: GT vs Raw vs Grid</title>
<style>
  .cm-block {{ margin: 8px 0 14px 0; }}
  .cm-title {{ font-size: 13px; margin-bottom: 4px; color: #333; }}
  table.cm {{ border-collapse: collapse; margin-top: 4px; font-size: 12px; }}
  table.cm th, table.cm td {{ border: 1px solid #ccc; padding: 4px 6px; text-align: center; }}
  table.cm th.label {{ text-align: left; font-weight: 600; }}

  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }}
  h1 {{ margin: 0 0 8px 0; font-size: 20px; }}
  .paths {{ font-size: 12px; color: #555; margin-bottom: 10px; }}
  .summary {{ margin: 10px 0 16px 0; font-size: 14px; }}
  table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
  th {{ background: #f8f8f8; }}
  td.idx {{ width: 160px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: #555; }}
  td.prompt {{ width: 24%; }}
  td.gt {{ width: 7%; }}
  td.raw {{ width: 14%; }}
  td.orig {{ width: 24%; }}
  td.grid {{ width: 31%; }}
  
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
  <h1>VPCT Ball Drop: Ground Truth • Raw Baseline • Grid Sketch</h1>
  <div class="paths">
    <div><b>GT root:</b> {{gt_root}}</div>
    <div><b>RAW:</b> {{raw_src}}</div>
    <div><b>GRID dir:</b> {{grid_dir}}</div>
  </div>
  {summary}
  <div class="legend">Columns: Image • Prompt • Ground Truth • Raw (value + raw_text) • Original Image • Grid (annotated + value)</div>
  <table>
    <thead>
      <tr>
        <th>Image</th>
        <th>Prompt</th>
        <th>Ground Truth</th>
        <th>Raw Baseline</th>
        <th>Original Image</th>
        <th>Grid Sketch</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    out_html.write_text(
        html.replace("{{gt_root}}", gt_root.as_posix())
            .replace("{{raw_src}}", "(jsonl)" if not raw_source or isinstance(raw_source, dict) else str(raw_source))
            .replace("{{grid_dir}}", grid_dir.as_posix()),
        encoding="utf-8"
    )
    print(f"Wrote: {out_html}")

def main():
    ap = argparse.ArgumentParser(description="VPCT Ball Drop: compare GT vs Raw vs Grid and write an HTML report.")
    ap.add_argument("--gt-root", type=Path, required=True)
    # Choose exactly one of the following:
    ap.add_argument("--raw-dir", type=Path, default=None, help="Directory of item_*.json raw baseline (legacy).")
    ap.add_argument("--raw-jsonl", type=Path, default=None, help="JSONL file of raw baseline results.")
    ap.add_argument("--grid-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thumb-width", type=int, default=520)
    ap.add_argument("--raw-fallback-last-number", action="store_true",
                   help="If raw 'answer' is null/invalid, extract the last integer token from model_output_full (greedy).")
    args = ap.parse_args()

    if args.raw_jsonl and args.raw_dir:
        raise SystemExit("Please provide either --raw-jsonl or --raw-dir, not both.")
    if not args.raw_jsonl and not args.raw_dir:
        raise SystemExit("Please provide --raw-jsonl (recommended) or --raw-dir.")

    if args.raw_jsonl:
        raw_map = load_raw_jsonl(args.raw_jsonl, raw_fallback_last_number=args.raw_fallback_last_number)
        raw_src_label = args.raw_jsonl.as_posix()
    else:
        raw_map = load_raw_dir(args.raw_dir, raw_fallback_last_number=args.raw_fallback_last_number)
        raw_src_label = args.raw_dir.as_posix()

    build_report(args.gt_root, raw_map, args.grid_dir, args.out, thumb_width=args.thumb_width)

if __name__ == "__main__":
    main()



'''

python report_vpct_ball_drop_compare.py --gt-root vpct-1 --raw-dir results/raw_eval/vpct_ball_drop/vpct_ball_drop --grid-dir results/mix_eval/vpct_ball_drop --out report_vpct_gptTEST.html --thumb-width 520


Fallback for answer not saved for VQA - will use last integer in model_output_full:
python report_vpct_ball_drop_compare.py --gt-root vpct-1 --raw-dir results/raw_eval/vpct_qwen25vl_7b_vqa --grid-dir results/mix_eval/gemini3pro_multiturn_vpct_sample --out report_vpct_gptTEST_QWEN.html --raw-fallback-last-number --thumb-width 520

    
'''