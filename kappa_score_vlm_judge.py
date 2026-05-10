#!/usr/bin/env python3
"""
Human vs VLM-judge Kappa (UNWEIGHTED / LINEAR / QUADRATIC)

Overlap is defined EXACTLY by (dataset, model_name, JOIN_KEY) across humans.
Then for those overlapped keys, we match to VLM-judge using the SAME JOIN_KEY.

JOIN_KEY rules (robust across model pipelines):
- Default: item_##### from sample_id (or digits -> item_#####). Fallback to item_ in paths.
- ThinkMorph: run_### + image filename => "run_076/image_0.png"
- Vilasr: EXACT path-segment sample_YYYYMMDD_HHMMSS + image filename
         => "sample_20251126_053346/image_5.png"
  IMPORTANT: require /sample_20251126_053346/ as a full path segment
             so sample_20251126_053346_2 will NOT match.

Output:
- Two separate tables (one per human user), users NOT mixed:
    dataset | model | n_used | kappa_unw | kappa_lin | kappa_qua

n_used = number of matched pairs used for kappa = | (human-human overlap keys) ∩ (judge keys) |
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ======================
# DEFAULTS (EDIT HERE)
# ======================
IN_DIR = Path("datasets/kappa/annotations_snapshots")

NAME_TOKENS = [
    "brandonfinal",
    "logan",
    "hung_final"
]

USERS_TO_COMPARE = [
    "brandonfinal",
    "hung_final"
]

USER_MATCH_MODE = "substring"   # "exact" or "substring"
DEDUP = "latest"                # "latest" or "first"

# ======================
# VLM judge files mapping
# ======================
VLM_JUDGE: Dict[str, Dict[str, str]] = {
    "Ball Path": {
        "Gemini-3.0-Pro": "datasets/kappa/judge_output/ball_quality/batch1_gemini3pro_0_1000.json",
        "Gemini-2.5-Pro": "datasets/kappa/judge_output/ball_quality/batch1_gemini_25_pro.json",
        "ThinkMorph": "datasets/kappa/judge_output/ball_quality/batch1_thinkmorph.json",
        "Vilasr": "datasets/kappa/judge_output/ball_quality/batch1_vilasr.json",
        "Nano Banana": "datasets/kappa/judge_output/ball_quality/batch1_nano_banana.json",
        "GPT-5 (low)": "datasets/kappa/judge_output/ball_quality/batch1_gpt5_low.json",
    },
    "VPCT": {
        "Gemini-3.0-Pro": "datasets/kappa/judge_output/vpct_quality/vpct_gemini3pro_0_1000.json",
        "Gemini-2.5-Pro": "datasets/kappa/judge_output/vpct_quality/vpct_gemini_pro25.json",
        "ThinkMorph": "datasets/kappa/judge_output/vpct_quality/vpct_thinkmorph.json",
        "Vilasr": "datasets/kappa/judge_output/vpct_quality/vpct_vilasr.json",
        "Nano Banana": "datasets/kappa/judge_output/vpct_quality/vpct_nanobanana.json",
        "GPT-5 (low)": "datasets/kappa/judge_output/vpct_quality/vpct_gpt5low.json",
    },
    "Maze Valid": {
        "Gemini-3.0-Pro": "datasets/kappa/judge_output/grid_world_quality/gemini3pro_validpaths_0_1000.json",
        "Gemini-2.5-Pro": "datasets/kappa/judge_output/grid_world_quality/gemini25_pro_valid.json",
        "ThinkMorph": "datasets/kappa/judge_output/grid_world_quality/thinkmorph_valid.json",
        "Vilasr": "datasets/kappa/judge_output/grid_world_quality/vilasr_valid.json",
        "Nano Banana": "datasets/kappa/judge_output/grid_world_quality/quality_results_nano_banana_valid.json",
        "GPT-5 (low)": "datasets/kappa/judge_output/grid_world_quality/gpt5_low_valid.json",
    },
    "Maze Invalid": {
        "Gemini-3.0-Pro": "datasets/kappa/judge_output/grid_world_quality/gemini3_pro_invalid.json",
        "Gemini-2.5-Pro": "datasets/kappa/judge_output/grid_world_quality/gemini25_pro_invalid.json",
        "ThinkMorph": "datasets/kappa/judge_output/grid_world_quality/thinkmorph_invalid.json",
        "Vilasr": "datasets/kappa/judge_output/grid_world_quality/vilasr_invalid.json",
        "Nano Banana": "datasets/kappa/judge_output/grid_world_quality/quality_results_nano_banana_invalid.json",
        "GPT-5 (low)": "datasets/kappa/judge_output/grid_world_quality/gpt5_low_invalid.json",
    },
}

# ======================
# Regex
# ======================
ITEM_RE = re.compile(r"(item_\d+)", re.IGNORECASE)
RUN_RE = re.compile(r"(run_\d+)", re.IGNORECASE)

# EXACT path-segment match for sample token:
# matches .../sample_20251126_053346/... but NOT .../sample_20251126_053346_2/...
SAMPLE_SEG_RE = re.compile(r"(?:^|[\\/])(sample_\d{8}_\d{6})(?:[\\/]|$)", re.IGNORECASE)

# image filename inside an image folder
IMG_RE = re.compile(r"(image_\d+\.(?:png|jpg|jpeg|webp))", re.IGNORECASE)

QS_RE = re.compile(r"Quality\s*Score\s*:\s*([1-5])", re.IGNORECASE)

# ======================
# Helpers
# ======================

def norm_text(x: str) -> str:
    s = (x or "").strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    return s

def parse_iso(t: Optional[str]) -> Optional[datetime]:
    if not t:
        return None
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00"))
    except Exception:
        return None

def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue

def safe_read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def file_matches(path: Path, tokens: List[str]) -> bool:
    if not tokens:
        return True
    name = path.name.lower()
    return any(tok.lower() in name for tok in tokens)

def canonical_user(rec_user: str) -> Optional[str]:
    ru = (rec_user or "").strip()
    if USER_MATCH_MODE == "exact":
        return ru if ru in USERS_TO_COMPARE else None
    rul = ru.lower()
    for u in USERS_TO_COMPARE:
        if u.lower() in rul:
            return u
    return None

def normalize_score(x) -> Optional[int]:
    try:
        v = int(x)
        if 1 <= v <= 5:
            return v
        return None
    except Exception:
        return None

def normalize_sample_to_item(sample_id: str) -> Optional[str]:
    """
    Default join key from sample_id:
    - if contains item_\d+ => item_#####
    - else if digits => item_#####
    - else None
    """
    if sample_id is None:
        return None
    s = str(sample_id).strip()
    if not s:
        return None

    m = ITEM_RE.search(s)
    if m:
        return m.group(1).lower()

    if s.isdigit() or re.fullmatch(r"\d{1,10}", s):
        return f"item_{int(s):05d}"

    return None

def get_dataset_model(rec: dict) -> Tuple[Optional[str], Optional[str]]:
    ds = rec.get("dataset")
    if ds is None:
        ds = rec.get("dataset_name")

    model = rec.get("model_name")
    if model is None:
        model = rec.get("model")

    if ds is None or model is None:
        return None, None

    return norm_text(str(ds)), norm_text(str(model))

def parse_vlm_judge_score(rec: dict) -> Optional[int]:
    # prefer explicit numeric fields
    for k in ["quality_score", "judge_score", "score"]:
        if k in rec:
            v = normalize_score(rec.get(k))
            if v is not None:
                return v

    # parse from text
    resp = rec.get("consistency_check_response") or rec.get("response") or ""
    if isinstance(resp, str):
        m = QS_RE.search(resp)
        if m:
            return int(m.group(1))
    return None

def find_first_match(regex: re.Pattern, *vals: object) -> Optional[str]:
    """
    Returns the *captured group(1)*, lowercased.
    Safe to use with SAMPLE_SEG_RE and others that define group(1).
    """
    for v in vals:
        if not v:
            continue
        if isinstance(v, (dict, list)):
            continue
        s = str(v)
        m = regex.search(s)
        if m:
            return m.group(1).lower()
    return None

def join_key_from_human(rec: dict, model_n: str) -> Optional[str]:
    """
    ThinkMorph: run_###/image_X.png
    Vilasr: sample_YYYYMMDD_HHMMSS/image_X.png (sample must be exact path segment)
    Default: item_#####
    """
    meta = rec.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}

    sample_id = rec.get("sample_id", "")
    orig_path = rec.get("orig_path", "")
    ann_path = rec.get("ann_path", "")
    sample_dir = meta.get("sample_dir", "")
    run_full = meta.get("run_full", "")

    is_thinkmorph = ("thinkmorph" in model_n)
    is_vilasr = ("vilasr" in model_n)

    img = find_first_match(IMG_RE, ann_path, orig_path, sample_dir, sample_id)

    if is_thinkmorph:
        run = find_first_match(RUN_RE, sample_id, orig_path, ann_path, sample_dir, run_full)
        if run and img:
            return f"{run}/{img}"
        if run:
            return run

    if is_vilasr:
        sample = find_first_match(SAMPLE_SEG_RE, ann_path, orig_path, sample_dir)
        if sample and img:
            return f"{sample}/{img}"
        if sample:
            return sample  # fallback; not ideal but keeps running

    # Default: item_ from sample_id/digits
    item = normalize_sample_to_item(sample_id)
    if item:
        return item

    # Fallback: try extracting item/run/sample from paths
    item = find_first_match(ITEM_RE, ann_path, orig_path, sample_dir)
    if item:
        return item

    run = find_first_match(RUN_RE, ann_path, orig_path, sample_dir)
    if run and img:
        return f"{run}/{img}"
    if run:
        return run

    sample = find_first_match(SAMPLE_SEG_RE, ann_path, orig_path, sample_dir)
    if sample and img:
        return f"{sample}/{img}"
    if sample:
        return sample

    return None

def join_key_from_judge(rec: dict, model_n: str) -> Optional[str]:
    """
    ThinkMorph: run_###/image_X.png
    Vilasr: sample_YYYYMMDD_HHMMSS/image_X.png (sample must be exact path segment)
    Default: item_#####
    """
    cand = []
    for k in ["image_path", "img_path", "path", "file_path", "overlay_path",
              "item_id", "sample_id", "id", "orig_path", "ann_path"]:
        if k in rec:
            cand.append(rec.get(k))

    # last resort scan all string values
    str_vals = [v for v in rec.values() if isinstance(v, str)]

    img = find_first_match(IMG_RE, *cand, *str_vals)

    is_thinkmorph = ("thinkmorph" in model_n)
    is_vilasr = ("vilasr" in model_n)

    if is_thinkmorph:
        run = find_first_match(RUN_RE, *cand, *str_vals)
        if run and img:
            return f"{run}/{img}"
        if run:
            return run

    if is_vilasr:
        sample = find_first_match(SAMPLE_SEG_RE, *cand, *str_vals)
        if sample and img:
            return f"{sample}/{img}"
        if sample:
            return sample

    item = find_first_match(ITEM_RE, *cand, *str_vals)
    if item:
        return item

    # fallback (rare)
    run = find_first_match(RUN_RE, *cand, *str_vals)
    if run and img:
        return f"{run}/{img}"
    if run:
        return run

    sample = find_first_match(SAMPLE_SEG_RE, *cand, *str_vals)
    if sample and img:
        return f"{sample}/{img}"
    if sample:
        return sample

    return None

# ======================
# Kappa
# ======================

def kappa_score(y1: List[int], y2: List[int], mode: str = "unweighted") -> float:
    assert len(y1) == len(y2)
    if len(y1) == 0:
        return float("nan")

    cats = [1, 2, 3, 4, 5]
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    m = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        if a not in idx or b not in idx:
            continue
        m[idx[a]][idx[b]] += 1

    total = sum(sum(r) for r in m)
    if total == 0:
        return float("nan")

    O = [[m[i][j] / total for j in range(k)] for i in range(k)]
    row = [sum(O[i][j] for j in range(k)) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    E = [[row[i] * col[j] for j in range(k)] for i in range(k)]

    denom = (k - 1) if (k - 1) != 0 else 1

    def w(i: int, j: int) -> float:
        if mode == "unweighted":
            return 0.0 if i == j else 1.0
        d = abs(i - j) / denom
        if mode == "linear":
            return d
        if mode in ("quadratic", "quadric"):
            return d * d
        raise ValueError(f"Unknown mode: {mode}")

    num = 0.0
    den = 0.0
    for i in range(k):
        for j in range(k):
            wij = w(i, j)
            num += wij * O[i][j]
            den += wij * E[i][j]

    if math.isclose(den, 0.0):
        return 1.0 if math.isclose(num, 0.0) else 0.0

    return 1.0 - (num / den)

def fmt(x: float) -> str:
    return "nan" if (x != x) else f"{x:.4f}"

# ======================
# Judge loader: JOIN_KEY -> score
# ======================

def load_judge_records(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".jsonl":
        return list(iter_jsonl(path))

    data = safe_read_json(path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ["data", "items", "records", "results"]:
            v = data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    raise ValueError(f"Unrecognized judge format in {path}")

def load_vlm_judge_by_key(judge_path: Path, model: str) -> Dict[str, int]:
    model_n = norm_text(model)
    recs = load_judge_records(judge_path)

    out: Dict[str, int] = {}
    collisions = 0

    for rec in recs:
        key = join_key_from_judge(rec, model_n)
        if not key:
            continue
        score = parse_vlm_judge_score(rec)
        if score is None:
            continue
        if key in out:
            collisions += 1
        out[key] = score

    if collisions:
        print(f"  [WARN] Judge key collisions for {model}: {collisions} overwrites (key not unique)")

    return out

# ======================
# Humans loader for a slice (dataset, model)
# ======================

Key = Tuple[str, str, str]          # (dataset_norm, model_norm, join_key)
HumanVal = Tuple[int, Optional[datetime], Path]

def load_humans_for_slice(dataset: str, model: str) -> Dict[str, Dict[Key, HumanVal]]:
    ds_n = norm_text(dataset)
    model_n = norm_text(model)

    humans: Dict[str, Dict[Key, HumanVal]] = {u: {} for u in USERS_TO_COMPARE}

    files = sorted(IN_DIR.glob("*.jsonl"))
    files = [p for p in files if file_matches(p, NAME_TOKENS)]

    for p in files:
        for rec in iter_jsonl(p):
            u_raw = str(rec.get("user", "")).strip()
            u = canonical_user(u_raw)
            if u is None:
                continue

            ds_rec, model_rec = get_dataset_model(rec)
            if ds_rec is None or model_rec is None:
                continue
            if ds_rec != ds_n or model_rec != model_n:
                continue

            join_key = join_key_from_human(rec, model_n)
            if not join_key:
                continue

            score = normalize_score(rec.get("quality_score"))
            if score is None:
                continue

            t = parse_iso(rec.get("time"))
            key: Key = (ds_n, model_n, join_key)

            prev = humans[u].get(key)
            if prev is None:
                humans[u][key] = (score, t, p)
                continue

            if DEDUP == "first":
                continue

            # latest
            _, prev_t, _ = prev
            if prev_t is None and t is not None:
                humans[u][key] = (score, t, p)
            elif prev_t is not None and t is None:
                continue
            elif prev_t is None and t is None:
                continue
            else:
                if t >= prev_t:
                    humans[u][key] = (score, t, p)

    return humans

# ======================
# Overlap + slice kappa computation
# ======================

def compute_overlap_keys_all(humans: Dict[str, Dict[Key, HumanVal]]) -> List[Key]:
    if len(USERS_TO_COMPARE) < 2:
        raise SystemExit("Need at least 2 users in USERS_TO_COMPARE to define overlap.")
    sets = [set(humans.get(u, {}).keys()) for u in USERS_TO_COMPARE]
    overlap = set.intersection(*sets) if sets else set()
    return sorted(overlap)

def compute_kappas_for_slice_per_user(
    dataset: str,
    model: str,
    humans: Dict[str, Dict[Key, HumanVal]],
    judge: Dict[str, int],
) -> Dict[str, List[str]]:
    ds_n = norm_text(dataset)
    md_n = norm_text(model)

    overlap_keys = compute_overlap_keys_all(humans)
    out: Dict[str, List[str]] = {}

    for u in USERS_TO_COMPARE:
        hu = humans.get(u, {})
        used_keys = [k for k in overlap_keys if (k in hu and k[2] in judge)]
        if not used_keys:
            out[u] = [ds_n, md_n, "0", "nan", "nan", "nan"]
            continue

        y_h = [hu[k][0] for k in used_keys]
        y_j = [judge[k[2]] for k in used_keys]

        k_unw = kappa_score(y_h, y_j, "unweighted")
        k_lin = kappa_score(y_h, y_j, "linear")
        k_qua = kappa_score(y_h, y_j, "quadratic")

        out[u] = [ds_n, md_n, str(len(used_keys)), fmt(k_unw), fmt(k_lin), fmt(k_qua)]

    return out

# ======================
# Table printing
# ======================

def table_row(cols: List[str], widths: List[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cols, widths))

def print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    print(table_row(headers, widths))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(table_row(r, widths))

# ======================
# Main
# ======================

def main():
    if not IN_DIR.exists():
        raise SystemExit(f"IN_DIR not found: {IN_DIR}")

    per_user_rows: Dict[str, List[List[str]]] = {u: [] for u in USERS_TO_COMPARE}
    headers = ["dataset", "model", "n_used", "kappa_unw", "kappa_lin", "kappa_qua"]

    for dataset, model_map in VLM_JUDGE.items():
        for model, judge_path_str in model_map.items():
            judge_path = Path(judge_path_str)

            humans = load_humans_for_slice(dataset, model)
            any_human = sum(len(humans[u]) for u in USERS_TO_COMPARE)
            if any_human == 0:
                ds_n = norm_text(dataset)
                md_n = norm_text(model)
                for u in USERS_TO_COMPARE:
                    per_user_rows[u].append([ds_n, md_n, "0", "nan", "nan", "nan"])
                continue

            try:
                judge = load_vlm_judge_by_key(judge_path, model)
            except Exception:
                ds_n = norm_text(dataset)
                md_n = norm_text(model)
                for u in USERS_TO_COMPARE:
                    per_user_rows[u].append([ds_n, md_n, "0", "nan", "nan", "nan"])
                continue

            out = compute_kappas_for_slice_per_user(dataset, model, humans, judge)
            for u in USERS_TO_COMPARE:
                per_user_rows[u].append(out.get(u, [norm_text(dataset), norm_text(model), "0", "nan", "nan", "nan"]))

    for u in USERS_TO_COMPARE:
        rows = per_user_rows[u]
        rows.sort(key=lambda r: (r[0], r[1]))
        print(f"\n=== {u} vs vlm_judge ===")
        print_table(headers, rows)

if __name__ == "__main__":
    main()
