from __future__ import annotations
import xml.etree.ElementTree as ET
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
from typing import Optional



# =========================
# ===== Grid related ======
# =========================

# ---------------------------------------------------------------------------
# NEW create_grid_image – thicker lines, bigger labels, optional “axis only”
# ---------------------------------------------------------------------------
# utils.py
from typing import Optional, Tuple, Dict
from PIL import Image, ImageDraw, ImageFont


def create_grid_image(
        res: int = 50,
        cell_size: int = 12,
        header_size: int = 12,
        *,
        line_w: int = 2,
        font_sz: Optional[int] = None,
        font_path: Optional[str] = None,
        full: bool = True
) -> Tuple[Image.Image, Dict[str, Tuple[int, int]]]:
    """
    Build a numbered grid.

    Parameters
    ----------
    res          : number of usable cells per axis (not counting headers)
    cell_size    : size of one grid square (px)
    header_size  : height/width reserved for the axis labels (px)
    line_w       : thickness of grid lines
    font_sz      : label size (defaults to 0.6 × cell_size)
    font_path    : custom TrueType path (falls back to default font)
    full         : if *True* draw every lattice line; if *False* draw only
                   axis lines plus tick “stubs” (the old compact style).

    Returns
    -------
    (image, positions) where *positions* maps "xAyB" → centre-pixel (x,y)
    """

    rows = cols = res
    W = (cols + 1) * cell_size          # +1 because of the left header col
    H = (rows + 1) * cell_size          # +1 because of the bottom header row

    im  = Image.new("RGB", (W, H), "white")
    drw = ImageDraw.Draw(im)

    # ─── font ──────────────────────────────────────────────────────────────
    if font_sz is None:
        font_sz = int(cell_size * 0.6)
    try:
        fnt = ImageFont.truetype(font_path or "arial.ttf", font_sz)
    except OSError:
        fnt = ImageFont.load_default()

    # ─── axis labels (outside the drawable grid) ───────────────────────────
    for c in range(cols):
        label = str(c + 1)
        tw = drw.textlength(label, font=fnt)
        x = (c + 1) * cell_size + (cell_size - tw) / 2
        y = H - cell_size + (cell_size - font_sz) / 2
        drw.text((x, y), label, fill="black", font=fnt)

    for r in range(rows):
        label = str(rows - r)
        tw = drw.textlength(label, font=fnt)
        tx = (cell_size - tw) / 2
        ty = r * cell_size + (cell_size - font_sz) / 2
        drw.text((tx, ty), label, fill="black", font=fnt)

    # ─── draw grid lines ───────────────────────────────────────────────────
    if full:
        # every vertical line
        for c in range(cols + 1):
            x = (c + 1) * cell_size
            drw.line([(x, 0), (x, H - cell_size)], fill="black", width=line_w)

        # every horizontal line
        for r in range(rows + 1):
            y = r * cell_size
            drw.line([(cell_size, y), (W, y)], fill="black", width=line_w)
    else:
        # compact style: only axis lines
        # vertical y-axis
        drw.line([(cell_size, 0), (cell_size, H - cell_size)],
                 fill="black", width=line_w)
        # horizontal x-axis
        drw.line([(cell_size, H - cell_size), (W, H - cell_size)],
                 fill="black", width=line_w)

        # little “tick” rectangles to mark each grid-cell step
        for r in range(rows):
            y0 = r * cell_size
            y1 = (r + 1) * cell_size
            # left-hand tick
            drw.rectangle([(0, y0), (cell_size, y1)], outline="black")
        for c in range(cols):
            x0 = (c + 1) * cell_size
            x1 = (c + 2) * cell_size
            # bottom tick
            drw.rectangle([(x0, H - cell_size), (x1, H)], outline="black")

    # ─── coordinate centres for each cell ──────────────────────────────────
    positions = {}
    for gx in range(1, cols + 1):
        for gy in range(1, rows + 1):
            cx = (gx + 0.5) * cell_size
            cy = (rows - gy + 0.5) * cell_size   # invert y
            positions[f"x{gx}y{gy}"] = (int(cx), int(cy))

    return im, positions



def cells_to_pixels(res=50, cell_size=12, header_size=12):
    # Define the size of the grid
    rows = res
    cols = res
    
    img_width = (cols + 1) * cell_size
    img_height = (rows + 1) * cell_size

    positions={}
    # Draw the grid
    for i in range(rows)[::-1]:
        for j in range(cols):
            # Calculate the position of the text
            text = f"x{j + 1}y{i + 1}"
            
            center_y = int(img_height - cell_size - (i * cell_size) - cell_size / 2)
            center_x = int(j * cell_size + cell_size / 2 + cell_size)
            positions[text] = (center_x, center_y)

    return positions


# =========================
# ===== LLM related =======
# =========================
def image_to_str(image: Image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode('utf-8')
    return image



# =================================
# ===== SVG process related =======
# =================================
def bezier_point(P, t):
    """Calculate a point on the Bézier curve for a given t."""
    return (1-t)**3 * P[0] + 3*(1-t)**2 * t * P[1] + 3*(1-t) * t**2 * P[2] + t**3 * P[3]


def estimate_bezier_control_points_helper(sampled_points, t_values):
    n = len(sampled_points)
    
    if n == 1:
        # Linear interpolation: the control points are simply the two points
        P0 = np.array(sampled_points[0])
        P1 = np.array(sampled_points[0]).astype(np.float64) + 0.0001
        return np.array([P0, P1])
        
    if n == 2:
        # Linear interpolation: the control points are simply the two points
        P0 = np.array(sampled_points[0])
        P1 = np.array(sampled_points[1])
        return np.array([P0, P1])

    if n > len(t_values):
        t_values = np.linespace(0,1,n)
    
    elif n == 3:
        # Quadratic Bézier curve: we need to solve for three control points
        A = np.zeros((n, 3))
        for i in range(n):
            t = t_values[i]
            A[i, 0] = (1-t)**2
            A[i, 1] = 2*(1-t)*t
            A[i, 2] = t**2
        
        # Points (flattened)
        B = np.array(sampled_points).reshape(-1, 2)  # Assuming 2D points
        
        # Solve the system (least squares)
        P = np.linalg.lstsq(A, B, rcond=None)[0]
        return P

    # Matrix A
    A = np.zeros((n, 4))
    for i in range(n):
        t = t_values[i]
        A[i, 0] = (1-t)**3
        A[i, 1] = 3*(1-t)**2 * t
        A[i, 2] = 3*(1-t) * t**2
        A[i, 3] = t**3
    
    # Points (flattened)
    B = np.array(sampled_points).reshape(-1, 2)  # Assuming 2D points
    
    # Solve the system (least squares)
    P = np.linalg.lstsq(A, B, rcond=None)[0]
    return P

    
def estimate_bezier_control_points( sampled_points, t_values):
    if len(sampled_points) != len(t_values):
        t_values = np.linspace(0,1, len(sampled_points))
    P = estimate_bezier_control_points_helper(sampled_points, t_values)

    if len(sampled_points) > 4:
        # Calculate the mean squared error between sampled points and the fitted Bézier curve.
        errors = []
        for i, t in enumerate(t_values):
            B_t = bezier_point(P, t)
            error = np.linalg.norm(B_t - sampled_points[i])
            errors.append(error)
        error = np.mean(errors)
        
        if error > 5 and len(sampled_points) >= 7:
            mid = len(sampled_points) // 2
            left_sampled_points = sampled_points[:mid+1]
            right_sampled_points = sampled_points[mid:]
            left_t_values = np.array(t_values[:mid+1])
            right_t_values = np.array(t_values[mid:])

            if len(left_sampled_points) == 3: # this applies in case we have 7 points
                left_sampled_points.append(right_sampled_points[0])
                left_t_values.append(right_t_values[0])
                
            # Normalize t_values for each segment
            left_t_values = (left_t_values - left_t_values[0]) / (left_t_values[-1] - left_t_values[0])
            right_t_values = (right_t_values - right_t_values[0]) / (right_t_values[-1] - right_t_values[0])

            # Recursively fit curves to each segment
            P_left = estimate_bezier_control_points_helper(left_sampled_points, left_t_values)
            P_right = estimate_bezier_control_points_helper(right_sampled_points, right_t_values)
            P_right[0] = P_left[-1] # I added this to have the long strokes look more connected
            return [P_left, P_right]
    return [P]


def get_control_points(strokes_all, t_values_all, cells_to_pixels_map):
    net_points = []      
    for j in range(len(strokes_all)):
        sampled_cells = strokes_all[j]
        t_values = t_values_all[j]
        sampled_points = []
        for cell in sampled_cells:
            y,x = cells_to_pixels_map[cell]
            sampled_points.append([y,x])
        points_lst = estimate_bezier_control_points(sampled_points, t_values)
        net_points.append(points_lst)
    return net_points


def get_control_points_single_stroke(strokes_all, t_values_all, cells_to_pixels_map):
    sampled_points = []
    for cell in strokes_all:
        y,x = cells_to_pixels_map[cell]
        sampled_points.append([y,x])
    points_lst = estimate_bezier_control_points(sampled_points, t_values_all)
    return points_lst


def create_svg_path_data(control_points):
    # Start the path with 'M' for the first point
    # print("control_points", control_points[0])
    path_data = 'M ' + np.array2string(np.array(control_points[0]), formatter={'float_kind':lambda x: "%.2f" % x}, separator=' ')[1:-1]    
    # Add 'L' for each subsequent point

    # check if point
    if len(control_points) == 1:
        path_data += ' '
    # check if line
    elif len(control_points) == 2:
        path_data += ' L '
    # check if quadratic
    elif len(control_points) == 3:
        path_data += ' Q '
    # check if cubic
    elif len(control_points) == 4:
        path_data += ' C '
    
    # path_data += ' C '
    for point in control_points[1:]:
        # print("pt", point[0], point[1])
        path_data += str(point[0]) + " " + str(point[1]) + " "
    
    # Return the complete 'd' attribute string
    return path_data


def format_svg(all_control_points, dim, stroke_width):
    svg_width, svg_height = dim
    sketch_text_svg = f"""<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">\n"""        
    for i, path in enumerate(all_control_points):
        gropu_text = f"""<g id="s{i + 1}" stroke="black" stroke-width="{stroke_width}" fill="none" stroke-linecap="round">\n"""
        for sub_path_cp in path:  #sometimes 1 or 2 
            path_data = create_svg_path_data(sub_path_cp)
            gropu_text += f"""<path d="{path_data}"/>\n"""
        gropu_text += "</g>\n"
        sketch_text_svg += gropu_text
    sketch_text_svg += "</svg>"
    return sketch_text_svg


def format_svg_single_stroke(
        group,
        dim,
        stroke_width,
        stroke_counter=None,
        *,                      # ← makes the next args keyword-only
        group_id=None,          # NEW  – pass a custom label here
        stroke_color="black"):

    """
    Build one <g> element containing all <path>s for a single stroke.

    Parameters
    ----------
    group         : list   – list-of-sub-paths (your existing control-points data)
    dim           : tuple  – (w, h) – kept for backwards-compat even if unused
    stroke_width  : float
    stroke_counter: int or None – still accepted for legacy callers
    group_id      : str or None – SVG id attribute.  If None we fall back to
                                  "s{stroke_counter}" (previous behaviour).
    stroke_color  : str   – any valid CSS colour.
    """

    # pick an id for the <g>
    if group_id is None:
        # maintain the old numeric ids when no custom label supplied
        group_id = f"s{stroke_counter or 0}"

    # build the SVG text
    svg_lines = [
        f'<g id="{group_id}" stroke="{stroke_color}" '
        f'stroke-width="{stroke_width}" fill="none" stroke-linecap="round">'
    ]

    for sub_path_cp in group:
        path_data = create_svg_path_data(sub_path_cp)
        svg_lines.append(f'    <path d="{path_data}"/>')

    svg_lines.append('</g>\n')
    return "\n".join(svg_lines)



# Note that this parse only the *first* part in the text in which you have the <strokes> </strokes> tags.
def parse_xml_string(llm_output, res):

    strokes_start_marker = "<strokes>"
    strokes_end_marker = "</strokes>"

    # Find the start and end indices of the JSON string
    start_index = llm_output.find(strokes_start_marker)
    if start_index != -1:
        # start_index += len(strokes_start_marker)  # Move past the marker
        end_index = llm_output.find(strokes_end_marker, start_index)
    else:
        return None  # XML markers not found

    if end_index == -1:
        return None  # End marker not found

    # Extract the JSON string
    strokes_str = llm_output[start_index:end_index + len(strokes_end_marker)].strip()#[:-1]
    xml_str = f"<wrap>{strokes_str}</wrap>"
    # Parse the XML string
    root = ET.fromstring(xml_str)
    
    # Initialize lists to hold strokes and t_values
    strokes_list = "[\n"
    t_values_list = "[\n"
    
    # Iterate over all the strokes
    for stroke in root.find('strokes'):
        # Extract points and clean them up
        points_text = stroke.find('points').text
    
        # Extract t_values and convert them to float
        t_values_text = stroke.find('t_values').text
    
        # Append to the lists
        strokes_list += f"[{points_text}],\n"
        t_values_list += f"[{t_values_text}],\n"
    
    strokes_list = re.sub(r'\d+', lambda x: str(min(int(x.group()), res)), strokes_list)
    strokes_list = re.sub(r'\d+', lambda x: str(max(int(x.group()), 1)), strokes_list)
    
    strokes_list += "]"
    t_values_list += "]"
    return strokes_list, t_values_list




def parse_xml_string_single_stroke(xml_text: str, res: int, stroke_no: int):
    """
    Extract a single stroke <s#> … </s#> from either
      • a full LLM answer  *or*
      • an already-isolated stroke block,
    and return two strings:
        points_list_str, t_values_list_str
    """

    # ------------------------------------------------------------------
    # 1)  Get the <s#> … </s#> block
    # ------------------------------------------------------------------
    single_block_pattern = r"^\s*<s\d+>"
    if re.match(single_block_pattern, xml_text):          # already a block
        stroke_xml = xml_text.strip()
    else:                                                 # old behaviour
        start_tag  = f"<s{stroke_no}>"
        end_tag    = f"</s{stroke_no}>"
        start_idx  = xml_text.find(start_tag)
        if start_idx == -1:
            raise ValueError(f"{start_tag} not found")
        end_idx    = xml_text.find(end_tag, start_idx)
        if end_idx == -1:
            raise ValueError(f"{end_tag} not found")
        stroke_xml = xml_text[start_idx:end_idx + len(end_tag)].strip()

    # ------------------------------------------------------------------
    # 2)  Parse XML
    # ------------------------------------------------------------------
    print("strokexml", stroke_xml)
    root = ET.fromstring(f"<wrap>{stroke_xml}</wrap>")

    print(f"stroke no: {stroke_no}")
    print("root", root)
    stroke = root.find(f"s{stroke_no}")
    if stroke is None:
        raise ValueError(f"malformed <s{stroke_no}> block")

    pts_text = stroke.findtext("points",   default="").strip()
    t_text   = stroke.findtext("t_values", default="").strip()

    # ------------------------------------------------------------------
    # 3)  Normalise -- clamp grid indices + tidy t-values
    # ------------------------------------------------------------------
    pts_text = re.sub(
        r"x(\d+)y(\d+)",
        lambda m: f"x{max(1, min(int(m.group(1)), res))}"
                  f"y{max(1, min(int(m.group(2)), res))}",
        pts_text
    )

    # accept 0.5, .5, 0.50 … and re-emit with two decimals
    t_vals = [f"{float(s):.2f}" for s in re.split(r"[\s,]+", t_text) if s]
    t_text = ", ".join(t_vals)

    return f"[{pts_text}]", f"[{t_text}]"



# =====================================
# ===== Collaborative Sketching =======
# =====================================
def get_cur_stroke_text(stroke_counter, llm_output):
    start_marker = f"<s{stroke_counter}>"
    end_marker = f"</s{stroke_counter}>"

    # Find the start and end indices of the JSON string
    start_index = llm_output.find(start_marker)
    if start_index != -1:
        # start_index += len(strokes_start_marker)  # Move past the marker
        end_index = llm_output.find(end_marker, start_index)
    else:
        return ""  # XML markers not found

    if end_index == -1:
        return ""  # End marker not found

    # Extract the JSON string
    strokes_str = llm_output[start_index:end_index + len(end_marker)].strip()#[:-1]
    return strokes_str
