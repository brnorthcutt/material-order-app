# streamlit_app.py
import streamlit as st
import pdfplumber
import io
import re
import math
from collections import defaultdict, Counter
import pandas as pd
from PIL import Image
import pytesseract


# If you are on Windows and tesseract is not in PATH, uncomment & set path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------- Configuration ----------
WASTE_PERCENT = 0.10  # 10% waste
CEILING_ALLOWED_LENGTHS = [10, 12, 14, 16]  # only order these for ceiling joists
MIN_ORDER_LENGTH = 10  # rafters start at 10 ft
SPLIT_PIECE_LENGTH_FOR_LONG_RAFTER = 14  # > most common -> break into 14ft pieces
FIND_TITLES = [
    r'ceiling framing schedule',
    r'ceiling joists',
    r'roof framing schedule',
    r'roof framing',
    r'roof rafters',
    r'master main floor',
    r'main floor',
    r'plan notes',
    r'notes'
]


# ---------- Helpers ----------
def parse_length_to_feet(s: str):
    if not s:
        return None
    s = s.strip()
    # normalize quotes and unicode fractions
    s = s.replace('\u2019', "'").replace('\u201d', '"').replace('”', '"').replace('’', "'")
    s = s.replace('¼', '1/4').replace('½', '1/2').replace('¾', '3/4')
    s = s.replace(',', '')
    # find feet part
    feet = 0.0
    inches = 0.0
    m = re.search(r"(\d+)\s*'", s)
    if m:
        feet = float(m.group(1))
    # fraction with inches like 6 1/2"
    m2 = re.search(r"(\d+)\s*(?:-|\s)?\s*(\d+/\d+)\s*\"", s)
    if m2:
        inches = float(m2.group(1)) + eval(m2.group(2))
    else:
        m3 = re.search(r"(\d+(\.\d+)?)\s*\"", s)
        if m3:
            inches = float(m3.group(1))
        else:
            m4 = re.search(r"(\d+/\d+)\s*\"", s)
            if m4:
                inches = eval(m4.group(1))
    # if nothing parsed, maybe it's just a number in feet
    if feet == 0 and inches == 0:
        m5 = re.search(r"^(\d+(\.\d+)?)\s*(ft|feet)?$", s.lower())
        if m5:
            return float(m5.group(1))
    total = feet + inches / 12.0
    return total if total > 0 else None


def ceil_int(x):
    try:
        return int(math.ceil(float(x)))
    except:
        return int(x)


def apply_waste_per_board_dict(board_counts_dict):
    out = {}
    for k, v in board_counts_dict.items():
        out[k] = math.ceil(v * (1 + WASTE_PERCENT))
    return out


def find_pages_with_key_text(pdf, key_regex):
    pages = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if re.search(key_regex, text, re.IGNORECASE):
            pages.append(i)
        else:
            # OCR fallback quick check: render as image and do OCR
            try:
                img = page.to_image(resolution=150).original
                ocr_text = pytesseract.image_to_string(img)
                if re.search(key_regex, ocr_text, re.IGNORECASE):
                    pages.append(i)
                # don't accumulate full OCR here
            except Exception:
                pass
    return pages


def extract_tables_from_pdf_page(page):
    results = []
    try:
        tables = page.extract_tables()
    except Exception:
        tables = []
    for t in tables:
        if not t or len(t) < 2:
            continue
        header = [ (c or "").strip() for c in t[0] ]
        for row in t[1:]:
            rowd = {}
            for i, cell in enumerate(row):
                key = header[i] if i < len(header) else f"col{i}"
                rowd[key.strip()] = (cell or "").strip()
            results.append(rowd)
    return results


def ocr_page_text(page):
    try:
        img = page.to_image(resolution=250).original
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return ""


def parse_framing_schedule_rows(table_rows):
    parsed = []
    for r in table_rows:
        kv = {k.lower(): v for k, v in r.items()}
        width_val = depth_val = length_val = count_val = None
        for key in kv:
            k = key.lower()
            if 'width' in k or (k.strip() in ('w','member','size','board')):
                width_val = kv[key] if width_val is None else width_val
            if 'depth' in k or k.strip() in ('d','thickness'):
                depth_val = kv[key] if depth_val is None else depth_val
            if 'length' in k or 'l' == k.strip():
                length_val = kv[key] if length_val is None else length_val
            if 'count' in k or 'qty' in k or 'quantity' in k or k.strip() in ('no.','count'):
                count_val = kv[key] if count_val is None else count_val
        # fallback combined cell like "2x6x14"
        if not (width_val and depth_val and length_val):
            combined = None
            for cell in r.values():
                if isinstance(cell, str) and re.search(r'\d+\s*[xX]\s*\d+', cell):
                    combined = cell
                    break
            if combined:
                nums = re.findall(r'\d+', combined)
                if len(nums) >= 3:
                    width_val = nums[0]
                    depth_val = nums[1]
                    length_val = nums[2]
        # parse fields
        try:
            w = int(re.findall(r'\d+', str(width_val))[0]) if width_val is not None else None
        except:
            w = None
        try:
            d = int(re.findall(r'\d+', str(depth_val))[0]) if depth_val is not None else None
        except:
            d = None
        try:
            lft = None
            if length_val:
                lft = parse_length_to_feet(str(length_val))
            else:
                # search for any length in row cells
                for v in r.values():
                    if isinstance(v, str) and re.search(r"\d+'\s*\d*", v):
                        lft = parse_length_to_feet(v)
                        if lft:
                            break
        except:
            lft = None
        try:
            c = int(re.findall(r'\d+', str(count_val))[0]) if count_val is not None else None
        except:
            # fallback last number
            nums = []
            for v in r.values():
                if isinstance(v, str):
                    found = re.findall(r'\d+', v)
                    if found:
                        nums.extend(found)
            c = int(nums[-1]) if nums else None
        if w and d and lft and c:
            parsed.append({'width': w, 'depth': d, 'length_ft': lft, 'count': c})
    return parsed


def parse_framing_from_raw_text(txt):
    entries = []
    lines = txt.splitlines()
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if re.search(r'\d+\s*[xX]\s*\d+', ln) and re.search(r'\d+\'', ln):
            m = re.search(r'(\d+)\s*[xX]\s*(\d+)', ln)
            if not m:
                continue
            w = int(m.group(1)); d = int(m.group(2))
            ml = re.search(r'(\d+\'(?:\s*\d+/\d+\"|\s*\d+\"|\s*)?)', ln)
            if ml:
                lft = parse_length_to_feet(ml.group(1))
            else:
                lft = None
            found_nums = re.findall(r'\b(\d+)\b', ln)
            count = int(found_nums[-1]) if found_nums else 1
            if w and d and lft:
                entries.append({'width': w, 'depth': d, 'length_ft': lft, 'count': count})
    return entries


# ---------- Business rules implementations ----------
def compute_ceiling_joists(framing_rows):
    analysis = []
    tally = defaultdict(int)
    for r in framing_rows:
        key_base = f"{r['width']}x{r['depth']}"
        length = r['length_ft']
        count = r['count']
        analysis.append(f"Found ceiling schedule row: {count} x {key_base} x {length:.3f} ft")
        half = length / 2.0
        if any(abs(half - L) < 1e-6 for L in CEILING_ALLOWED_LENGTHS):
            chosen_len = int(round(half))
            tally[f"{key_base}x{chosen_len}"] += count * 2
            analysis.append(f" -> split each into 2 x {chosen_len} ft pieces (total {count*2})")
        else:
            chosen = None
            for L in CEILING_ALLOWED_LENGTHS:
                if length <= L:
                    chosen = L
                    break
            if chosen is None:
                chosen = CEILING_ALLOWED_LENGTHS[-1]
            tally[f"{key_base}x{chosen}"] += count
            analysis.append(f" -> order {count} x {key_base} x {chosen} ft")
    condensed = apply_waste_per_board_dict(tally)
    analysis.append("Applied 10% waste per distinct ceiling board type.")
    return condensed, analysis


def compute_rafters(fr_rows):
    analysis = []
    tally = defaultdict(int)
    cnt = Counter()
    for r in fr_rows:
        cnt[r['length_ft']] += r['count']
    most_common_len = cnt.most_common(1)[0][0] if cnt else 16
    analysis.append(f"Most common rafter length: {most_common_len:.3f} ft")
    allowed = [L for L in range(MIN_ORDER_LENGTH, int(math.ceil(most_common_len)) + 1, 2)]
    if not allowed:
        allowed = [10,12,14,16]
    analysis.append(f"Allowed rafter order lengths: {allowed}")
    for r in fr_rows:
        base = f"{r['width']}x{r['depth']}"
        l = r['length_ft']
        c = r['count']
        analysis.append(f"Found rafter row: {c} x {base} x {l:.3f} ft")
        half = l / 2.0
        if any(abs(half - L) < 1e-6 for L in allowed):
            chosen = int(round(half))
            tally[f"{base}x{chosen}"] += c * 2
            analysis.append(f" -> split into 2 x {chosen} ft pieces")
            continue
        if l > most_common_len:
            pieces = math.ceil(l / SPLIT_PIECE_LENGTH_FOR_LONG_RAFTER)
            tally[f"{base}x{SPLIT_PIECE_LENGTH_FOR_LONG_RAFTER}"] += pieces * c
            analysis.append(f" -> longer than most-common; represent as {pieces} x {SPLIT_PIECE_LENGTH_FOR_LONG_RAFTER} ft pieces (count {pieces*c})")
            continue
        next_even = None
        for L in range(int(math.ceil(l)), int(max(allowed)+3)):
            if L % 2 == 0 and L >= MIN_ORDER_LENGTH:
                next_even = L
                break
        if next_even is None:
            next_even = allowed[-1]
        if (next_even - l) <= 1.0:
            candidate = next_even + 2
            if candidate > most_common_len:
                candidate = int(most_common_len)
            chosen_len = candidate
            analysis.append(f" -> within 1 ft of {next_even}, rounding up to {chosen_len} ft per rule")
        else:
            chosen_len = next_even if next_even <= most_common_len else int(most_common_len)
            analysis.append(f" -> rounding up to {chosen_len} ft")
        chosen_final = None
        for L in allowed:
            if chosen_len <= L:
                chosen_final = L
                break
        if chosen_final is None:
            chosen_final = allowed[-1]
        tally[f"{base}x{chosen_final}"] += c
        analysis.append(f" -> ordered as {c} x {base} x {chosen_final} ft")
    condensed = apply_waste_per_board_dict(tally)
    analysis.append("Applied 10% waste per distinct rafter board type.")
    return condensed, analysis


def compute_drywall(master_spaces, plan_notes_text, perimeter_override=None):
    analysis = []
    total_ceiling_sheets = 0
    ceiling_by_space = {}
    for sp in master_spaces:
        if sp.get('exclude','').lower() == 'porch':
            analysis.append(f"Excluding porch: {sp.get('name')}")
            continue
        L = sp.get('length_ft')
        W = sp.get('width_ft')
        if L is None or W is None:
            analysis.append(f"Missing dims for {sp.get('name')}; skipping")
            continue
        area = L * W
        sheets_raw = math.ceil(area / 48.0)
        sheets_total = math.ceil(sheets_raw * (1 + WASTE_PERCENT))
        ceiling_by_space[sp['name']] = {'area_sf': round(area,3), 'sheets_raw': sheets_raw, 'sheets_total': sheets_total}
        total_ceiling_sheets += sheets_total
        analysis.append(f"Ceiling {sp['name']}: area {area:.2f} sf -> raw {sheets_raw} -> with waste {sheets_total}")
    # wall height
    wall_height = None
    mh = re.search(r'wall height[:\s]*([0-9\.\'\"]+)', plan_notes_text, re.IGNORECASE)
    if mh:
        wall_height = parse_length_to_feet(mh.group(1))
        analysis.append(f"Found wall height in plan notes: {wall_height} ft")
    else:
        # fallback search for '9' or '8' in notes
        m2 = re.search(r'\b(9|8)\s*\'', plan_notes_text)
        if m2:
            wall_height = float(m2.group(1))
            analysis.append(f"Found wall height approx: {wall_height} ft")
    if wall_height is None:
        wall_height = 8
        analysis.append("Wall height not found; defaulting to 8 ft.")
    # perimeter
    perimeter_total = None
    mper = re.search(r'perimeter[:\s]*([0-9\.\,]+)', plan_notes_text, re.IGNORECASE)
    if mper:
        try:
            perimeter_total = float(mper.group(1).replace(',', ''))
            analysis.append(f"Found perimeter in notes: {perimeter_total} lf")
        except:
            perimeter_total = None
    if perimeter_override:
        perimeter_total = perimeter_override
        analysis.append(f"Using perimeter override: {perimeter_total} lf")
    if perimeter_total is None:
        analysis.append("Exterior perimeter not found; UI will prompt for perimeter override if needed.")
        perimeter_total = 0
    interior_partitions = 0
    mpart = re.search(r'interior partition[s]?:\s*([0-9\.]+)', plan_notes_text, re.IGNORECASE)
    if mpart:
        interior_partitions = float(mpart.group(1))
    total_perimeter_lf = perimeter_total + interior_partitions
    wall_area_gross = total_perimeter_lf * wall_height
    openings_pct = 0.10
    wall_area_net = wall_area_gross * (1 - openings_pct)
    sheets_raw_stretch = math.ceil(wall_area_net / 54.0) if wall_area_net > 0 else 0
    sheets_total_stretch = math.ceil(sheets_raw_stretch * (1 + WASTE_PERCENT))
    analysis.append(f"Stretch rock: gross {wall_area_gross:.2f} sf -> net {wall_area_net:.2f} sf -> raw {sheets_raw_stretch} -> with waste {sheets_total_stretch}")
    return {
        'ceiling': {'by_space': ceiling_by_space, 'total_sheets': total_ceiling_sheets},
        'stretch_rock': {'perimeter_total_lf': total_perimeter_lf, 'wall_height_ft': wall_height, 'sheets_raw': sheets_raw_stretch, 'sheets_total': sheets_total_stretch},
        'analysis': analysis
    }


# ---------- PDF orchestration ----------
def process_pdf_bytes(pdf_bytes, perimeter_override=None):
    buf = io.BytesIO(pdf_bytes)
    analysis = []
    ceiling_rows = []
    rafter_rows = []
    master_spaces = []
    plan_notes_text = ""
    with pdfplumber.open(buf) as pdf:
        # search pages for keys
        for key in FIND_TITLES:
            pages = find_pages_with_key_text(pdf, key)
            if pages:
                analysis.append(f'Found pages for pattern "{key}": {pages}')
        # specifically find schedule pages
        ceiling_pages = []
        rafter_pages = []
        master_pages = []
        notes_pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            ocr_text = ""
            # quick OCR only if needed
            if not text or any(term in text.lower() for term in ['schedule', 'framing', 'ceiling', 'roof', 'main floor', 'plan notes']):
                try:
                    img = page.to_image(resolution=200).original
                    ocr_text = pytesseract.image_to_string(img)
                except Exception:
                    ocr_text = ""
            combined_text = (text + "\n" + ocr_text).lower()
            if re.search(r'ceiling framing schedule|ceiling joists', combined_text):
                ceiling_pages.append(i)
            if re.search(r'roof framing schedule|roof framing|roof rafters', combined_text):
                rafter_pages.append(i)
            if re.search(r'master main floor|main floor|master floor|main house', combined_text):
                master_pages.append(i)
            if re.search(r'plan notes|notes', combined_text):
                notes_pages.append(i)
            # collect plan notes text for later
            if combined_text:
                plan_notes_text += "\n" + combined_text


        analysis.append(f"Ceiling pages: {ceiling_pages}")
        analysis.append(f"Rafter pages: {rafter_pages}")
        analysis.append(f"Master/Main pages: {master_pages}")
        analysis.append(f"Notes pages: {notes_pages}")


        # Extract framing tables or parse by OCR fallback
        for pnum in ceiling_pages:
            page = pdf.pages[pnum]
            tables = extract_tables_from_pdf_page(page)
            parsed = parse_framing_schedule_rows(tables)
            if parsed:
                ceiling_rows.extend(parsed)
            else:
                text = page.extract_text() or ocr_page_text(page)
                plan_notes_text += "\n" + text
                parsed2 = parse_framing_from_raw_text(text)
                ceiling_rows.extend(parsed2)


        for pnum in rafter_pages:
            page = pdf.pages[pnum]
            tables = extract_tables_from_pdf_page(page)
            parsed = parse_framing_schedule_rows(tables)
            if parsed:
                rafter_rows.extend(parsed)
            else:
                text = page.extract_text() or ocr_page_text(page)
                plan_notes_text += "\n" + text
                parsed2 = parse_framing_from_raw_text(text)
                rafter_rows.extend(parsed2)


        # master/main pages: extract spaces dimensions
        for pnum in master_pages:
            page = pdf.pages[pnum]
            text = page.extract_text() or ocr_page_text(page)
            plan_notes_text += "\n" + text
            matches = re.findall(r'([A-Za-z0-9\s\-\_]{2,30})\s+(\d+\'(?:\s*\d+/\d+\"|\s*\d+\"|\s*)?)\s*[xX]\s*(\d+\'(?:\s*\d+/\d+\"|\s*\d+\"|\s*)?)', text)
            for m in matches:
                nm = m[0].strip()
                l = parse_length_to_feet(m[1])
                w = parse_length_to_feet(m[2])
                if l and w:
                    master_spaces.append({'name': nm, 'length_ft': l, 'width_ft': w})


        for pnum in notes_pages:
            page = pdf.pages[pnum]
            t = page.extract_text() or ocr_page_text(page)
            plan_notes_text += "\n" + (t or "")


    if not ceiling_rows:
        analysis.append("No ceiling framing rows parsed automatically.")
    else:
        analysis.append(f"Parsed {len(ceiling_rows)} ceiling framing rows.")
    if not rafter_rows:
        analysis.append("No roof framing rows parsed automatically.")
    else:
        analysis.append(f"Parsed {len(rafter_rows)} rafter framing rows.")
    if not master_spaces:
        analysis.append("No master/main floor spaces parsed automatically.")


    ceiling_condensed, ceiling_analysis = compute_ceiling_joists(ceiling_rows)
    rafter_condensed, rafter_analysis = compute_rafters(rafter_rows)
    drywall_results = compute_drywall(master_spaces, plan_notes_text, perimeter_override=perimeter_override)


    combined = defaultdict(int)
    for k, v in ceiling_condensed.items():
        combined[k] += v
    for k, v in rafter_condensed.items():
        combined[k] += v
    combined['DRYWALL_4x12_sheets'] = drywall_results['ceiling']['total_sheets']
    combined['STRETCHROCK_54x12_sheets'] = drywall_results['stretch_rock']['sheets_total']


    full_analysis = []
    full_analysis.extend(analysis)
    full_analysis.append('--- Ceiling Analysis ---')
    full_analysis.extend(ceiling_analysis)
    full_analysis.append('--- Rafter Analysis ---')
    full_analysis.extend(rafter_analysis)
    full_analysis.append('--- Drywall Analysis ---')
    full_analysis.extend(drywall_results['analysis'])


    condensed_list = [{'item': k, 'quantity': v} for k, v in combined.items()]


    return {
        'condensed_list': condensed_list,
        'analysis': full_analysis,
        'details': {
            'ceiling': ceiling_condensed,
            'rafter': rafter_condensed,
            'drywall': drywall_results
        },
        'parsed_rows': {
            'ceiling_rows': ceiling_rows,
            'rafter_rows': rafter_rows,
            'master_spaces': master_spaces
        }
    }


# ---------- Streamlit UI (single page with analysis + condensed total) ----------
st.set_page_config(page_title="Material Order Extractor", layout="wide")
st.title("Material Order Extractor — Ceiling Joists, Drywall & Stretch Rock, Roof Rafters")
st.write("Upload the full PDF house plan set. The app will attempt to find and extract the Ceiling Framing Schedule, Roof Framing Schedule, and Master/Main Floor measurements, then apply your rules to produce a condensed material order.")


with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload plans PDF", type=["pdf"])
    perimeter_override = st.number_input("Optional: Exterior perimeter (lf) override (enter 0 to ignore)", min_value=0.0, value=0.0, step=1.0)
    run_button = st.button("Process PDF")


if uploaded_file and run_button:
    pdf_bytes = uploaded_file.read()
    st.info("Processing PDF — this can take a few seconds (OCR may be used)...")
    result = process_pdf_bytes(pdf_bytes, perimeter_override=perimeter_override if perimeter_override > 0 else None)


    st.header("Parsed Inputs (verify)")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Ceiling Framing Rows")
        cr = result['parsed_rows']['ceiling_rows']
        if cr:
            st.table(pd.DataFrame(cr))
        else:
            st.write("No ceiling schedule parsed. If your schedule didn't parse, export it to CSV and upload or share a sample for tuning.")
    with cols[1]:
        st.subheader("Rafter Framing Rows")
        rr = result['parsed_rows']['rafter_rows']
        if rr:
            st.table(pd.DataFrame(rr))
        else:
            st.write("No rafter schedule parsed. If your schedule didn't parse, export it to CSV and upload or share a sample for tuning.")


    st.subheader("Master/Main Floor Spaces (detected)")
    ms = result['parsed_rows']['master_spaces']
    if ms:
        st.table(pd.DataFrame(ms))
    else:
        st.write("No spaces auto-detected. The app will need either clearer plan formatting or manual dimensions.")


    st.header("Step-by-step analysis")
    for line in result['analysis']:
        st.text(line)


    st.header("Condensed Total Order")
    df = pd.DataFrame(result['condensed_list']).sort_values(by='item')
    st.table(df)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download condensed order CSV", csv_bytes, file_name="material_order.csv", mime="text/csv")
else:
    st.info("Upload a PDF and press Process PDF to start.")



import streamlit as st
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io
import re


st.set_page_config(page_title="House Plan Schedule Reader", layout="wide")


st.title("🏠 Construction Schedule Extractor")
st.write("Upload your full set of plans (PDF). This tool will scan for schedules like Ceiling Joists, Roof Rafters, and Drywall information using OCR.")


# ---- OCR + PDF Conversion ----
def extract_text_and_images_from_pdf(pdf_file):
    """Convert PDF to images and extract text using OCR."""
    text = ""
    images = []
    pages = convert_from_path(pdf_file)
    
    for i, page in enumerate(pages, start=1):
        st.write(f"Processing page {i}...")
        page_text = pytesseract.image_to_string(page)
        text += f"\n--- PAGE {i} ---\n" + page_text
        
        # Save image bytes for potential debugging
        img_byte_arr = io.BytesIO()
        page.save(img_byte_arr, format='PNG')
        images.append(img_byte_arr.getvalue())
    
    return text, images


# ---- Identify schedule type ----
def identify_schedule_type(text):
    """Detect what kind of schedule or chart the text represents."""
    lower_text = text.lower()


    if "ceiling joist" in lower_text or "ceiling joists layout" in lower_text:
        return "Ceiling Joists Schedule"
    elif "roof rafter" in lower_text or "rafter schedule" in lower_text:
        return "Roof Rafters Schedule"
    elif "drywall" in lower_text or "rock" in lower_text or "stretch rock" in lower_text:
        return "Drywall & Stretch Rock Calculation"
    else:
        return "Unidentified / Other Page"


# ---- Data extraction helpers ----
def extract_measurements(text):
    """Pull out numeric values that look like measurements."""
    # Finds numbers followed by 'ft', 'in', or standalone decimals
    measurements = re.findall(r'\d+\s?(?:ft|in|[.]\d+)?', text)
    return measurements


# ---- Upload Section ----
uploaded_file = st.file_uploader("📤 Upload a PDF of your plans", type=["pdf"])


if uploaded_file:
    with st.spinner("Reading and scanning the PDF..."):
        text, images = extract_text_and_images_from_pdf(uploaded_file)


    st.success("✅ PDF processed with OCR!")


    # ---- Detect Schedules ----
    schedule_type = identify_schedule_type(text)
    st.subheader("Detected Schedule Type:")
    st.info(schedule_type)


    # ---- Extract measurements ----
    measurements = extract_measurements(text)
    st.subheader("Extracted Measurements:")
    if measurements:
        st.write(measurements)
    else:
        st.write("No clear measurements detected yet. Try checking if the chart has numeric text.")


    # ---- Display raw text ----
    with st.expander("📄 View Extracted Text"):
        st.text_area("OCR Text Output", text, height=400)
    
    # ---- Debug images (optional) ----
    with st.expander("🖼 View Pages as Images"):
        for i, img_bytes in enumerate(images):
            st.image(img_bytes, caption=f"Page {i+1}")
else:
    st.info("Please upload your plan set PDF to begin.")


