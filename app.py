from pathlib import Path
import re
import math

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import pytesseract
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Spectrum Image → MGF", page_icon="📈", layout="wide")
st.title("Spectrum Image → MGF")
st.caption("Select horizontal or vertical m/z labels with boxes, read them by OCR, estimate relative intensity, export MGF, and validate.")

DECIMAL_RE = re.compile(r"(\d{2,5}[.,]\d{2,8})")

# -----------------------------
# LOGOs (optional)
# -----------------------------
STATIC_DIR = Path(__file__).parent / "static"
for logo_name in ["LAABio.png"]: #"logo_massQL.png", 
    p = STATIC_DIR / logo_name
    try:
        from PIL import Image
        st.image(Image.open(p), use_container_width=True)
    except Exception:
        pass

st.sidebar.divider()

def gray_image(img):
    a = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

def detect_baseline(gray, threshold=235):
    h,w = gray.shape
    mask = gray < threshold
    y0, y1 = int(h*0.70), int(h*0.985)
    d = mask[y0:y1,:].mean(axis=1)
    cand = np.where(d > 0.45)[0]
    if not len(cand):
        return int(h*0.92)
    groups, cur = [], [cand[0]]
    for i in cand[1:]:
        if i-cur[-1] <= 2: cur.append(i)
        else: groups.append(cur); cur=[i]
    groups.append(cur)
    # avoid the lower image frame; choose the upper strong horizontal group
    groups = sorted(groups, key=lambda g: y0 + np.mean(g))
    return int(round(y0 + np.mean(groups[0])))

@st.cache_data(show_spinner=False)
def full_ocr(gray_bytes, shape):
    gray = np.frombuffer(gray_bytes, dtype=np.uint8).reshape(shape)
    big = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    df = pytesseract.image_to_data(big, config="--psm 11", output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"]).copy()
    for c in ["left","top","width","height"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")/2.5
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce")
    df["cx"] = df["left"] + df["width"]/2
    df["cy"] = df["top"] + df["height"]/2
    return df

def fit_axis(ocr, baseline, h):
    rows=[]
    for _,r in ocr.iterrows():
        t=str(r["text"]).strip()
        if not re.fullmatch(r"\d{2,4}", t): continue
        v=float(t)
        if not (20 <= v <= 3000): continue
        if not (baseline-5 <= r["cy"] <= min(h, baseline+80)): continue
        if pd.isna(r["conf"]) or r["conf"] < 35: continue
        rows.append((float(r["cx"]),v,float(r["conf"])))
    if len(rows)<2: return None,pd.DataFrame()
    df=pd.DataFrame(rows,columns=["x","mz","conf"]).drop_duplicates(["x","mz"]).sort_values("x")
    x=df["x"].to_numpy(); y=df["mz"].to_numpy()
    best=None
    for i in range(len(df)):
        for j in range(i+1,len(df)):
            if x[j]-x[i] < 20: continue
            m=(y[j]-y[i])/(x[j]-x[i])
            if m<=0: continue
            b=y[i]-m*x[i]
            res=np.abs(y-(m*x+b))
            inl=res<=3
            score=(int(inl.sum()),-float(np.mean(res[inl]**2)) if inl.any() else -999)
            if best is None or score>best[0]: best=(score,inl)
    if best is None: return None,df
    good=df.loc[best[1]].copy()
    m,b=np.polyfit(good["x"],good["mz"],1)
    return {"slope":float(m),"intercept":float(b)},good

def mz_to_x(mz, cal):
    return (float(mz)-cal["intercept"])/cal["slope"]

def trace_height(gray, baseline, x, threshold=245, max_gap=3):
    x=int(round(x))
    if x<0 or x>=gray.shape[1]: return 0
    started=False; gap=0; top=baseline
    for y in range(baseline-1,-1,-1):
        dark=gray[y,x] < threshold
        if dark:
            started=True; top=y; gap=0
        else:
            gap+=1
            if not started and gap>4: return 0
            if started and gap>max_gap: break
    return max(0,baseline-top)

def local_peak(gray, baseline, expected_x, radius=3, threshold=245):
    vals=[]
    for x in range(max(0,int(round(expected_x))-radius), min(gray.shape[1],int(round(expected_x))+radius+1)):
        vals.append((x,trace_height(gray,baseline,x,threshold)))
    if not vals: return np.nan,0
    vals.sort(key=lambda z:(z[1],-abs(z[0]-expected_x)),reverse=True)
    return float(vals[0][0]),float(vals[0][1])


def detect_global_base_peak(gray, baseline, ticks, threshold=245):
    """
    Detect the tallest spectral stick across the entire plotted m/z region.

    This is deliberately independent of OCR boxes and of the peaks selected
    for export. Relative intensities must be normalized to the true base peak
    in the source spectrum, not to the tallest labeled/accepted peak.
    """
    h, w = gray.shape

    if ticks is not None and len(ticks) >= 2:
        xs = np.sort(pd.to_numeric(ticks["x"], errors="coerce").dropna().to_numpy(float))
        if len(xs) >= 2:
            diffs = np.diff(xs)
            spacing = float(np.median(diffs[diffs > 2])) if np.any(diffs > 2) else 40.0
            left = int(max(0, np.floor(xs.min() - 0.80 * spacing)))
            right = int(min(w - 1, np.ceil(xs.max() + 1.05 * spacing)))
        else:
            left, right = int(w * 0.05), int(w * 0.97)
    else:
        left, right = int(w * 0.05), int(w * 0.97)

    # Avoid frame/y-axis artifacts.
    left = max(left, int(w * 0.03))
    right = min(right, int(w * 0.985))

    heights = []
    for x in range(left, right + 1):
        ht = trace_height(
            gray,
            int(baseline),
            x,
            threshold=threshold,
            max_gap=1,
        )
        heights.append((x, float(ht)))

    if not heights:
        return np.nan, 0.0, left, right

    # Ignore implausibly tiny runs.
    valid = [(x, ht) for x, ht in heights if ht >= 5]
    if not valid:
        return np.nan, 0.0, left, right

    # The tallest strict baseline-connected stick is the base peak.
    base_x, base_height = max(valid, key=lambda z: z[1])

    return float(base_x), float(base_height), left, right

def ocr_box(gray, box):
    """
    OCR one user-selected m/z label box.

    The label may be horizontal or rotated 90 degrees. The function tests:
      - original orientation
      - 90° clockwise
      - 90° counter-clockwise

    For each orientation it also tests grayscale and Otsu-thresholded versions.
    The valid decimal reading with the highest OCR confidence is returned.
    """
    x1,y1,x2,y2=[int(v) for v in box]
    crop=gray[
        max(0,y1):min(gray.shape[0],y2),
        max(0,x1):min(gray.shape[1],x2)
    ]

    if crop.size==0:
        return None,"",0,""

    orientations = [
        ("horizontal", crop),
        ("90° clockwise", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)),
        ("90° counter-clockwise", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    outs=[]

    for orientation_name, oriented in orientations:
        for mode in ["gray","otsu"]:
            img=oriented.copy()

            if mode=="otsu":
                img=cv2.threshold(
                    img,0,255,
                    cv2.THRESH_BINARY+cv2.THRESH_OTSU
                )[1]

            # Increase the cropped label substantially for OCR.
            img=cv2.resize(
                img,None,
                fx=4,fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            # Try a single-line interpretation first.
            for psm in [7, 8, 13]:
                d=pytesseract.image_to_data(
                    img,
                    config=(
                        f"--psm {psm} "
                        "-c tessedit_char_whitelist=0123456789."
                    ),
                    output_type=pytesseract.Output.DATAFRAME
                )

                d=d.dropna(subset=["text"])

                # Tesseract may split a value into more than one token.
                # First inspect individual tokens, then the concatenated text.
                token_texts=[]
                token_confs=[]

                for _,r in d.iterrows():
                    txt=str(r["text"]).strip()
                    if not txt:
                        continue

                    token_texts.append(txt)

                    conf=float(r["conf"]) if pd.notna(r["conf"]) else 0
                    token_confs.append(conf)

                    m=DECIMAL_RE.search(txt)
                    if m:
                        try:
                            val=float(m.group(1).replace(",","."))
                            outs.append(
                                (conf,val,txt,orientation_name,mode,psm)
                            )
                        except:
                            pass

                joined="".join(token_texts).replace(" ","")

                m=DECIMAL_RE.search(joined)
                if m:
                    try:
                        val=float(m.group(1).replace(",","."))
                        conf=max(token_confs) if token_confs else 0
                        outs.append(
                            (conf,val,joined,orientation_name,mode,psm)
                        )
                    except:
                        pass

    if not outs:
        return None,"",0,""

    # Highest-confidence valid reading wins.
    outs.sort(reverse=True,key=lambda z:z[0])
    best=outs[0]

    return best[1],best[2],best[0],best[3]

def draw_boxes(img, boxes, pending=None):
    """Draw boxes in the coordinate system of *img*."""
    arr=np.asarray(img.convert("RGB")).copy()
    for i,b in enumerate(boxes,1):
        x1,y1,x2,y2=[int(v) for v in b]
        cv2.rectangle(arr,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(
            arr,str(i),(x1+4,max(18,y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2
        )
    if pending:
        cv2.circle(arr,(int(pending[0]),int(pending[1])),6,(255,0,0),2)
    return Image.fromarray(arr)

def make_interaction_image(img, max_width=1200):
    """
    Create a fixed-size image for clicking.

    streamlit_image_coordinates returns coordinates in the rendered image.
    We therefore render an image ourselves at a known size and DO NOT ask
    Streamlit to resize it to the column width.
    """
    ow, oh = img.size

    if ow <= max_width:
        return img.copy(), 1.0, 1.0

    scale = max_width / float(ow)
    dw = int(round(ow * scale))
    dh = int(round(oh * scale))

    display_img = img.resize((dw, dh), Image.Resampling.LANCZOS)

    # factors to convert display coordinates -> original image coordinates
    sx = ow / float(dw)
    sy = oh / float(dh)

    return display_img, sx, sy

def display_box_to_original(box, sx, sy):
    x1,y1,x2,y2 = box
    return (
        int(round(x1*sx)),
        int(round(y1*sy)),
        int(round(x2*sx)),
        int(round(y2*sy)),
    )


def box_id(box):
    """Stable identifier for one user-selected box in display coordinates."""
    return "|".join(str(int(v)) for v in box)

def spectrum_fig(df,title):
    fig=go.Figure()
    for _,r in df.dropna(subset=["mz","intensity"]).iterrows():
        fig.add_trace(go.Scatter(x=[r["mz"],r["mz"]],y=[0,r["intensity"]],mode="lines",
                                 line=dict(width=1.5),showlegend=False))
    fig.update_layout(title=title,xaxis_title="m/z",yaxis_title="Relative intensity (%)",
                      yaxis_range=[0,105],height=420)
    return fig

def mgf_text(df, spectrum_id, compound_name=None, pepmass=None, charge=None):
    v=df[df["use"].fillna(False)].copy()
    v["mz"]=pd.to_numeric(v["mz"],errors="coerce")
    v["intensity"]=pd.to_numeric(v["intensity"],errors="coerce")
    v=v.dropna(subset=["mz","intensity"]).sort_values("mz")

    # TITLE is the standard free-text spectrum title in MGF.
    # If a compound name is supplied, use it as TITLE.
    # Otherwise fall back to the spectrum/file identifier.
    title_value = (
        str(compound_name).strip()
        if compound_name is not None and str(compound_name).strip()
        else str(spectrum_id).strip()
    )

    lines=["BEGIN IONS",f"TITLE={title_value}"]

    if pepmass is not None:
        lines.append(f"PEPMASS={pepmass:.6f}")

    if charge:
        lines.append(f"CHARGE={charge}")

    for _,r in v.iterrows():
        lines.append(f"{r['mz']:.6f} {r['intensity']:.4f}")

    lines.append("END IONS")
    return "\n".join(lines)+"\n"

uploaded=st.file_uploader("Upload spectrum image",type=["png","jpg","jpeg","tif","tiff","bmp"])
if uploaded is None: st.stop()

img=Image.open(uploaded).convert("RGB")
gray=gray_image(img)
h,w=gray.shape

# Fixed-size image used only for interactive box selection.
# Click coordinates and boxes live in this display coordinate system.
interaction_img, display_to_orig_x, display_to_orig_y = make_interaction_image(
    img,
    max_width=1200,
)
display_w, display_h = interaction_img.size

baseline_auto=detect_baseline(gray)
baseline=st.number_input("Spectrum baseline y (automatic; adjust only if needed)",
                         min_value=0,max_value=h-1,value=int(baseline_auto),step=1)

ocr=full_ocr(gray.tobytes(),gray.shape)
cal,ticks=fit_axis(ocr,int(baseline),h)

if cal:
    st.success(f"m/z axis calibrated automatically from {len(ticks)} tick labels.")
else:
    st.error("Could not calibrate the x-axis automatically. Use an image with visible axis tick values.")
    st.stop()

# Detect the true base peak from the whole plotted spectrum.
base_peak_x, base_peak_height, scan_left, scan_right = detect_global_base_peak(
    gray,
    int(baseline),
    ticks,
    threshold=245,
)

if base_peak_height <= 0:
    st.error("Could not detect the spectrum base peak for intensity normalization.")
    st.stop()

base_peak_mz_approx = cal["slope"] * base_peak_x + cal["intercept"]

bc1, bc2, bc3 = st.columns(3)
bc1.metric("Detected base-peak height", f"{base_peak_height:.1f} px")
bc2.metric("Base-peak x", f"{base_peak_x:.1f} px")
bc3.metric("Approx. base-peak m/z", f"{base_peak_mz_approx:.2f}")

st.caption(
    "Relative intensities are normalized to this tallest stick in the complete "
    "source spectrum, even if that peak is not boxed or exported."
)

with st.expander("Axis calibration details",expanded=False):
    st.dataframe(ticks,use_container_width=True,hide_index=True)
    st.write(f"m/z = {cal['slope']:.8f} × x + {cal['intercept']:.6f}")

st.subheader("1. Select labeled peaks")
st.write(
    "For each peak, click **two opposite corners** to draw a box around its printed m/z label. "
    "The label may be horizontal or vertical. Keep the box reasonably tight around the number; "
    "the app automatically tests horizontal and ±90° orientations."
)
st.caption(
    f"Selection view: {display_w} × {display_h} px · "
    f"Original image: {w} × {h} px. "
    "The app converts the selected box back to the original image automatically for OCR."
)

if "boxes" not in st.session_state: st.session_state.boxes=[]
if "pending_corner" not in st.session_state: st.session_state.pending_corner=None
if "last_click" not in st.session_state: st.session_state.last_click=None

if "box_selector_epoch" not in st.session_state:
    st.session_state.box_selector_epoch = 0

if "peak_table_epoch" not in st.session_state:
    st.session_state.peak_table_epoch = 0

if "peak_manual_edits" not in st.session_state:
    st.session_state.peak_manual_edits = {}

c1,c2,c3=st.columns([1,1,4])

if c1.button("Undo last box"):
    if st.session_state.boxes:
        removed = st.session_state.boxes.pop()
        st.session_state.peak_manual_edits.pop(box_id(removed), None)

    st.session_state.pending_corner = None
    st.session_state.last_click = None
    st.session_state.box_selector_epoch += 1
    st.session_state.peak_table_epoch += 1
    st.rerun()

if c2.button("Clear all"):
    st.session_state.boxes = []
    st.session_state.peak_manual_edits = {}
    st.session_state.pending_corner = None
    st.session_state.last_click = None

    st.session_state.box_selector_epoch += 1
    st.session_state.peak_table_epoch += 1
    st.rerun()

c3.write(f"Selected boxes: **{len(st.session_state.boxes)}**")

shown=draw_boxes(
    interaction_img,
    st.session_state.boxes,
    st.session_state.pending_corner,
)

# Important: do NOT use use_column_width=True here.
# That would resize the rendered image again and break the coordinate mapping.
click=streamlit_image_coordinates(
    shown,
    key=f"box_selector_{st.session_state.box_selector_epoch}",
    use_column_width=False,
)

if click:
    p=(int(click["x"]),int(click["y"]))
    if p != st.session_state.last_click:
        st.session_state.last_click=p
        if st.session_state.pending_corner is None:
            st.session_state.pending_corner=p
        else:
            xa,ya=st.session_state.pending_corner
            xb,yb=p
            x1,x2=sorted([xa,xb]); y1,y2=sorted([ya,yb])
            if x2-x1>5 and y2-y1>5:
                st.session_state.boxes.append((x1,y1,x2,y2))
            st.session_state.pending_corner=None
        st.rerun()

if st.session_state.pending_corner:
    st.info("First corner selected. Click the opposite corner to complete the box.")

if not st.session_state.boxes:
    st.stop()

st.subheader("2. OCR + peak intensity")

rows=[]
for i,b_display in enumerate(st.session_state.boxes,1):
    this_box_id = box_id(b_display)

    # Convert the visually selected rectangle to source-image coordinates.
    b = display_box_to_original(
        b_display,
        display_to_orig_x,
        display_to_orig_y,
    )

    mz,txt,conf,orientation=ocr_box(gray,b)
    if mz is None:
        rows.append({
            "use":False,
            "box":i,
            "_box_id":this_box_id,
            "mz":np.nan,
            "intensity":np.nan,
            "ocr_text":"",
            "ocr_conf":0,
            "ocr_orientation":"",
            "status":"Review",
            "peak_x":np.nan,
            "height_px":0,
            "box_original":str(b),
        })
        continue
    expected=mz_to_x(mz,cal)
    px,height=local_peak(gray,int(baseline),expected,radius=3,threshold=245)
    status="Reliable" if height>=5 and abs(px-expected)<=3 and conf>=40 else "Review"
    rows.append({
        "use":status=="Reliable",
        "box":i,
        "_box_id":this_box_id,
        "mz":mz,
        "intensity":np.nan,
        "ocr_text":txt,
        "ocr_conf":conf,
        "ocr_orientation":orientation,
        "status":status,
        "peak_x":px,
        "height_px":height,
        "expected_x":expected,
        "box_original":str(b),
    })

df=pd.DataFrame(rows)

# IMPORTANT: normalize every labeled peak against the true base peak detected
# in the complete spectrum, not against the tallest OCR-selected peak.
if len(df) and base_peak_height > 0:
    df["intensity"] = (
        pd.to_numeric(df["height_px"], errors="coerce")
        / float(base_peak_height)
        * 100.0
    )
    df["intensity"] = df["intensity"].clip(lower=0.0, upper=100.0)

for idx, row in df.iterrows():
    saved = st.session_state.peak_manual_edits.get(row["_box_id"])
    if saved is not None:
        df.at[idx, "use"] = saved.get("use", df.at[idx, "use"])
        df.at[idx, "mz"] = saved.get("mz", df.at[idx, "mz"])
        df.at[idx, "intensity"] = saved.get("intensity", df.at[idx, "intensity"])

st.caption(
    "Review the OCR result before export. You can edit **m/z**, "
    "**relative intensity**, and **Use**. Diagnostic columns remain read-only."
)

# Put the most important scientific fields first.
editor_columns = [
    "_box_id",
    "use",
    "mz",
    "intensity",
    "status",
    "box",
    "ocr_text",
    "ocr_conf",
    "ocr_orientation",
    "height_px",
    "peak_x",
    "expected_x",
    "box_original",
]

edited=st.data_editor(
    df[editor_columns],
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    column_config={
        "_box_id": None,
        "use":st.column_config.CheckboxColumn(
            "Use",
            help="Include this peak in the exported MGF.",
        ),
        "mz":st.column_config.NumberColumn(
            "m/z",
            format="%.6f",
            help="Editable. Correct the OCR result here if necessary.",
        ),
        "intensity":st.column_config.NumberColumn(
            "Relative intensity (%)",
            format="%.3f",
            min_value=0.0,
            help="Editable relative intensity used in the exported MGF.",
        ),
        "status":st.column_config.TextColumn(
            "Status",
            disabled=True,
        ),
        "box":st.column_config.NumberColumn(
            "Box",
            disabled=True,
        ),
        "ocr_text":st.column_config.TextColumn(
            "OCR text",
            disabled=True,
        ),
        "ocr_conf":st.column_config.NumberColumn(
            "OCR confidence",
            format="%.1f",
            disabled=True,
        ),
        "ocr_orientation":st.column_config.TextColumn(
            "OCR orientation",
            help="Orientation that produced the accepted OCR reading.",
            disabled=True,
        ),
        "height_px":st.column_config.NumberColumn(
            "Measured height (px)",
            format="%.1f",
            disabled=True,
        ),
        "peak_x":st.column_config.NumberColumn(
            "Detected peak x",
            format="%.2f",
            disabled=True,
        ),
        "expected_x":st.column_config.NumberColumn(
            "Expected x",
            format="%.2f",
            disabled=True,
        ),
        "box_original":st.column_config.TextColumn(
            "OCR crop (original px)",
            disabled=True,
        ),
    },
    disabled=[
        "status",
        "box",
        "ocr_text",
        "ocr_conf",
        "ocr_orientation",
        "height_px",
        "peak_x",
        "expected_x",
        "box_original",
    ],
    key=(
        f"peak_table_{st.session_state.peak_table_epoch}_"
        f"{len(st.session_state.boxes)}"
    ),
)

# Force edited scientific columns back to numeric types.
edited["mz"] = pd.to_numeric(edited["mz"], errors="coerce")
edited["intensity"] = pd.to_numeric(edited["intensity"], errors="coerce")

for _, row in edited.iterrows():
    st.session_state.peak_manual_edits[row["_box_id"]] = {
        "use": bool(row["use"]) if pd.notna(row["use"]) else False,
        "mz": float(row["mz"]) if pd.notna(row["mz"]) else np.nan,
        "intensity": float(row["intensity"]) if pd.notna(row["intensity"]) else np.nan,
    }

selected=edited[edited["use"].fillna(False)].copy()

st.caption(
    "Intensity formula: measured stick height / full-spectrum base-peak height × 100."
)

st.plotly_chart(
    spectrum_fig(selected,"Selected / reconstructed spectrum"),
    use_container_width=True,
)

st.subheader("3. Generate MGF")

st.caption(
    "In standard MGF, the free-text spectrum/compound name is stored in "
    "`TITLE=`. `NAME=` is not a standard MGF field. For MS/MS, the precursor "
    "ion is stored as `PEPMASS=<precursor m/z>`."
)

a,b=st.columns(2)

spectrum_id=a.text_input(
    "Spectrum ID / file name",
    value=Path(uploaded.name).stem,
    help="Used as TITLE only when Compound name is left empty.",
)

compound_name=b.text_input(
    "Compound name (optional)",
    value="",
    placeholder="e.g. Caffeine",
    help="If provided, this value is written as TITLE= in the MGF.",
)

c,d=st.columns(2)

pep=c.text_input(
    "Precursor m/z — PEPMASS (optional)",
    value="",
    placeholder="e.g. 195.0877",
    help=(
        "For an MS/MS spectrum, enter the observed precursor ion m/z. "
        "MGF writes this as PEPMASS=. This is not necessarily the neutral molecular mass."
    ),
)

charge=d.text_input(
    "CHARGE (optional)",
    value="",
    placeholder="e.g. 1+",
)

try:
    pepmass=float(pep) if pep.strip() else None
except:
    pepmass=None
    st.warning("Precursor m/z (PEPMASS) is not numeric and was omitted.")

mgf=mgf_text(
    edited,
    spectrum_id=spectrum_id,
    compound_name=compound_name,
    pepmass=pepmass,
    charge=charge.strip() or None,
)

st.code(mgf,language="text")

st.download_button(
    "Download .mgf",
    mgf.encode("utf-8"),
    file_name=f"{Path(uploaded.name).stem}.mgf",
    mime="text/plain",
)

st.subheader("4. Validation")
if len(selected):
    val=selected.copy()
    val["reconstructed_x"]=[mz_to_x(x,cal) for x in val["mz"]]
    val["delta_x_px"]=val["reconstructed_x"]-val["peak_x"]
    med=float(np.nanmedian(np.abs(val["delta_x_px"])))
    mx=float(np.nanmax(np.abs(val["delta_x_px"])))
    score=float(np.exp(-med/3.0))  # spatial agreement indicator, not library spectral cosine
    c1,c2,c3=st.columns(3)
    c1.metric("Spatial agreement",f"{score:.4f}")
    c2.metric("Median |Δx|",f"{med:.2f} px")
    c3.metric("Maximum |Δx|",f"{mx:.2f} px")
    st.dataframe(val[["mz","intensity","peak_x","reconstructed_x","delta_x_px"]],
                 use_container_width=True,hide_index=True)

st.caption(
    "m/z values come from OCR inside user-selected boxes or manual correction. "
    "The x-axis calibration is used only to locate the corresponding stick and validate spatial consistency."
)


# =============================================================================
# 5. MassQL query export
# =============================================================================

with st.expander("5. Convert selected peaks to a MassQL query", expanded=False):
    st.caption(
        "Generate a single MassQL query from all peaks currently marked as `Use` "
        "in the OCR + peak intensity table."
    )

    mq1, mq2, mq3 = st.columns(3)

    massql_polarity = mq1.selectbox(
        "Polarity",
        options=["POSITIVE", "NEGATIVE"],
        index=0,
        key="massql_polarity",
    )

    precursor_ppm = mq2.number_input(
        "MS2 precursor tolerance (ppm)",
        min_value=0.1,
        value=20.0,
        step=1.0,
        key="massql_precursor_ppm",
    )

    product_ppm = mq3.number_input(
        "MS2 product tolerance (ppm)",
        min_value=0.1,
        value=20.0,
        step=1.0,
        key="massql_product_ppm",
    )

    intensity_percent = st.number_input(
        "Minimum fragment intensity — INTENSITYPERCENT (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="massql_intensity_percent",
    )

    massql_selected = edited[
        edited["use"].fillna(False)
    ].copy()

    massql_selected["mz"] = pd.to_numeric(
        massql_selected["mz"],
        errors="coerce",
    )

    massql_selected = (
        massql_selected
        .dropna(subset=["mz"])
        .sort_values("mz")
    )

    fragment_masses = [
        f"{float(mz):.6f}"
        for mz in massql_selected["mz"].tolist()
    ]

    if fragment_masses:
        fragments_or = " OR ".join(fragment_masses)

        conditions = [
            f"POLARITY={massql_polarity}"
        ]

        if pepmass is not None:
            conditions.append(
                f"MS2PREC={float(pepmass):.6f}:"
                f"TOLERANCEPPM={float(precursor_ppm):g}"
            )
        else:
            st.info(
                "No PEPMASS was entered in Section 3. "
                "The MassQL query will therefore omit MS2PREC."
            )

        conditions.append(
            f"MS2PROD=({fragments_or}):"
            f"TOLERANCEPPM={float(product_ppm):g}:"
            f"INTENSITYPERCENT={float(intensity_percent):g}"
        )

        massql_query = (
            "QUERY scaninfo(MS2DATA) WHERE "
            + " AND ".join(conditions)
        )

        massql_title = (
            compound_name.strip()
            if compound_name.strip()
            else spectrum_id.strip()
        )

        massql_text = (
            f"# {massql_title}\n"
            f"{massql_query}"
        )

        st.code(
            massql_text,
            language="text",
        )

        st.download_button(
            "Download MassQL query",
            data=massql_text.encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}_massql.txt",
            mime="text/plain",
            key="download_massql_query",
        )

    else:
        st.warning(
            "No valid peaks are currently marked as `Use` in the table."
        )

