"""
app_landing.py - Landing page + classifier in ONE Streamlit site.
Run:  python -m streamlit run app_landing.py
"""
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# ----------------------------- SETTINGS (same as app_live.py) ---------------
MODEL_PATH = "best_model.keras"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SCALING = "div255"            # "div255", "mobilenet" or "none"
MIN_CONFIDENCE = 0.50
SMOOTHING_FRAMES = 8

# bin colour + what to do with it
BINS = {
    "cardboard": ("#B08A5A", "Flatten the box and keep it dry."),
    "glass":     ("#3E9B6B", "Rinse it out. Keep lids and caps separate."),
    "metal":     ("#7C8791", "Rinse cans and drop them in loose."),
    "paper":     ("#3F6FD1", "Clean and dry only. Greasy paper goes to general waste."),
    "plastic":   ("#E3B52E", "Rinse and check the number on the bottom."),
    "trash":     ("#2B2F33", "General waste. This can't be recycled."),
}
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Sortly - Waste Classifier", page_icon="♻️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=DM+Sans:wght@400;500&display=swap');
:root{--bg:#EEF3F1;--ink:#10231F;--muted:#4F615C;--line:#CBD8D3;--accent:#1F7A5C;}
.stApp{background:var(--bg);color:var(--ink);font-family:'DM Sans',sans-serif;}
header[data-testid="stHeader"],#MainMenu,footer{display:none!important;}
.block-container{max-width:1080px;padding-top:1.5rem;padding-bottom:3rem;}
h1,h2,h3,.brand{font-family:'Bricolage Grotesque',sans-serif;color:var(--ink);letter-spacing:-0.02em;}
.brand{font-size:1.6rem;font-weight:800;padding-top:.3rem;}
.hero{padding:3.5rem 0 1rem;}
.hero h1{font-size:clamp(2.6rem,6.5vw,5rem);line-height:1;font-weight:800;margin:0 0 1.2rem;max-width:14ch;}
.hero p{font-size:1.2rem;color:var(--muted);max-width:46ch;line-height:1.55;margin:0 0 1.5rem;}
.bins{display:flex;gap:14px;align-items:flex-end;margin:2.5rem 0 1rem;}
.bin{flex:1;display:flex;flex-direction:column;align-items:stretch;gap:8px;}
.bin .body{border-radius:6px 6px 14px 14px;position:relative;transform-origin:bottom;animation:rise .7s cubic-bezier(.2,.8,.2,1) both;}
.bin .body:before{content:"";position:absolute;left:-5px;right:-5px;top:-9px;height:9px;border-radius:5px;background:inherit;filter:brightness(.85);}
.bin span{font-weight:500;font-size:.95rem;text-align:center;}
@keyframes rise{from{transform:scaleY(.15);opacity:0;}to{transform:scaleY(1);opacity:1;}}
@media (prefers-reduced-motion:reduce){.bin .body{animation:none;}}
.step{border-top:2px solid var(--ink);padding-top:.8rem;}
.step h3{font-size:1.3rem;margin:.2rem 0;}
.step p{color:var(--muted);line-height:1.5;margin:0;}
.result{background:#fff;border:1px solid var(--line);border-left:10px solid var(--c);border-radius:10px;padding:1rem 1.3rem;margin:1rem 0;}
.result b{font-family:'Bricolage Grotesque',sans-serif;font-size:1.7rem;}
.result div{color:var(--muted);margin-top:.2rem;}
.stButton>button,button[data-testid^="stBaseButton"]{border-radius:999px;font-weight:500;padding:.55rem 1.4rem;border:1px solid var(--ink);color:var(--ink);background:transparent;}
button[data-testid="stBaseButton-primary"],.stButton>button[kind="primary"]{background:var(--ink);color:#fff;}
button:focus-visible{outline:3px solid var(--accent)!important;outline-offset:2px;}
.small{color:var(--muted);font-size:.9rem;margin-top:3rem;}
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("page", "home")
ss.setdefault("mode", "Live camera")


def go(page, mode=None):
    ss.page = page
    if mode:
        ss.mode = mode


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def prep(rgb, w, h):
    rgb = cv2.resize(rgb, (w, h)).astype("float32")
    if SCALING == "div255":
        rgb /= 255.0
    elif SCALING == "mobilenet":
        rgb = rgb / 127.5 - 1.0
    return np.expand_dims(rgb, 0)


class WasteProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.h, self.w = model.input_shape[1:3]
        self.hist, self.n = deque(maxlen=SMOOTHING_FRAMES), 0
        self.text, self.color = "...", (255, 255, 255)

    def recv(self, frame):
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        fh, fw = img.shape[:2]
        s = int(min(fh, fw) * 0.6)
        x, y = (fw - s) // 2, (fh - s) // 2
        crop = img[y:y + s, x:x + s]
        cv2.rectangle(img, (x, y), (x + s, y + s), (255, 255, 0), 2)
        if self.n % 3 == 0:
            p = self.model.predict(prep(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), self.w, self.h), verbose=0)[0]
            self.hist.append(p)
            avg = np.mean(self.hist, axis=0)
            i = int(np.argmax(avg))
            if avg[i] >= MIN_CONFIDENCE:
                self.text, self.color = f"{CLASS_NAMES[i]}  {avg[i] * 100:.0f}%", (0, 200, 0)
            else:
                self.text, self.color = "Not sure", (0, 165, 255)
        self.n += 1
        cv2.rectangle(img, (0, 0), (fw, 45), (0, 0, 0), -1)
        cv2.putText(img, self.text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------------- NAV ------------------------------------------
c1, c2, c3 = st.columns([6, 1.2, 1.6])
c1.markdown('<div class="brand">♻️ Sortly</div>', unsafe_allow_html=True)
c2.button("Home", on_click=go, args=("home",), use_container_width=True)
c3.button("Classifier", on_click=go, args=("app",), type="primary", use_container_width=True)

# ----------------------------- HOME -----------------------------------------
if ss.page == "home":
    st.markdown("""
    <div class="hero">
      <h1>Know which bin it belongs in.</h1>
      <p>Show an item to your camera or upload a photo. Sortly tells you if it's cardboard, glass,
      metal, paper, plastic or trash, and how to dispose of it.</p>
    </div>""", unsafe_allow_html=True)

    b1, b2, _ = st.columns([1.4, 1.4, 4])
    b1.button("Start live camera", on_click=go, args=("app", "Live camera"), type="primary", use_container_width=True)
    b2.button("Upload a photo", on_click=go, args=("app", "Upload image"), use_container_width=True)

    heights = [150, 190, 130, 170, 210, 120]
    bins_html = "".join(
        f'<div class="bin"><div class="body" style="height:{h}px;background:{BINS[c][0]};animation-delay:{i * 0.08}s"></div>'
        f'<span>{c.capitalize()}</span></div>'
        for i, (c, h) in enumerate(zip(CLASS_NAMES, heights))
    )
    st.markdown(f'<div class="bins">{bins_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3, gap="large")
    steps = [
        ("1. Show it", "Hold the item inside the box on your camera, or upload a photo from your device."),
        ("2. We identify it", "A neural network trained on thousands of waste photos picks one of six materials."),
        ("3. Sort it", "Get the bin colour and a one-line tip, like rinsing or flattening, before you toss it."),
    ]
    for col, (t, d) in zip((s1, s2, s3), steps):
        col.markdown(f'<div class="step"><h3>{t}</h3><p>{d}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="small">Built with TensorFlow, Keras and Streamlit. Trained on the TrashNet dataset.</div>',
                unsafe_allow_html=True)

# ----------------------------- CLASSIFIER -----------------------------------
else:
    model = get_model()
    h, w = model.input_shape[1:3]
    st.markdown("## Classify your waste")
    st.radio("Mode", ["Live camera", "Upload image"], key="mode", horizontal=True, label_visibility="collapsed")

    if ss.mode == "Live camera":
        st.write("Click **START**, allow camera access, and hold the item inside the yellow box.")
        webrtc_streamer(
            key="sortly-live",
            video_processor_factory=lambda: WasteProcessor(model),
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
    else:
        file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if file:
            img = Image.open(file).convert("RGB")
            left, right = st.columns(2, gap="large")
            left.image(img, use_container_width=True)
            p = model.predict(prep(np.array(img), w, h), verbose=0)[0]
            i = int(np.argmax(p))
            name = CLASS_NAMES[i]
            color, tip = BINS.get(name, ("#2B2F33", ""))
            right.markdown(
                f'<div class="result" style="--c:{color}"><b>{name.capitalize()} · {p[i] * 100:.0f}%</b>'
                f'<div>{tip}</div></div>', unsafe_allow_html=True)
            right.bar_chart({c: float(v) for c, v in zip(CLASS_NAMES, p)})