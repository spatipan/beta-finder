"""
app.py - BetaFinder CNX Streamlit Web UI
Modern minimalistic climbing gym aesthetic with holds, stripes, volume colors

Theme: Dark wall (climbing gym wall aesthetic)
- Background: deep charcoal (#1a1a1a, #0f0f0f) like gym wall
- Primary holds: warm accents (#ff6b35, #ffa500, #ffd700)
- Secondary holds: cool accents (#00d4ff, #00ff88) volume markers
- Stripes: subtle (#2a2a2a) like volume tape
- Text: clean (#ffffff, #e0e0e0, #a0a0a0)
"""

import streamlit as st
import json
import os
import uuid
import tempfile
from pathlib import Path
from PIL import Image

# Import from src modules
from src.config import load_config, get_gym_names, get_nested
from src.search import search
from src.feedback import save_feedback, get_feedback_stats
from src.logger import setup_logger

log = setup_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration & Theme
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BetaFinder CNX",
    page_icon="🧗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for climbing gym aesthetic
st.markdown("""
<style>
:root {
    --wall-bg: #1a1a1a;
    --wall-dark: #0f0f0f;
    --wall-light: #2a2a2a;
    --hold-warm: #ff6b35;
    --hold-gold: #ffd700;
    --hold-orange: #ffa500;
    --volume-cyan: #00d4ff;
    --volume-green: #00ff88;
    --stripe: #2a2a2a;
    --text-primary: #ffffff;
    --text-secondary: #e0e0e0;
    --text-muted: #a0a0a0;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a1a 0%, #252525 100%);
}

.stApp {
    color: var(--text-primary);
}

h1, h2, h3 {
    color: var(--text-primary) !important;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.stButton > button {
    background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%);
    color: white;
    border: none;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
}

input, textarea, select {
    background-color: var(--wall-light) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--stripe) !important;
    border-radius: 6px !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, var(--hold-warm) 0%, var(--hold-gold) 100%) !important;
}

a {
    color: var(--hold-warm) !important;
    text-decoration: none;
    transition: color 0.3s ease;
}

a:hover {
    color: var(--hold-gold) !important;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "last_query_path" not in st.session_state:
    st.session_state.last_query_path = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_gym_index():
    """Load gym index from file"""
    index_file = Path("data/gym_index.json")
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def get_stats():
    """Get dataset statistics"""
    metadata = load_gym_index()
    if not metadata:
        return {"total": 0, "alpine": 0, "mainwall": 0, "progression": 0}

    return {
        "total": len(metadata),
        "alpine": sum(1 for m in metadata if m.get("gym") == "alpine"),
        "mainwall": sum(1 for m in metadata if m.get("gym") == "mainwall"),
        "progression": sum(1 for m in metadata if m.get("gym") == "progression"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    model_choice = st.selectbox(
        "CLIP Model",
        ["ViT-B-32 (Fast)", "ViT-L-14 (Accurate)"],
        help="ViT-B-32: ~10s per image on CPU\nViT-L-14: ~30s per image on CPU"
    )
    model_map = {"ViT-B-32 (Fast)": "ViT-B-32", "ViT-L-14 (Accurate)": "ViT-L-14"}
    selected_model = model_map[model_choice]

    top_k = st.slider(
        "Top Results",
        min_value=3,
        max_value=20,
        value=get_nested("search.default_top_k"),
        step=1,
        help="Number of results to display"
    )

    st.markdown("---")
    st.markdown("### 📊 Index Statistics")

    stats = get_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Images", stats["total"], help="Images in FAISS index")
    with col2:
        st.metric("Gyms Indexed", 3 if stats["total"] > 0 else 0)

    st.markdown("**By Gym:**")
    gym_cols = st.columns(3)
    gyms = [
        ("alpine", "🟢 Alpine", stats["alpine"]),
        ("mainwall", "🔵 MainWall", stats["mainwall"]),
        ("progression", "🟠 Progression", stats["progression"]),
    ]
    for (gym_key, gym_label, gym_count), col in zip(gyms, gym_cols):
        with col:
            st.metric(gym_label, gym_count)

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    **For best results:**
    - Take photo straight-on at wall
    - Ensure good lighting
    - Include multiple holds/features
    - Avoid people blocking view
    """)

    st.markdown("---")
    st.markdown("### 🎯 Feedback Progress")
    fb_stats = get_feedback_stats()
    st.caption(f"Helping improve AI accuracy — goal: {fb_stats['goal']} labeled pairs")
    st.progress(
        fb_stats["pct_complete"] / 100,
        text=f"{fb_stats['total']} / {fb_stats['goal']} pairs ({fb_stats['pct_complete']}%)"
    )
    fb_col1, fb_col2 = st.columns(2)
    with fb_col1:
        st.metric("👍 Relevant", fb_stats["positive"])
    with fb_col2:
        st.metric("👎 Not this", fb_stats["negative"])

    st.markdown("---")
    st.markdown("### 📝 About")
    st.caption("""
    **BetaFinder CNX**

    Find climbing beta (routes) using AI image matching.

    Built for Chiang Mai climbing community.

    [GitHub](https://github.com/spatipan/beta-finder)
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────────────

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🧗")
with col2:
    st.markdown("# BetaFinder CNX")
    st.caption("Find your beta on the wall — สำหรับยิมปีนผาเชียงใหม่")

st.markdown("---")

# Gym selector
st.markdown("### 🏢 Filter by Gym")

gym_filters = {
    "alpine": {"label": "🟢 Alpine Outpost", "color": "#4CAF50"},
    "mainwall": {"label": "🔵 Main Wall", "color": "#2196F3"},
    "progression": {"label": "🟠 Progression Vertical", "color": "#FF9800"},
    "all": {"label": "🌐 All Gyms", "color": "#FFD700"},
}

selected_gyms = []
col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]

for (gym_key, gym_info), col in zip(gym_filters.items(), cols):
    with col:
        if st.checkbox(gym_info["label"], value=(gym_key == "all"), key=f"gym_{gym_key}"):
            selected_gyms.append(gym_key)

# Remove "all" if other gyms selected
if selected_gyms and "all" in selected_gyms and len(selected_gyms) > 1:
    selected_gyms.remove("all")

gym_filter = None if "all" in selected_gyms or not selected_gyms else selected_gyms[0]

st.markdown("---")

# Upload section
st.markdown("### 📸 Upload Wall Photo")
uploaded_file = st.file_uploader(
    "Choose a wall photo",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of the climbing wall"
)

if uploaded_file:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, width="stretch", caption="📷 Your wall photo")

    with col2:
        st.markdown("**Photo Info:**")
        width, height = image.size
        st.caption(f"Size: {width}×{height}")
        st.caption(f"Format: {image.format}")
        st.caption(f"Mode: {image.mode}")

    st.markdown("---")

    # Search button
    if st.button("🔍 Find Beta", key="search_btn", use_container_width=True):

        # Save uploaded file temporarily (keep path for feedback attribution)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("🔄 Embedding your photo..."):
                results = search(
                    Path(tmp_path),
                    top_k=top_k,
                    gym_filter=gym_filter,
                    model_name=selected_model,
                    pretrained="openai"
                )

            # Store in session state so feedback buttons work across reruns
            st.session_state.last_results = results
            st.session_state.last_query_path = tmp_path
            st.session_state.last_model = selected_model

        except Exception as e:
            st.error(f"Search failed: {e}")
            st.session_state.last_results = []

        finally:
            # Don't delete tmp_path here — needed for feedback recording
            pass

    # Show results (from session state so they persist after 👍/👎 clicks)
    results = st.session_state.last_results
    if results:
        st.success(f"✅ Found {len(results)} matching betas!")
        st.markdown("---")

        # Results header with stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Match", f"{results[0]['score']:.3f}")
        with col2:
            avg_score = sum(r['score'] for r in results) / len(results)
            st.metric("📊 Avg Score", f"{avg_score:.3f}")
        with col3:
            st.metric("📸 Results", len(results))

        st.markdown("---")
        st.markdown("### 🎯 Results")

        gym_emoji = {"alpine": "🟢", "mainwall": "🔵", "progression": "🟠", "?": "⚪"}

        for i, result in enumerate(results, 1):
            with st.container():
                col1, col2 = st.columns([1, 2])

                with col1:
                    img_path = Path(result['filename'])
                    if img_path.exists():
                        st.image(str(img_path), width="stretch")
                    else:
                        st.caption("🖼️ Image not found")

                with col2:
                    gym_label = result['gym'].upper()
                    emoji = gym_emoji.get(result['gym'].lower(), "⚪")
                    st.markdown(f"**#{i} {emoji} {gym_label}**")

                    st.progress(result['score'], text=f"Similarity: {result['score']:.1%}")

                    if result['caption']:
                        st.markdown(f"*{result['caption']}*")
                    else:
                        st.caption("(No caption)")

                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.caption(f"📅 {result['date']}")
                    with meta_col2:
                        st.caption(f"👤 {result['filename'].split('/')[-2]}")

                    if result['url']:
                        st.markdown(f"[🔗 View on Instagram]({result['url']})")

                    # Feedback buttons
                    st.caption("Was this the same route?")
                    fb_col1, fb_col2 = st.columns(2)
                    with fb_col1:
                        if st.button("👍 Yes!", key=f"pos_{i}_{result['filename']}"):
                            save_feedback(
                                result, "positive",
                                st.session_state.session_id,
                                st.session_state.last_query_path or "",
                                model_used=st.session_state.last_model or selected_model,
                            )
                            st.toast("Thanks! Marked as relevant 🎯")
                    with fb_col2:
                        if st.button("👎 No", key=f"neg_{i}_{result['filename']}"):
                            save_feedback(
                                result, "negative",
                                st.session_state.session_id,
                                st.session_state.last_query_path or "",
                                model_used=st.session_state.last_model or selected_model,
                            )
                            st.toast("Thanks! Noted 📝")

            st.divider()

    if not results and not st.session_state.last_results:
        st.warning("❌ No matching betas found. Try another photo!")

else:
    # Placeholder when no image uploaded
    st.info("""
    👆 **Upload a wall photo to get started!**

    BetaFinder will search through Instagram posts from:
    - Alpine Outpost
    - Main Wall CNX
    - Progression Vertical
    - Community climber accounts

    And return the most visually similar climbing routes.
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #a0a0a0; margin-top: 3rem;'>
    <p>Built with ❤️ for Chiang Mai climbing community</p>
    <p>Data from official gym accounts + community contributors</p>
</div>
""", unsafe_allow_html=True)
