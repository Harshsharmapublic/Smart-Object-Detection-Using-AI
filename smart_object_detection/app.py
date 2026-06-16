"""
app.py — Production-grade AI Smart Surveillance Dashboard
========================================================
ARCHITECTURE NOTES (important for Streamlit correctness):
  • All stateful objects (tracker, line_counter, heatmap, logger) are
    stored in st.session_state so they survive widget-triggered reruns.
  • input_source / video_file / overlay_mode are collected BEFORE the
    processing loop so they are always defined in the outer scope.
  • The video-capture loop renders frames synchronously; Streamlit's
    st.empty() placeholder is updated in-place each iteration.
  • FPS uses a rolling 30-frame average (utils.calculate_fps).
"""

import os
import tempfile
import time
from datetime import datetime

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from analytics import SurveillanceAnalytics
from detector import YOLODetector
from heatmap import MovementHeatmap
from line_counter import LineCrossingCounter
from logger import DetectionLogger
from tracker import ObjectTracker
import utils

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Smart Surveillance Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  (premium dark-mode design)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Base ────────────────────────────────────────────────── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Header ──────────────────────────────────────────────── */
    .app-title {
        font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 2.6rem;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #a855f7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        line-height: 1.15; margin-bottom: .15rem;
    }
    .app-sub {
        color: #64748b; font-size: 1rem; font-weight: 300; margin-bottom: 1.6rem;
        letter-spacing: .02rem;
    }

    /* ── KPI cards ───────────────────────────────────────────── */
    .kpi-card {
        background: linear-gradient(145deg, rgba(15,23,42,.85), rgba(17,24,39,.95));
        border: 1px solid rgba(51,65,85,.6); border-radius: 18px;
        padding: 1.1rem 1rem; text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,.35);
        backdrop-filter: blur(14px);
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 40px rgba(0,242,254,.18);
        border-color: rgba(0,242,254,.35);
    }
    .kpi-val   { font-family:'Outfit',sans-serif; font-size:2.2rem; font-weight:700;
                 line-height:1; margin-bottom:.25rem; }
    .kpi-label { font-size:.72rem; font-weight:500; color:#475569;
                 text-transform:uppercase; letter-spacing:1.1px; }
    .c-cyan   { color:#00f2fe; }
    .c-orange { color:#f97316; }
    .c-green  { color:#22c55e; }
    .c-red    { color:#ef4444; }
    .c-purple { color:#a855f7; }
    .c-amber  { color:#fbbf24; }

    /* ── Alert banner ────────────────────────────────────────── */
    .alert-box {
        padding:.85rem 1.3rem; border-radius:12px; font-weight:600;
        border:1px solid rgba(239,68,68,.45);
        background:rgba(239,68,68,.12); color:#fca5a5;
        box-shadow:0 4px 18px rgba(239,68,68,.12);
        animation: alert-pulse 2s ease-in-out infinite;
        margin-bottom:.8rem;
    }
    @keyframes alert-pulse {
        0%,100% { border-color:rgba(239,68,68,.45); }
        50%      { border-color:rgba(239,68,68,.85); box-shadow:0 4px 22px rgba(239,68,68,.28); }
    }

    /* ── Timeline ────────────────────────────────────────────── */
    .tl-item {
        background:rgba(30,41,59,.55); border-left:3px solid #3b82f6;
        border-radius:4px 10px 10px 4px; padding:.55rem 1rem;
        margin-bottom:.5rem;
    }
    .tl-alert  { border-left-color:#ef4444; background:rgba(239,68,68,.06); }
    .tl-cross  { border-left-color:#f97316; background:rgba(249,115,22,.06); }
    .tl-time   { font-family:'JetBrains Mono',monospace; font-size:.75rem;
                 color:#475569; margin-right:8px; }
    .tl-text   { font-size:.85rem; font-weight:500; color:#cbd5e1; }

    /* ── Section headings ────────────────────────────────────── */
    .sec-head  { font-family:'Outfit',sans-serif; font-weight:700; font-size:1.25rem;
                 color:#e2e8f0; margin-bottom:.6rem; }

    /* ── Feed offline placeholder ────────────────────────────── */
    .offline-box {
        background:rgba(15,23,42,.7); border:1px dashed rgba(51,65,85,.8);
        border-radius:14px; padding:3rem 2rem; text-align:center; color:#334155;
    }
    .offline-icon { font-size:3.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# DIRECTORY SETUP
# ═══════════════════════════════════════════════════════════════════════════
DIRS = dict(logs="logs", screenshots="screenshots", models="models", reports="reports")
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE — initialise ONCE per browser session
# ═══════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "running":        False,
    "capture_flag":   False,
    "alert_count":    0,
    "entry_count":    0,
    "exit_count":     0,
    "last_breach_id": None,
    # Stateful objects — survive Streamlit reruns
    "tracker":        None,
    "line_counter":   None,
    "heatmap":        None,
    "det_logger":     None,
    "analytics":      None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES — model loading (heavy, only once per model name)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def _load_detector(model_name: str) -> YOLODetector:
    return YOLODetector(model_name=model_name, models_dir=DIRS["models"])

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — all controls
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<p class='sec-head'>⚙️ Control Panel</p>", unsafe_allow_html=True)

    # ── Model ──────────────────────────────────────────────────────────
    model_choice = st.selectbox(
        "YOLOv8 Model Size",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="Nano = fastest · Small = balanced · Medium = most accurate",
    )
    detector = _load_detector(model_choice)

    # ── Class filter ───────────────────────────────────────────────────
    all_classes = list(detector.get_class_names().values())
    default_sel = [c for c in ["person", "bottle", "car", "laptop", "cell phone"] if c in all_classes]
    selected_classes = st.multiselect(
        "Detect Classes",
        options=all_classes,
        default=default_sel or all_classes[:6],
    )
    class_id_map = {v: k for k, v in detector.get_class_names().items()}
    selected_ids = [class_id_map[c] for c in selected_classes if c in class_id_map]

    conf_thresh  = st.slider("Confidence Threshold", 0.10, 0.95, 0.30, 0.05)
    use_tracking = st.toggle("Multi-Object Tracking", value=True,
                             help="Assigns persistent IDs across frames")

    # ── Restricted zone ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p class='sec-head' style='font-size:1rem;'>🛡️ Restricted Zone</p>",
                unsafe_allow_html=True)
    zone_on = st.toggle("Enable Zone Alert", value=True)
    c1, c2 = st.columns(2)
    with c1:
        zx0 = st.slider("X min", 0.0, 1.0, 0.25, 0.05, key="zx0")
        zy0 = st.slider("Y min", 0.0, 1.0, 0.25, 0.05, key="zy0")
    with c2:
        zx1 = st.slider("X max", 0.0, 1.0, 0.75, 0.05, key="zx1")
        zy1 = st.slider("Y max", 0.0, 1.0, 0.75, 0.05, key="zy1")
    # Guard: ensure min < max
    zx1 = max(zx1, zx0 + 0.05); zy1 = max(zy1, zy0 + 0.05)
    zone = {"xmin": zx0, "ymin": zy0, "xmax": zx1, "ymax": zy1} if zone_on else None

    # ── Counting line ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p class='sec-head' style='font-size:1rem;'>📏 Counting Line</p>",
                unsafe_allow_html=True)
    line_on  = st.toggle("Enable Line Counter", value=True)
    line_pct = st.slider("Line Position (% height)", 0.10, 0.90, 0.50, 0.05)

    # ── Heatmap ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p class='sec-head' style='font-size:1rem;'>🔥 Movement Heatmap</p>",
                unsafe_allow_html=True)
    heatmap_on    = st.toggle("Enable Heatmap Overlay", value=False)
    heatmap_decay = st.slider(
        "Trail Decay", 0.90, 1.00, 1.00, 0.005,
        help="1.0 = permanent · lower = faster fade",
    )

    # ── Overlay mode ───────────────────────────────────────────────────
    st.markdown("---")
    overlay_mode = st.selectbox(
        "Feed Overlay Mode",
        ["Hybrid (All Layers)", "Boxes & HUD Only", "Heatmap Only", "Line Counter Only"],
        index=0,
    )

    # ── Reset ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🧹 Reset All State & Logs"):
        if st.session_state.tracker:
            st.session_state.tracker.reset()
        if st.session_state.line_counter:
            st.session_state.line_counter.reset()
        if st.session_state.heatmap:
            st.session_state.heatmap.reset()
        if st.session_state.det_logger:
            st.session_state.det_logger.clear_logs()
        utils.reset_fps()
        st.session_state.alert_count  = 0
        st.session_state.entry_count  = 0
        st.session_state.exit_count   = 0
        st.session_state.last_breach_id = None
        st.success("All state reset.")
        time.sleep(0.8)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# LAZILY INIT STATEFUL OBJECTS (once per session)
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.tracker is None:
    st.session_state.tracker = ObjectTracker(detector)
if st.session_state.line_counter is None:
    st.session_state.line_counter = LineCrossingCounter(line_y_pct=line_pct)
if st.session_state.heatmap is None:
    st.session_state.heatmap = MovementHeatmap(decay=heatmap_decay)
if st.session_state.det_logger is None:
    st.session_state.det_logger = DetectionLogger(logs_dir=DIRS["logs"])
if st.session_state.analytics is None:
    st.session_state.analytics = SurveillanceAnalytics(
        logs_dir=DIRS["logs"], reports_dir=DIRS["reports"]
    )

# Shorthand aliases (no reassignment — keep session_state references)
tracker      = st.session_state.tracker
tracker.detector = detector          # refresh in case model changed
line_counter = st.session_state.line_counter
heatmap_gen  = st.session_state.heatmap
det_logger   = st.session_state.det_logger
analytics    = st.session_state.analytics

# Sync line counter position from sidebar
line_counter.line_y_pct = line_pct
heatmap_gen.decay       = heatmap_decay

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("<h1 class='app-title'>🛡️ AI Smart Surveillance & Edge Analytics</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p class='app-sub'>YOLOv8 · Multi-Object Tracking · Restricted Zones · "
    "Line Counting · Movement Heatmaps · Export Reports</p>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# KPI ROW (always visible, updates on rerun)
# ═══════════════════════════════════════════════════════════════════════════
stats = analytics.get_summary_statistics(
    det_logger.detection_log_path,
    det_logger.alert_log_path,
    entry_count=st.session_state.entry_count,
    exit_count=st.session_state.exit_count,
)
active_now = sum(tracker.get_active_counts().values())

k1, k2, k3, k4, k5, k6 = st.columns(6)
for col, val, label, cls in [
    (k1, stats["total_detections"],   "Logged Objects",    "c-cyan"),
    (k2, active_now,                  "Active in Frame",   "c-purple"),
    (k3, st.session_state.entry_count,"Line Entries",      "c-orange"),
    (k4, st.session_state.exit_count, "Line Exits",        "c-green"),
    (k5, st.session_state.alert_count,"Zone Alerts",       "c-red"),
    (k6, stats["most_detected"],      "Top Class",         "c-amber"),
]:
    col.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-val {cls}'>{val}</div>"
        f"<div class='kpi-label'>{label}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# INPUT SOURCE — collected OUTSIDE tabs so always in scope for the loop
# ═══════════════════════════════════════════════════════════════════════════
src_col, cam_col, _ = st.columns([2, 1, 1])
with src_col:
    input_source = st.radio(
        "Feed Source",
        ["📷 Webcam (live)", "🎬 Upload Video"],
        horizontal=True,
        label_visibility="collapsed",
    )
with cam_col:
    cam_index = st.selectbox(
        "Camera Index",
        [0, 1, 2, 3],
        index=0,
        help="Try 0 first. If blank, try 1 or 2 for external cameras.",
        disabled=(input_source != "📷 Webcam (live)"),
    )
video_file = None
if input_source == "🎬 Upload Video":
    video_file = st.file_uploader(
        "Upload surveillance clip",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
(
    tab_feed, tab_analytics, tab_history,
    tab_export, tab_gallery
) = st.tabs([
    "📹 Surveillance Feed",
    "📊 Analytics",
    "🕒 Event Timeline",
    "💾 Export Reports",
    "📸 Screenshot Gallery",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SURVEILLANCE FEED
# ═══════════════════════════════════════════════════════════════════════════
with tab_feed:
    feed_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        st.markdown("<p class='sec-head'>🕹️ Controls</p>", unsafe_allow_html=True)

        # Start / Stop
        if not st.session_state.running:
            if st.button("▶️ Start Detection", type="primary"):
                utils.reset_fps()
                st.session_state.running = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Detection"):
                st.session_state.running = False
                st.rerun()

        # Capture snapshot
        if st.button("📸 Capture Frame  [S]",
                     disabled=not st.session_state.running):
            st.session_state.capture_flag = True

        # Reset counters only
        if st.button("🔄 Reset Counters"):
            line_counter.reset()
            tracker.reset()
            heatmap_gen.reset()
            utils.reset_fps()
            st.session_state.entry_count  = 0
            st.session_state.exit_count   = 0
            st.session_state.alert_count  = 0
            st.session_state.last_breach_id = None
            st.rerun()

        st.markdown("---")
        st.markdown("<p style='font-weight:600;font-size:.85rem;'>Active Objects</p>",
                    unsafe_allow_html=True)
        live_counts = tracker.get_active_counts()
        if live_counts:
            for cls_n, cnt in sorted(live_counts.items()):
                st.markdown(
                    f"<span style='color:#94a3b8;font-size:.85rem;'>"
                    f"**{cls_n.capitalize()}** — {cnt}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No objects in frame.")

    with feed_col:
        frame_ph  = st.empty()   # video frame
        alert_ph  = st.empty()   # zone breach alert

        if not st.session_state.running:
            frame_ph.markdown(
                "<div class='offline-box'>"
                "<div class='offline-icon'>📷</div>"
                "<p style='margin-top:.8rem;font-size:1rem;'>Surveillance feed offline</p>"
                "<p style='font-size:.85rem;color:#475569;'>Press ▶️ Start Detection to begin</p>"
                "</div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════
# MAIN VIDEO PROCESSING LOOP
# Runs only when running == True; lives OUTSIDE all tab contexts so that
# `input_source`, `video_file`, `overlay_mode` (sidebar) are always in scope.
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.running:
    cap       = None
    tmp_path  = None

    # ── Open video source ─────────────────────────────────────────────
    if input_source == "📷 Webcam (live)":
        # Use DirectShow backend on Windows for reliable webcam access
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback: try default backend
            cap = cv2.VideoCapture(cam_index)
        # Request HD resolution — camera may clamp to its native max
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_FPS,           30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,     1)  # Minimise buffer lag

    elif input_source == "🎬 Upload Video" and video_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_file.read())
        tmp_path = tmp.name
        tmp.close()
        cap = cv2.VideoCapture(tmp_path)

    if cap is None or not cap.isOpened():
        st.error(
            f"❌ Cannot open {'camera index ' + str(cam_index) if input_source == '📷 Webcam (live)' else 'the uploaded video'}. "
            "\n\n**Webcam tips:** Make sure no other app (Teams, Zoom) is using it. "
            "Try a different Camera Index (0, 1, 2) in the dropdown above."
        )
        st.session_state.running = False
        if cap:
            cap.release()

    else:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                if input_source == "🎬 Upload Video":
                    st.info("✅ Video playback complete.")
                else:
                    st.warning(
                        f"⚠️ Camera index {cam_index} stopped sending frames. "
                        "Try restarting or selecting a different camera index."
                    )
                st.session_state.running = False
                break

            fh, fw = frame.shape[:2]

            # ── Detect / Track ────────────────────────────────────────
            detections = tracker.update(
                frame,
                conf_threshold=conf_thresh,
                classes=selected_ids if selected_ids else None,
                use_tracking=use_tracking,
            )
            active_counts = tracker.get_active_counts()

            # ── Line crossing ─────────────────────────────────────────
            if line_on:
                prev_entry = line_counter.entry_count
                prev_exit  = line_counter.exit_count
                line_counter.update(detections, tracker.paths, fw, fh)

                if line_counter.entry_count != prev_entry:
                    st.session_state.entry_count = line_counter.entry_count
                    det_logger.log_alert(
                        "Entry: object crossed counting line",
                        next(
                            (d["track_id"] for d in detections if d.get("track_id") is not None),
                            None,
                        ),
                        0.99,
                    )
                if line_counter.exit_count != prev_exit:
                    st.session_state.exit_count = line_counter.exit_count
                    det_logger.log_alert(
                        "Exit: object crossed counting line",
                        next(
                            (d["track_id"] for d in detections if d.get("track_id") is not None),
                            None,
                        ),
                        0.99,
                    )

            # ── Heatmap accumulation ──────────────────────────────────
            if heatmap_on:
                heatmap_gen.update(detections, fw, fh)

            # ── Zone breach check ─────────────────────────────────────
            zone_breached = False
            if zone:
                for det in detections:
                    if det["class_name"].lower() == "person":
                        if utils.check_zone_breach(det["box"], zone, fw, fh):
                            zone_breached = True
                            tid = det.get("track_id")
                            if tid != st.session_state.last_breach_id:
                                st.session_state.last_breach_id = tid
                                st.session_state.alert_count   += 1
                                det_logger.log_alert(
                                    "Person entered restricted zone",
                                    tid,
                                    det["confidence"],
                                )

            # ── Log detections (de-duped) ─────────────────────────────
            det_logger.log_detections(detections)

            # ── Screenshot ────────────────────────────────────────────
            fps = utils.calculate_fps()

            # Build annotated frame first, then maybe snapshot it
            annotated = frame.copy()

            if overlay_mode in ("Hybrid (All Layers)", "Heatmap Only") and heatmap_on:
                annotated = heatmap_gen.get_overlay(annotated)

            if overlay_mode in ("Hybrid (All Layers)", "Boxes & HUD Only"):
                annotated = utils.draw_annotations(
                    annotated, detections, use_tracking,
                    active_counts, zone, zone_breached, fps,
                )

            if overlay_mode in ("Hybrid (All Layers)", "Line Counter Only") and line_on:
                annotated = line_counter.draw(annotated)

            if overlay_mode == "Heatmap Only" and not heatmap_on:
                annotated = utils.draw_annotations(
                    annotated, detections, use_tracking,
                    active_counts, zone, zone_breached, fps,
                )

            if st.session_state.capture_flag:
                path = utils.save_screenshot(annotated, DIRS["screenshots"])
                st.session_state.capture_flag = False
                st.toast(f"📸 Saved: {os.path.basename(path)}", icon="✅")

            # ── Render to Streamlit ───────────────────────────────────
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_ph.image(rgb, channels="RGB", width="stretch")

            if zone_breached:
                alert_ph.markdown(
                    "<div class='alert-box'>⚠️ ALERT: Person entered restricted zone!</div>",
                    unsafe_allow_html=True,
                )
            else:
                alert_ph.empty()

            time.sleep(0.005)   # yield — keep Streamlit responsive

        # ── Cleanup ───────────────────────────────────────────────────
        cap.release()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("<p class='sec-head'>📊 Detection Analytics</p>", unsafe_allow_html=True)

    if stats["total_detections"] == 0:
        st.info("No detection data yet. Start surveillance to populate analytics.")
    else:
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("**Object Distribution**")
            dist_df = pd.DataFrame(
                stats["distribution"].items(),
                columns=["Class", "Count"],
            ).sort_values("Count", ascending=False)

            donut = (
                alt.Chart(dist_df)
                .mark_arc(innerRadius=45, cornerRadius=5)
                .encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color(
                        "Class:N",
                        scale=alt.Scale(scheme="category20c"),
                        legend=alt.Legend(orient="bottom", columns=3),
                    ),
                    tooltip=["Class", "Count"],
                )
                .properties(height=280)
            )
            st.altair_chart(donut, use_container_width=True)  # altair charts keep this API

        with col_b:
            st.markdown("**Summary Metrics**")
            st.dataframe(dist_df, hide_index=True, use_container_width=True)  # dataframe keeps this API
            st.markdown(
                f"""
                | Metric | Value |
                |--------|-------|
                | Most Detected | **{stats['most_detected']}** ({stats['most_detected_count']}) |
                | Zone Alerts | **{st.session_state.alert_count}** |
                | Line Entries | **{st.session_state.entry_count}** |
                | Line Exits | **{st.session_state.exit_count}** |
                | Total Logged | **{stats['total_detections']}** |
                """
            )

        # ── Hourly frequency bar chart ────────────────────────────────
        if stats["frequency_by_hour"]:
            st.markdown("**Detection Frequency by Hour**")
            freq_df = pd.DataFrame(
                [(f"{h:02d}:00", c) for h, c in stats["frequency_by_hour"].items()],
                columns=["Hour", "Events"],
            )
            bar = (
                alt.Chart(freq_df)
                .mark_bar(
                    color="#4facfe",
                    cornerRadiusTopLeft=5,
                    cornerRadiusTopRight=5,
                )
                .encode(
                    x=alt.X("Hour:N", sort=None, title="Hour of Day"),
                    y=alt.Y("Events:Q", title="Detections"),
                    tooltip=["Hour", "Events"],
                )
                .properties(height=220)
            )
            st.altair_chart(bar, use_container_width=True)  # altair charts keep this API

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — EVENT TIMELINE
# ═══════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("<p class='sec-head'>🕒 Activity Timeline</p>", unsafe_allow_html=True)

    df_dets   = det_logger.get_recent_detections(limit=500)
    df_alerts = det_logger.get_recent_alerts(limit=200)

    # Merge all events into one chronological list
    events = []

    if not df_dets.empty:
        for _, row in df_dets.iterrows():
            try:
                rt = datetime.strptime(str(row["Timestamp"]), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                rt = datetime.now()
            events.append({
                "rt": rt, "type": "Detection",
                "msg": (f"Detected {row['Object']}  "
                        f"conf {float(row['Confidence'])*100:.0f}%  "
                        f"ID #{row['Tracking_ID']}"),
            })

    if not df_alerts.empty:
        for _, row in df_alerts.iterrows():
            try:
                rt = datetime.strptime(str(row["Timestamp"]), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                rt = datetime.now()
            ev = str(row["Event"])
            etype = "Line Crossing" if "line" in ev.lower() else "Zone Violation"
            events.append({"rt": rt, "type": etype, "msg": ev})

    events.sort(key=lambda e: e["rt"], reverse=True)

    if not events:
        st.info("No events yet. Start the surveillance feed to populate the timeline.")
    else:
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            type_filter = st.multiselect(
                "Event Types",
                ["Detection", "Zone Violation", "Line Crossing"],
                default=["Detection", "Zone Violation", "Line Crossing"],
            )
        with fc2:
            kw = st.text_input("Search", placeholder="e.g. Person, Alert, Entry…")

        filtered = [
            e for e in events
            if e["type"] in type_filter
            and (not kw or kw.lower() in e["msg"].lower())
        ]

        st.markdown("<br>", unsafe_allow_html=True)
        for ev in filtered[:80]:
            t_str  = ev["rt"].strftime("%I:%M:%S %p")
            etype  = ev["type"]
            extra  = " tl-alert" if etype == "Zone Violation" else " tl-cross" if etype == "Line Crossing" else ""
            icon   = "🚨" if etype == "Zone Violation" else "📏" if etype == "Line Crossing" else "🔍"
            st.markdown(
                f"<div class='tl-item{extra}'>"
                f"<span class='tl-time'>{t_str}</span>"
                f"<span class='tl-text'>{icon} {ev['msg']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT REPORTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("<p class='sec-head'>💾 Export Reports</p>", unsafe_allow_html=True)
    st.caption("Download session data as Excel workbook or raw CSV.")

    ec1, ec2 = st.columns(2)

    with ec1:
        st.markdown("#### 📊 Excel Workbook (.xlsx)")
        st.markdown(
            "Three worksheets: **Executive Summary**, **Detection Log**, **Alerts Log**."
        )
        xlsx_data = analytics.generate_excel_report(
            det_logger.detection_log_path,
            det_logger.alert_log_path,
            stats,
        )
        st.download_button(
            "⬇️ Download Excel Report",
            data=xlsx_data,
            file_name=f"surveillance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with ec2:
        st.markdown("#### 📄 Raw CSV Log (.csv)")
        st.markdown("Timestamps, object classes, confidence scores, tracking IDs.")
        csv_data = analytics.generate_csv_report(det_logger.detection_log_path)
        st.download_button(
            "⬇️ Download CSV Log",
            data=csv_data,
            file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — SCREENSHOT GALLERY
# ═══════════════════════════════════════════════════════════════════════════
with tab_gallery:
    st.markdown("<p class='sec-head'>📸 Screenshot Gallery</p>", unsafe_allow_html=True)

    shots = sorted(
        [
            os.path.join(DIRS["screenshots"], f)
            for f in os.listdir(DIRS["screenshots"])
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ],
        key=os.path.getmtime,
        reverse=True,
    )

    if not shots:
        st.info("No screenshots yet. Click '📸 Capture Frame' during live detection.")
    else:
        st.caption(f"{len(shots)} screenshot(s) saved.")
        cols = st.columns(4)
        for i, path in enumerate(shots[:16]):
            with cols[i % 4]:
                try:
                    img = Image.open(path)
                    st.image(img, caption=os.path.basename(path), width="stretch")
                    with open(path, "rb") as fp:
                        st.download_button(
                            "⬇️ Download",
                            data=fp.read(),
                            file_name=os.path.basename(path),
                            mime="image/jpeg",
                            key=f"dl_{i}",
                        )
                except Exception:
                    st.warning(f"Could not load {os.path.basename(path)}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Delete All Screenshots"):
            for p in shots:
                try:
                    os.remove(p)
                except OSError:
                    pass
            st.success("Gallery cleared.")
            time.sleep(0.5)
            st.rerun()
