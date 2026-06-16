# 🛡️ AI Smart Surveillance Platform — Complete Feature Guide

> **Technology Stack:** Python · YOLOv8 (Ultralytics) · OpenCV · Streamlit · ByteTrack  
> **Version:** Production v2.0 | **Compatibility:** Windows 10/11 · Python 3.10+

---

## 📋 Table of Contents

1. [Getting Started](#1-getting-started)
2. [Real-Time Object Detection](#2-real-time-object-detection)
3. [Multi-Object Tracking](#3-multi-object-tracking)
4. [Live Object Counting](#4-live-object-counting)
5. [FPS Performance Monitor](#5-fps-performance-monitor)
6. [Feed Sources — Webcam & Video Upload](#6-feed-sources--webcam--video-upload)
7. [Restricted Zone Alert System](#7-restricted-zone-alert-system)
8. [Virtual Line Crossing Counter](#8-virtual-line-crossing-counter)
9. [Movement Heatmap Overlay](#9-movement-heatmap-overlay)
10. [Overlay Display Modes](#10-overlay-display-modes)
11. [Screenshot Capture](#11-screenshot-capture)
12. [Detection Logging (CSV)](#12-detection-logging-csv)
13. [Analytics Dashboard](#13-analytics-dashboard)
14. [Event Timeline Viewer](#14-event-timeline-viewer)
15. [Export Reports (Excel & CSV)](#15-export-reports-excel--csv)
16. [Screenshot Gallery](#16-screenshot-gallery)
17. [Model Selection](#17-model-selection)
18. [Class / Category Filter](#18-class--category-filter)
19. [Confidence Threshold](#19-confidence-threshold)
20. [Session State & Reset Controls](#20-session-state--reset-controls)
21. [Project Architecture](#21-project-architecture)
22. [Troubleshooting](#22-troubleshooting)

---

## 1. Getting Started

### Prerequisites
Make sure you have installed all required dependencies:

```bash
pip install -r requirements.txt
```

### Running the App
```bash
python -m streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

### Quick Start Checklist
- [ ] Select your **AI model** from the sidebar (start with `yolov8n.pt` for fastest speed)
- [ ] Choose **Detect Classes** to filter what objects you want tracked
- [ ] Select **Feed Source** — Webcam or Upload Video
- [ ] Press **▶️ Start Detection** to begin
- [ ] Explore the 5 tabs: Feed · Analytics · Timeline · Export · Gallery

---

## 2. Real-Time Object Detection

### What It Does
Runs the **YOLOv8 neural network** on every camera frame to locate objects within milliseconds. It draws labelled bounding boxes around each detected item.

### What You See on Screen
```
┌──────────────────────┐
│  Person #3  92%      │   ← Label chip (class + ID + confidence)
│                      │
│                      │
└──────────────────────┘   ← Coloured bounding box
```

### Colour Coding
Each object class gets a **unique distinct colour**:

| Class | Colour |
|-------|--------|
| Person | 🔵 Sky Blue |
| Car | 🟠 Neon Orange |
| Bottle | 🟢 Lime Green |
| Laptop | 🟣 Magenta |
| Cell Phone | 🌸 Rose Pink |
| Dog | 🔶 Amber |
| Cat | 💜 Light Violet |

### Technical Details
- **Model:** YOLOv8 (COCO dataset, 80 object classes)
- **Input:** BGR frames from OpenCV
- **Output:** Bounding boxes `[xmin, ymin, xmax, ymax]`, class name, confidence score
- **Framework:** Ultralytics YOLO Python SDK

---

## 3. Multi-Object Tracking

### What It Does
Assigns a **persistent unique ID** (e.g., `Person #1`, `Person #2`) to each detected object and maintains that ID as the object moves across the frame. Even if an object briefly leaves frame and returns, it may retain the same ID.

### How It Works
Uses **YOLOv8's built-in ByteTrack algorithm** — a high-performance multi-object tracker that uses motion prediction (Kalman filter) and IoU matching to associate detections across frames.

### Example Output
```
Person #1 (94%)    ← Person walking left
Person #2 (87%)    ← Different person standing still
Bottle #3 (76%)    ← Bottle on table
Car #4 (91%)       ← Car moving in background
```

### Controls
- **Toggle:** "Multi-Object Tracking" switch in the sidebar
- When **OFF**: Objects are detected each frame but get no persistent ID (ephemeral numbering only)
- When **ON**: IDs persist across frames (default)

### Path History
- The system maintains the last **30 coordinate points** for each tracked ID
- This history is used by the **Line Counter** and **Heatmap** features
- Stale IDs (absent for 60+ frames) are automatically pruned to prevent memory leaks

---

## 4. Live Object Counting

### What It Does
Shows a real-time panel in the top-left corner of the video feed listing how many objects of each class are currently visible in the current frame.

### On-Screen Display
```
┌─────────────────┐
│  LIVE OBJECTS   │
│  Person: 3      │
│  Car: 1         │
│  Bottle: 2      │
└─────────────────┘
```

### In the Sidebar (Controls Panel)
The **"Active Objects"** section in the feed controller also shows the live per-class count, updated every frame.

### KPI Cards (Top of Dashboard)
The 6 KPI cards at the top always show:
- **Logged Objects** — total records written to CSV across the session
- **Active in Frame** — objects detected right now
- **Line Entries** — people who crossed the virtual line going down
- **Line Exits** — people who crossed the virtual line going up
- **Zone Alerts** — times the restricted zone was breached
- **Top Class** — most frequently detected category

---

## 5. FPS Performance Monitor

### What It Does
Displays the current **Frames Per Second** rate in the top-right corner of the video feed. Uses a **30-frame rolling average** for a smooth, stable reading.

### FPS Colour Coding
| FPS | Colour | Meaning |
|-----|--------|---------|
| ≥ 20 | 🟢 Green | Excellent — running smoothly |
| 10–19 | 🟠 Orange | Acceptable — slight lag possible |
| < 10 | 🔴 Red | Poor — consider switching to `yolov8n.pt` |

### How to Improve FPS
1. Use **yolov8n.pt** (Nano) instead of larger models
2. Lower the camera resolution
3. Reduce the number of **selected classes** in the filter
4. Increase the **Confidence Threshold** slider to skip weak detections

---

## 6. Feed Sources — Webcam & Video Upload

### Webcam (Live)

**How to use:**
1. Select `📷 Webcam (live)` from the feed source radio button
2. Choose **Camera Index** (0 = built-in webcam, 1/2 = external USB cameras)
3. Press **▶️ Start Detection**

**Camera Index Guide:**
| Index | Typically |
|-------|-----------|
| 0 | Built-in laptop camera |
| 1 | First USB/external camera |
| 2 | Second USB camera |
| 3 | Virtual cameras (OBS, etc.) |

**Technical implementation:**
- Uses **DirectShow (`cv2.CAP_DSHOW`) backend** on Windows for fastest, most reliable access
- Falls back to default OpenCV backend if DirectShow fails
- Buffer size set to **1 frame** to minimise latency
- Resolution requested: 1280×720 (camera may override if unsupported)

### Video Upload

**How to use:**
1. Select `🎬 Upload Video`
2. Drag-and-drop or click to upload an MP4 / AVI / MOV / MKV file
3. Press **▶️ Start Detection**

**Supported formats:** `.mp4`, `.avi`, `.mov`, `.mkv`  
**Max upload size:** 500 MB (configured in `.streamlit/config.toml`)

The video plays back frame-by-frame through the detection pipeline, exactly like a webcam. When the video ends, it stops automatically and shows "✅ Video playback complete."

---

## 7. Restricted Zone Alert System

### What It Does
You define a **rectangular restricted area** on the frame. Whenever a **person** enters this zone, the system:
- Flashes a **red pulsing alert banner** below the video feed
- Draws a **red filled overlay** on the zone
- Increments the **Zone Alerts** counter
- Logs an alert entry to `alerts_YYYYMMDD.csv`

### How to Configure
In the sidebar under **"🛡️ Restricted Zone"**:
1. Enable the **"Enable Zone Alert"** toggle
2. Use the **X min / X max / Y min / Y max sliders** to define zone boundaries
   - All values are **proportions of the frame** (0.0 to 1.0)
   - Example: X min=0.3, X max=0.7, Y min=0.3, Y max=0.7 → central rectangle

### Visual Feedback
| State | Zone Colour | Effect |
|-------|-------------|--------|
| Normal | 🟠 Orange border | "RESTRICTED AREA" label |
| Breached | 🔴 Red border + fill | "⚠ ZONE BREACHED!" label + red overlay |

### Alert Deduplication
The system only triggers a **new alert when a new unique tracking ID** enters the zone. If the same person (same track ID) is already inside, no repeated alerts fire. This prevents spam.

---

## 8. Virtual Line Crossing Counter

### What It Does
Draws a **horizontal counting line** across the video frame. When any tracked object crosses it:
- Crossing **downward** = **Entry** (counted with green arrow)
- Crossing **upward** = **Exit** (counted with orange arrow)

This is used in retail, doorways, turnstiles, and corridors to count foot traffic.

### How to Configure
In the sidebar under **"📏 Counting Line"**:
1. Enable the **"Enable Line Counter"** toggle
2. Use the **"Line Position (% height)"** slider to move the line (0.1 = top, 0.9 = bottom)

### On-Screen Display
```
  ENTRY  ↓                    ↑  EXIT
──────────── COUNTING LINE ────────────
              Entry : 12
              Exit  : 8
```

The counter chip appears in the top-right of the video feed.

### Technical Logic
- Uses the **last 2 coordinate points** of each track's path history
- If previous Y < line_Y and current Y ≥ line_Y → **Entry**
- If previous Y > line_Y and current Y ≤ line_Y → **Exit**
- Each track ID can only trigger one state change per direction (prevents jitter double-counting)

---

## 9. Movement Heatmap Overlay

### What It Does
Visualises **where objects have been moving** by painting a colour-coded heat layer over the video. Areas with heavy traffic glow **red**, medium traffic is **yellow/green**, and low-traffic zones stay **blue** or transparent.

### Colour Meaning
| Colour | Meaning |
|--------|---------|
| 🔵 Blue | Low / rare activity |
| 🟡 Yellow | Moderate traffic |
| 🔴 Red | High-frequency hotspot |

### How to Configure
In the sidebar under **"🔥 Movement Heatmap"**:
1. Enable the **"Enable Heatmap Overlay"** toggle
2. Adjust **Trail Decay** slider:
   - `1.00` = Permanent accumulation — trails never fade
   - `0.97` = Slow fade — trails persist ~30 frames
   - `0.90` = Fast fade — only recent movement visible

### Technical Implementation
- Each object centre point is drawn as a **Gaussian-blurred circle** on a float32 accumulator grid
- Normalised to 0–255 and coloured with OpenCV's **COLORMAP_JET**
- A threshold mask ensures only active pixels overlay the background (background stays clear)
- Blended 55% heatmap / 45% original frame for readability

---

## 10. Overlay Display Modes

### What It Does
Controls which visual layers are rendered on the video feed.

### Available Modes

| Mode | What's Shown |
|------|-------------|
| **Hybrid (All Layers)** | Heatmap + Boxes/HUD + Line Counter (everything) |
| **Boxes & HUD Only** | Bounding boxes, labels, zone, FPS panel |
| **Heatmap Only** | Pure heatmap overlay (no boxes) |
| **Line Counter Only** | Just the counting line and Entry/Exit counter |

> **Tip:** Use "Boxes & HUD Only" for surveillance. Use "Heatmap Only" to analyze traffic patterns after recording.

---

## 11. Screenshot Capture

### What It Does
Saves the **current annotated frame** (with all overlays drawn) as a JPEG image to the `screenshots/` folder.

### How to Use
- Click the **"📸 Capture Frame [S]"** button in the feed controller panel
- A toast notification confirms: `📸 Saved: detection_20261201_143022.jpg`

### File Naming
```
screenshots/
  detection_20261201_143022.jpg
  detection_20261201_144510.jpg
```
Format: `detection_YYYYMMDD_HHMMSS.jpg`

### Quality
Saved at **JPEG quality 95** — high quality with moderate file size.

---

## 12. Detection Logging (CSV)

### What It Does
Automatically writes every detected object to a **daily CSV log file**. New file created each day.

### File Location
```
logs/
  detections_20261201.csv    ← Today's detection log
  alerts_20261201.csv        ← Today's alert log
```

### Detection Log Schema
```
Timestamp,Object,Confidence,Tracking_ID
2026-12-01 14:30:22,person,0.94,1
2026-12-01 14:30:22,bottle,0.77,2
2026-12-01 14:30:25,car,0.91,3
```

### Alert Log Schema
```
Timestamp,Event,Tracking_ID,Confidence
2026-12-01 14:31:05,Person entered restricted zone,4,0.94
2026-12-01 14:31:12,Entry: object crossed counting line,5,0.99
```

### De-duplication
Detection rows are written **once per unique tracking ID per session** — not every frame. This keeps file sizes small and meaningful. Alert events are **always written** (no deduplication) since every breach matters.

### Thread Safety
The logger uses a **`threading.Lock()`** to ensure no data corruption when the detection loop writes concurrently with the UI reading the file.

---

## 13. Analytics Dashboard

### What It Does
The **📊 Analytics** tab shows statistical insights compiled from the current day's log files.

### Panels

#### Object Distribution Donut Chart
An interactive Altair chart showing the proportion of each detected class logged. Hover over segments to see exact counts.

#### Summary Metrics Table
A sortable table showing:
- All detected classes
- Count per class
- Sorted by frequency (most detected first)

#### System Core Metrics
```
Most Tracked Category: Person (Count: 147)
Virtual Line Entry Count: 23
Virtual Line Exit Count: 19
Restricted Zone Alerts Raised: 4
Total Detections Logged: 312
```

#### Detection Frequency by Hour
A bar chart showing which hours of the day had the most detection activity — useful for understanding peak traffic periods.

---

## 14. Event Timeline Viewer

### What It Does
The **🕒 Event Timeline** tab shows a **chronological log** of all events (detections, zone breaches, line crossings) in a modern card-based layout.

### Event Types & Colours
| Type | Left Border | Icon | Meaning |
|------|-------------|------|---------|
| Detection | 🔵 Blue | 🔍 | Normal object detected |
| Zone Violation | 🔴 Red | 🚨 | Person entered restricted zone |
| Line Crossing | 🟠 Orange | 📏 | Object crossed counting line |

### Filters
- **Event Types** multi-select: show/hide detection/violation/crossing events
- **Search bar**: filter by keyword (e.g., "person", "entry", "alert")

### Display Limit
Shows the **most recent 80 events** matching the filter. Older events are still in the CSV but not shown to keep the UI fast.

---

## 15. Export Reports (Excel & CSV)

### What It Does
The **💾 Export Reports** tab lets you download your session data in two formats.

### Excel Workbook (.xlsx)
Contains **3 worksheets**:

| Sheet | Contents |
|-------|---------|
| Executive Summary | Key KPIs in a clean table |
| Detection Log | Full detection CSV as a formatted sheet |
| Alerts Log | All zone/line events as a formatted sheet |

Download button: **"⬇️ Download Excel Report"**  
Filename: `surveillance_YYYYMMDD_HHMMSS.xlsx`

### Raw CSV Log (.csv)
The raw detection log file — timestamps, objects, confidence, IDs.

Download button: **"⬇️ Download CSV Log"**  
Filename: `detections_YYYYMMDD_HHMMSS.csv`

---

## 16. Screenshot Gallery

### What It Does
The **📸 Screenshot Gallery** tab shows a **4-column grid** of all saved screenshots, with individual download buttons.

### Features
- Displays up to **16 most recent screenshots**
- **Download button** per image
- **"🗑️ Delete All Screenshots"** wipes the entire gallery
- Auto-sorted newest-first

---

## 17. Model Selection

### Available Models

| Model | File | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **YOLOv8 Nano** | `yolov8n.pt` | ⚡⚡⚡ Fastest | ⭐⭐ Good | Low-power machines, real-time webcam |
| **YOLOv8 Small** | `yolov8s.pt` | ⚡⚡ Fast | ⭐⭐⭐ Better | Balanced webcam use |
| **YOLOv8 Medium** | `yolov8m.pt` | ⚡ Moderate | ⭐⭐⭐⭐ Best | Video analysis where accuracy matters |

### Auto-Download
Models are **automatically downloaded** from Ultralytics on first use and cached to the `models/` folder. Subsequent runs load from cache instantly.

### Switching Models
Change the model in the sidebar → the new model loads immediately (cached with `@st.cache_resource`).

---

## 18. Class / Category Filter

### What It Does
Restricts detection to only the object classes you care about — improving both speed and relevance.

### How to Use
In the sidebar under **"Detect Classes"**:
- Select or deselect classes from the multiselect dropdown
- Default selection: person, bottle, car, laptop, cell phone

### Available Classes
YOLOv8 (COCO) supports **80 classes** including:
`person, bicycle, car, motorcycle, bus, truck, cat, dog, chair, bottle, cup, laptop, cell phone, book, clock, umbrella, handbag, backpack` and many more.

### Impact
Filtering to fewer classes:
- Speeds up inference (fewer NMS comparisons)
- Reduces false positives
- Keeps logs clean and focused

---

## 19. Confidence Threshold

### What It Does
Controls the **minimum confidence score** a detection must have to be shown.

### Slider: `0.10 → 0.95` (default: `0.30`)

| Value | Effect |
|-------|--------|
| Low (0.10–0.20) | Very sensitive — catches distant/partial objects but may give false positives |
| Default (0.30) | Balanced — good for most scenarios |
| High (0.60–0.95) | Only very certain detections — cleaner but may miss some objects |

### Tuning Tips
- **Crowded scene** → raise to 0.40–0.50 to cut noise
- **Distant objects** → lower to 0.20–0.25 to catch them
- **Testing** → lower to 0.15 to verify the model can see everything

---

## 20. Session State & Reset Controls

### Session Persistence
All tracking state persists **as long as your browser tab is open**. Refreshing the page resets everything. The CSV logs on disk are preserved across refreshes.

### Reset Buttons

| Button | Location | What It Resets |
|--------|----------|---------------|
| **🔄 Reset Counters** | Feed Controller | Line counts, tracker paths, heatmap, FPS |
| **🧹 Reset All State & Logs** | Sidebar bottom | Everything above + wipes CSV log files |

---

## 21. Project Architecture

```
smart_object_detection/
│
├── app.py             Main Streamlit dashboard (UI + orchestration)
├── detector.py        YOLOv8 inference engine (detect + track)
├── tracker.py         Stateful multi-object tracking manager
├── logger.py          Thread-safe CSV detection & alert logger
├── line_counter.py    Virtual line crossing detector
├── heatmap.py         Gaussian movement heatmap generator
├── analytics.py       Statistics aggregator + Excel/CSV report generator
├── utils.py           Frame drawing, FPS, zone check, screenshot utils
│
├── .streamlit/
│   └── config.toml    Dark theme + server settings
│
├── models/            YOLOv8 weight files (auto-downloaded)
├── logs/              Daily CSV detection & alert logs
├── screenshots/       Captured annotated frames
├── reports/           Generated Excel reports
│
└── requirements.txt   Python dependencies
```

### Data Flow Per Frame
```
Camera/Video → detector.track() → tracker.update()
                                       ↓
                          line_counter.update()
                          heatmap.update()
                          zone breach check
                          logger.log_detections()
                                       ↓
                          utils.draw_annotations()
                          line_counter.draw()
                          heatmap.get_overlay()
                                       ↓
                          frame_ph.image() → Browser
```

---

## 22. Troubleshooting

### 📷 Camera Shows Black Screen / Doesn't Open

| Problem | Solution |
|---------|---------|
| Camera used by another app | Close Teams, Zoom, Discord, or any app using the webcam |
| Wrong camera index | Try indices 1, 2, 3 in the Camera Index dropdown |
| DirectShow backend fails | Update your webcam driver from Device Manager |
| Permissions | Run as Administrator, or check Windows Camera Privacy Settings |

### 🐌 Low FPS / Laggy Feed
1. Switch to **yolov8n.pt** (Nano model)
2. Lower confidence threshold so fewer NMS operations run
3. Reduce the number of selected detection classes
4. Close other browser tabs and heavy apps

### ⚠️ "Cannot open camera" Error
- Make sure camera is physically connected
- Try a different USB port
- Check Windows Device Manager for camera driver errors

### 📊 Analytics Shows No Data
- You must have run detection (▶️ Start) at least once to create logs
- Check if the `logs/` folder exists and contains CSV files
- Click "🧹 Reset All State & Logs" only if you want to start fresh

### 🚨 Zone Alerts Not Triggering
- Make sure **"Enable Zone Alert"** toggle is ON
- Make sure **"person"** is included in the Detect Classes list
- Adjust zone boundaries — verify the zone overlaps where people walk
- Lower the confidence threshold if people near zone edges aren't detected

### 📏 Line Crossing Not Counting
- Make sure **"Enable Line Counter"** toggle is ON
- Make sure **"Multi-Object Tracking"** toggle is ON (tracking IDs needed)
- Position the line where objects actually cross (adjust Y% slider)
- Objects must cross fully — partial crossings may not register

---

## 🏆 Portfolio & Resume Tips

This project demonstrates:

- **Computer Vision**: Real-time inference with state-of-the-art YOLO architecture
- **Machine Learning**: Deploying pre-trained deep learning models in production
- **Software Engineering**: Modular OOP design, threading, error handling
- **Data Engineering**: CSV logging, Excel reporting, Pandas analytics
- **UI/UX**: Professional Streamlit dashboard with custom CSS and charts
- **Systems Programming**: Camera interfacing, frame buffering, FPS optimisation

**Suggested resume bullet points:**
> - Built a real-time AI surveillance system using YOLOv8 and ByteTrack achieving 25+ FPS on CPU
> - Implemented multi-object tracking, virtual zone monitoring, and automated CSV/Excel reporting
> - Designed a modular Python architecture (8 decoupled modules) with thread-safe logging and session state persistence in Streamlit
