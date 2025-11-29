# FootVision-PlayAnnotator

AI-driven football video annotator with **YOLO-based player detection**, **ground-aware chroma key masking**, and **animated highlight rings** that lock precisely near player boots.

This project enables analysts, coaches, content creators, and broadcasters to visually mark players with **broadcast-style glowing rings**, generating replay-quality visuals with minimal manual input.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 YOLO-powered player detection | Automatically detects football players in each frame |
| 🟢 Grass segmentation (HSV chroma key) | Isolates field to avoid drawing on players’ legs |
| 🔴 Multi-color animated highlight rings | Dynamic pulsing effect for replay-quality visuals |
| 👆 Click-to-mark interface | Pause → click a player → highlight is instantly placed |
| 🎬 Export annotated replays | Save as video with effects frozen across highlight frames |
| ↩ Undo & frame-level event storage | Flexible highlight placement during analysis |

---

## 🔥 Why It Stands Out

Most annotation tools draw circles *on top* of players — but football requires **boots-level precision** to analyze pressure, space, and positioning.

This system anchors highlights using both:

1. **YOLO bounding box geometry**
2. **Green chroma mask to identify playable ground**

Result → Rings attach to the *pitch*, not the player’s shin or torso.  
Just like professional broadcast analysis.

---

## 🚀 Quick Start

### 1️⃣ Install dependencies

```bash
pip install ultralytics opencv-python numpy
