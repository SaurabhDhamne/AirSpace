# 🚀 AirSpace  
### A spatial computing interface that transforms air gestures into intelligent system actions
AirSpace is a real-time computer vision system that enables gesture-based drawing and OCR-powered command execution using only a webcam.

It combines hand tracking, virtual canvas rendering, and text recognition to create a touchless interaction interface.

---

## 🔥 Features

- ✋ Real-time Hand Tracking (MediaPipe)
- 🖌️ Air Drawing with Index Finger
- 🧽 Gesture-Based Erasing
- 🧠 OCR Text Recognition from Canvas
- ⚡ Keyword-Based System Automation
- 🎨 Brush Color Switching via Text Commands
- 🔒 OCR Lock & Cooldown Protection
- 🖥️ Opens Apps & Websites via Hand Gestures

---

## 🧠 Gesture Controls

| Gesture | Action |
|----------|--------|
| ☝️ Index Finger | Draw Mode |
| ✌️ Index + Middle | Erase Mode |
| ✊ Fist | Hover Mode |
| 👍 Thumb Up | OCR Scan & Execute Command |
| ✋ Open Palm | Unlock OCR |

---

## 🤖 OCR Command Examples

After writing text in air and showing 👍:

| Written Text | Action |
|--------------|--------|
| `CAL` | Opens Calculator |
| `GG` | Opens Google |
| `YOU` | Opens YouTube |
| `MOM` | Opens WhatsApp Chat |
| `RED` | Switch Brush to Red |
| `BLU` | Switch Brush to Blue |
| `PIN` | Switch Brush to Pink |

---

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- Tesseract OCR
- NumPy

---

## 🏗️ System Architecture

1. Hand landmarks detected via MediaPipe.
2. Finger state calculated (thumb → pinky).
3. Gesture classified based on finger combination.
4. Drawing rendered on virtual canvas.
5. Canvas processed through adaptive thresholding.
6. OCR performed using Tesseract.
7. Keywords parsed and mapped to system commands.

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/SaurabhDhamne/AirSpace.git
cd AirSpace
