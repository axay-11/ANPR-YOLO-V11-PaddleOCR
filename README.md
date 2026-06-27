# ANPR — Indian License Plate Recognition

Production-oriented ANPR pipeline:

```
YOLOv11 (plate detection)  ->  ByteTrack (one ID per vehicle)
   ->  optional plate alignment (--deskew)  ->  PaddleOCR
   ->  IND/noise removal + Indian-format validation  ->  vote across frames
```

The main script is **`anpr_hf.py`**. It downloads a YOLOv11 license-plate model
from the Hugging Face Hub on first run, OCRs each plate with PaddleOCR, and votes
the most frequent reading per tracked vehicle across frames (robust to single-frame
noise).

Plate text is validated against the Indian registration format
`SS DD L(1-3) NNNN` (e.g. `KA01AB1234`). This automatically strips the **`IND`**
emblem text and any spaces/symbols instead of cropping the plate.

---

## 1. Setup

This repo already has a virtual environment in `.venv/`. Activate it:

```bash
source .venv/bin/activate
```

If you ever need to recreate it from scratch:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **macOS camera permission:** the first time you use a camera, macOS asks the
> terminal app (Terminal / iTerm / VS Code) for camera access. You must allow it,
> then **fully quit and reopen** that app, or you'll get a black/empty frame.
> Check **System Settings → Privacy & Security → Camera**.

---

## 2. Pick the right camera

macOS does **not** number cameras predictably (the built-in webcam, a USB camera,
and Continuity Camera all compete for low indices). Use the helper to see what's
connected before running the full pipeline:

```bash
python list_cameras.py          # probes indices 0..5
python list_cameras.py --max 8  # probe more indices
```

It saves one snapshot per working camera to `camera_probe/cam_<index>.jpg`. Open
those images and pick the index showing your camera.

> On this machine: **index `0` = external/USB camera**, **index `1` = built-in
> FaceTime webcam**. (Counter-intuitive, but confirmed via the snapshots.)

---

## 3. Run the ANPR pipeline

```bash
python anpr_hf.py [options]
```

### Common commands

| Goal | Command |
|------|---------|
| Test on the bundled sample video | `python anpr_hf.py --source test_data/20260531153909_043719F.MP4` |
| Same video, with plate alignment | `python anpr_hf.py --source test_data/20260531153909_043719F.MP4 --deskew` |
| Live external/USB camera | `python anpr_hf.py --source 0 --deskew` |
| Built-in webcam | `python anpr_hf.py --source 1` |
| A single image | `python anpr_hf.py --source car.jpg` |
| An IP / RTSP camera | `python anpr_hf.py --source "rtsp://user:pass@192.168.1.50:554/stream1"` |
| Faster (real-time on laptop) | `python anpr_hf.py --source 0 --weights n` |
| More accurate (slower) | `python anpr_hf.py --source 0 --weights m` |
| Show only, don't save a file | `python anpr_hf.py --source 0 --no-save` |
| Headless (no preview window) | `python anpr_hf.py --source 0 --no-show` |

**Stop a live run:** press **`q`** in the preview window, or **`Ctrl+C`** in the
terminal. The final voted plate readings print on exit.

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Camera index (`0`, `1`, …), video file, image, folder, or RTSP URL |
| `--weights` | `s` | YOLOv11 model size: `n` (fastest) → `x` (most accurate): `n,s,m,l,x` |
| `--conf` | `0.25` | Detection confidence threshold (lower = detect more plates) |
| `--min-ocr-conf` | `0.5` | Ignore OCR reads below this confidence |
| `--deskew` | off | Perspective-correct (flatten) each plate before OCR |
| `--no-save` | off | Do not write the annotated output video |
| `--no-show` | off | Do not open a preview window (use for headless/SSH) |

---

## 4. Output

- **Preview window** — live boxes with the current voted plate per vehicle.
- **`anpr_output/annotated.mp4`** — annotated video (unless `--no-save`).
- **Terminal summary** on exit — the final voted reading per tracked vehicle:

  ```
  === Final plate readings (voted across frames) ===
    track 1: KA50MA1665
    track 3: MH12DE5678
  ```

---

## 5. How testing should go

1. **Start with the sample video** (repeatable input, easy to judge accuracy):

   ```bash
   python anpr_hf.py --source test_data/20260531153909_043719F.MP4
   ```

   Check that plate labels are clean (no `IND`, no spaces) and the final summary
   is correct.

2. **A/B test alignment** by adding `--deskew` and comparing the final readings —
   did any plate go wrong→right (or right→wrong)?

3. **Then go live** on the external camera (`--source 0 --deskew`). Make sure the
   plate appears roughly upright in the feed — OCR needs upright text.

---

## Notes & limitations

- **`--deskew`** uses classical segmentation inside each detected crop to find the
  plate's 4 corners and warp it flat. It is a **safe no-op** when it can't find a
  reliable quad (it never makes a readable plate worse), and helps with *moderate*
  skew. It is **not** robust to extreme angles — that needs a trained plate
  **segmentation** model dropped into `_segment_plate_quad()` in `anpr_hf.py`
  (no such Indian model exists off the shelf today).
- The model auto-downloads from Hugging Face on first run and is cached locally.
  Override the repo with the `HF_REPO_ID` environment variable if needed.
- Validation targets standard Indian plates (`SS DD L(1-3) NNNN`). BH-series and
  other special formats are not yet validated.

## Helper scripts

| Script | Purpose |
|--------|---------|
| `anpr_hf.py` | Main ANPR pipeline (detection + tracking + OCR + validation) |
| `list_cameras.py` | Probe/snapshot connected cameras to find the right index |
