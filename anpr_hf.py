"""
Production-oriented ANPR: YOLOv11 (Hugging Face) + ByteTrack + PaddleOCR.

Pipeline:
  1. Detect license plates with a YOLOv11 model from the Hugging Face Hub.
  2. Track each plate across frames with ByteTrack (so a vehicle keeps one ID).
  3. OCR each plate crop with PaddleOCR.
  4. Vote across frames per track ID -> the most frequent reading wins (much more
     robust than trusting a single noisy frame).

Uses the MODERN `ultralytics` pip package (not any vendored copy): this script
removes its own directory from sys.path so a local `ultralytics/` folder cannot
shadow the installed package.

Usage:
    python anpr_hf.py                      # laptop webcam (default)
    python anpr_hf.py --source video.mp4   # a video file
    python anpr_hf.py --source car.jpg     # a single image
    python anpr_hf.py --source 1           # external/USB camera
    python anpr_hf.py --weights n          # smaller/faster model (n,s,m,l,x)
    python anpr_hf.py --deskew             # perspective-correct tilted plates
    python anpr_hf.py --no-save            # show only, don't write output

Plate text is validated against the Indian registration format
(SS DD L(1-3) NNNN), which automatically strips the "IND" emblem text and
other noise instead of cropping. See `extract_indian_plate`.
"""

import os
import sys

# --- Ensure a local ./ultralytics folder does NOT shadow the pip package ---
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _here]

import argparse
import re
from collections import Counter, defaultdict

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# YOLOv11 license-plate model on the Hugging Face Hub (sizes: n, s, m, l, x).
HF_REPO_ID = os.getenv("HF_REPO_ID", "morsetechlab/yolov11-license-plate-detection")
HF_WEIGHT_TEMPLATE = "license-plate-finetune-v1{size}.pt"

# Keep only plausible plate characters when voting/validating.
PLATE_RE = re.compile(r"[^A-Z0-9]")

# Indian registration format: 2 state letters, 1-2 RTO digits, 1-3 series
# letters, 4 unique-number digits (e.g. KA01AB1234, MH12DE5678).
INDIAN_PLATE_RE = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}")

# Noise tokens stamped on the plate that are NOT part of the registration.
# "IND" / the IND emblem appears on every modern Indian plate; strip it.
NOISE_WORDS = ("INDIA", "IND")

# Common OCR confusions, applied only to slots whose type is unambiguous:
# the first 2 chars are always letters, the last 4 are always digits.
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "A": "4", "S": "5", "G": "6", "B": "8"}


def _fix(s, mapping):
    return "".join(mapping.get(c, c) for c in s)


def _coerce(window):
    """Apply position-aware OCR corrections to a plate-length candidate.

    Only the two *unambiguous* zones are corrected: the leading state code
    (must be letters) and the trailing 4-digit number (must be digits). The
    middle (RTO digits + series letters) has a variable split, so we leave it
    untouched rather than guess wrong.
    """
    if len(window) < 7:
        return window
    return _fix(window[:2], TO_LETTER) + window[2:-4] + _fix(window[-4:], TO_DIGIT)


def extract_indian_plate(text):
    """Pull a valid Indian registration number out of raw OCR text.

    Strips IND/INDIA and punctuation, then extracts the registration pattern
    (this discards leftover noise instead of blindly removing substrings).
    Returns (plate, is_valid). When nothing validates, returns the cleaned
    text with is_valid=False so live testing still shows *something*.
    """
    s = PLATE_RE.sub("", (text or "").upper())
    for word in NOISE_WORDS:
        s = s.replace(word, "")

    # Best case: the pattern is already clean.
    m = INDIAN_PLATE_RE.search(s)
    if m:
        return m.group(0), True

    # Fallback: slide plate-length windows and fix the unambiguous slots.
    for length in (10, 9, 8):
        for i in range(0, max(0, len(s) - length) + 1):
            cand = _coerce(s[i:i + length])
            if INDIAN_PLATE_RE.fullmatch(cand):
                return cand, True

    return s, False


def load_ocr():
    """Create a PaddleOCR reader. Imported lazily so detection can run even if
    the OCR install is being sorted out."""
    from paddleocr import PaddleOCR
    # angle classification helps with slightly rotated plates.
    return PaddleOCR(lang="en", use_textline_orientation=True)


def read_plate_text(ocr, crop):
    """Run PaddleOCR on a plate crop and return (text, confidence).

    Handles both PaddleOCR 3.x (`.predict`) and 2.x (`.ocr`) result shapes.
    """
    if crop is None or crop.size == 0:
        return "", 0.0

    texts, scores = [], []
    try:
        # PaddleOCR 3.x
        results = ocr.predict(crop)
        for res in results:
            data = res.json.get("res", res) if hasattr(res, "json") else res
            texts += list(data.get("rec_texts", []))
            scores += list(data.get("rec_scores", []))
    except (AttributeError, TypeError):
        # PaddleOCR 2.x
        results = ocr.ocr(crop)
        for page in results or []:
            for line in page or []:
                texts.append(line[1][0])
                scores.append(float(line[1][1]))

    if not texts:
        return "", 0.0
    # Join multi-line reads, keep only plate-like characters.
    text = PLATE_RE.sub("", "".join(texts).upper())
    conf = sum(scores) / len(scores) if scores else 0.0
    return text, conf


# ---------------------------------------------------------------------------
# Plate alignment (perspective correction)
#
# Pipeline shape: segment the plate -> find its 4 corners -> warpPerspective to
# a flat, fronto-parallel rectangle so OCR reads a straight plate.
#
# A deep plate-segmentation model is the most robust source of the mask, but no
# license-plate-specific seg model is available off the shelf (only generic COCO
# yolov8-seg, which has no plate class). So we segment the plate *classically*
# inside the detected crop. To swap in a trained yolov8-seg model later, replace
# `_segment_plate_quad` with one that returns the mask's 4 corners.
# ---------------------------------------------------------------------------

def _order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],  # top-left  (smallest x+y)
        pts[np.argmin(d)],  # top-right (smallest x-y... i.e. y-x largest -> use diff)
        pts[np.argmax(s)],  # bottom-right (largest x+y)
        pts[np.argmax(d)],  # bottom-left
    ], dtype="float32")


def _segment_plate_quad(crop):
    """Find the plate's 4 corners within a (padded) crop via classical seg.

    Returns a (4, 2) float array of corners, or None if no trustworthy quad.
    """
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)   # denoise, keep edges
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 25, 15)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    # The plate body should dominate the crop; reject specks and full-frame blobs.
    if not (0.15 * h * w <= area <= 0.98 * h * w):
        return None

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return _order_corners(approx.reshape(4, 2))

    # Not a clean quad -> fall back to the minimum-area rotated rectangle,
    # which still corrects in-plane rotation/skew.
    return _order_corners(cv2.boxPoints(cv2.minAreaRect(c)))


def deskew_plate(crop):
    """Return a flattened (fronto-parallel) version of a plate crop.

    Safe no-op: returns the original crop unchanged if no reliable quad is found,
    so enabling this never makes a readable plate worse than the raw crop.
    """
    if crop is None or crop.size == 0:
        return crop

    quad = _segment_plate_quad(crop)
    if quad is None:
        return crop

    (tl, tr, br, bl) = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 8 or height < 8:
        return crop

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                    [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(crop, M, (width, height))


def main():
    parser = argparse.ArgumentParser(description="ANPR: YOLOv11 + ByteTrack + PaddleOCR")
    parser.add_argument("--source", default="0",
                        help="webcam index ('0'), video file, image, or folder. Default: '0'")
    parser.add_argument("--weights", default="s", choices=list("nsmlx"),
                        help="YOLOv11 model size: n(fastest)..x(most accurate). Default: s")
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument("--min-ocr-conf", type=float, default=0.5,
                        help="ignore OCR reads below this confidence")
    parser.add_argument("--deskew", action="store_true",
                        help="perspective-correct (flatten) each plate before OCR")
    parser.add_argument("--no-save", action="store_true", help="do not write an annotated output file")
    parser.add_argument("--no-show", action="store_true", help="do not open a preview window")
    args = parser.parse_args()

    weight_file = HF_WEIGHT_TEMPLATE.format(size=args.weights)
    print(f"Downloading YOLOv11 weights '{weight_file}' from '{HF_REPO_ID}'...")
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=weight_file)
    print(f"Model ready at: {model_path}")
    model = YOLO(model_path)

    print("Loading PaddleOCR (first run downloads its models)...")
    ocr = load_ocr()

    source = int(args.source) if str(args.source).isdigit() else args.source

    out_dir = os.path.join(_here, "anpr_output")
    os.makedirs(out_dir, exist_ok=True)
    writer = None

    # Per-track-ID vote tallies: {track_id: Counter({text: weighted_count})}
    votes = defaultdict(Counter)
    best_text = {}  # track_id -> current best string

    print("Running... press 'q' in the window (or Ctrl+C in the terminal) to stop.")
    # track() assigns persistent IDs across frames (ByteTrack).
    for result in model.track(source=source, conf=args.conf, stream=True,
                              persist=True, tracker="bytetrack.yaml", verbose=False):
        frame = result.orig_img.copy()

        H, W = frame.shape[:2]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tid = int(box.id[0]) if box.id is not None else -1

            if args.deskew:
                # Pad the tight detection box so classical segmentation has some
                # background context to locate the plate's edges, then flatten.
                px = int(0.12 * (x2 - x1))
                py = int(0.12 * (y2 - y1))
                cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
                cx2, cy2 = min(W, x2 + px), min(H, y2 + py)
                plate_img = deskew_plate(frame[cy1:cy2, cx1:cx2])
            else:
                plate_img = frame[y1:y2, x1:x2]

            text, conf = read_plate_text(ocr, plate_img)
            plate, valid = extract_indian_plate(text)
            if plate and conf >= args.min_ocr_conf:
                # Validated plates carry full vote weight; unvalidated reads vote
                # at a discount so a clean match can still overtake noisy guesses.
                weight = conf if valid else conf * 0.25
                if tid >= 0:
                    votes[tid][plate] += weight
                    best_text[tid] = votes[tid].most_common(1)[0][0]
                else:
                    best_text[tid] = plate

            label = best_text.get(tid, "")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if label:
                cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not args.no_save:
            if writer is None:
                h, w = frame.shape[:2]
                out_path = os.path.join(out_dir, "annotated.mp4")
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                print(f"Saving annotated output to: {out_path}")
            writer.write(frame)

        if not args.no_show:
            cv2.imshow("ANPR - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if best_text:
        print("\n=== Final plate readings (voted across frames) ===")
        for tid, txt in sorted(best_text.items()):
            print(f"  track {tid}: {txt}")
    print("Done.")


if __name__ == "__main__":
    main()
