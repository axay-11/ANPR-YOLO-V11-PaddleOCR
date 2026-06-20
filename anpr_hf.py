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
    python anpr_hf.py --no-save            # show only, don't write output
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
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# YOLOv11 license-plate model on the Hugging Face Hub (sizes: n, s, m, l, x).
HF_REPO_ID = os.getenv("HF_REPO_ID", "morsetechlab/yolov11-license-plate-detection")
HF_WEIGHT_TEMPLATE = "license-plate-finetune-v1{size}.pt"

# Keep only plausible plate characters when voting/validating.
PLATE_RE = re.compile(r"[^A-Z0-9]")


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


def main():
    parser = argparse.ArgumentParser(description="ANPR: YOLOv11 + ByteTrack + PaddleOCR")
    parser.add_argument("--source", default="0",
                        help="webcam index ('0'), video file, image, or folder. Default: '0'")
    parser.add_argument("--weights", default="s", choices=list("nsmlx"),
                        help="YOLOv11 model size: n(fastest)..x(most accurate). Default: s")
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument("--min-ocr-conf", type=float, default=0.5,
                        help="ignore OCR reads below this confidence")
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

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tid = int(box.id[0]) if box.id is not None else -1

            text, conf = read_plate_text(ocr, frame[y1:y2, x1:x2])
            if text and conf >= args.min_ocr_conf:
                if tid >= 0:
                    votes[tid][text] += conf            # weight vote by confidence
                    best_text[tid] = votes[tid].most_common(1)[0][0]
                else:
                    best_text[tid] = text

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
