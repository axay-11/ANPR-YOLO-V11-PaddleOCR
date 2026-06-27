"""
Probe available cameras on this machine and save a snapshot from each.

macOS enumerates cameras in an order you can't always predict (built-in webcam,
USB cameras, and Continuity Camera/iPhone all compete for low indices). This
script opens indices 0..N, grabs one frame from each working camera, and writes
it to ./camera_probe/cam_<index>.jpg so you can eyeball which index is which.

Usage:
    python list_cameras.py            # probe indices 0..5
    python list_cameras.py --max 8    # probe indices 0..8
"""

import argparse
import os

import cv2

_here = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="List / snapshot available cameras")
    parser.add_argument("--max", type=int, default=5, help="highest camera index to probe")
    args = parser.parse_args()

    out_dir = os.path.join(_here, "camera_probe")
    os.makedirs(out_dir, exist_ok=True)

    # AVFoundation is the native macOS capture backend; explicit is more reliable.
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY

    found = []
    for idx in range(args.max + 1):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            path = os.path.join(out_dir, f"cam_{idx}.jpg")
            cv2.imwrite(path, frame)
            found.append((idx, w, h, path))
            print(f"  index {idx}: OK  {w}x{h}  -> {path}")
        else:
            print(f"  index {idx}: opened but no frame (likely in use or blocked)")
        cap.release()

    print()
    if found:
        print(f"Found {len(found)} working camera(s). Open the JPGs in ./camera_probe/")
        print("to see which index is your connected camera, then run e.g.:")
        print(f"    python anpr_hf.py --source {found[-1][0]}")
    else:
        print("No cameras returned frames. Check System Settings -> Privacy &")
        print("Security -> Camera and allow access for your terminal app.")


if __name__ == "__main__":
    main()
