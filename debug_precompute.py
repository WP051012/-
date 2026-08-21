"""Test precompute on single site_B file."""
from pathlib import Path
import numpy as np
import re
from collections import defaultdict

IMG_W, IMG_H = 3840, 2160

SITE_B_ZONES = {
    "crosswalk": np.array([[2547,582],[2673,1533],[3627,1311],[3189,549],[2547,579]]),
    "lane_a":    np.array([[2148,630],[2112,798],[1128,837],[1638,561],[2145,627]]),
}

def point_in_polygon_vec(x, y, poly):
    n = len(poly); inside = False; j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-8) + xi):
            inside = not inside
        j = i
    return inside

labels_dir = Path("/root/red-light-prediction/labels")
site_b_files = sorted([f for f in labels_dir.glob("*20260123*")])
if not site_b_files:
    # Try different pattern
    site_b_files = sorted([f for f in labels_dir.glob("*.txt") if "20260123" in f.name])

print(f"Found {len(site_b_files)} site_B files")

if site_b_files:
    f = site_b_files[0]
    print(f"Processing: {f.name}")
    frames_data = defaultdict(list)
    current_frame = None

    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith("### Frame:"):
                m = re.search(r"_(\d+)\.txt", line)
                current_frame = int(m.group(1)) if m else current_frame
                continue
            if current_frame is None: continue
            parts = line.split()
            if len(parts) < 6: continue
            cls_id = int(parts[0])
            xc = float(parts[1]) * IMG_W
            yc = float(parts[2]) * IMG_H
            frames_data[current_frame].append((xc, yc, cls_id))

    print(f"Total frames: {len(frames_data)}")

    frames_with_objects = 0
    total_zone_objs = 0
    for fid, objs in sorted(frames_data.items()):
        in_zone = sum(1 for xc, yc, _ in objs if any(
            point_in_polygon_vec(xc, yc, poly) for poly in SITE_B_ZONES.values()))
        total_zone_objs += in_zone
        if in_zone > 0:
            frames_with_objects += 1

    print(f"Frames with in-zone objects: {frames_with_objects}/{len(frames_data)}")
    print(f"Total in-zone objects: {total_zone_objs}")
    print(f"Would produce output: {frames_with_objects > 0}")
