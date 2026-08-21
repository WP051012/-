"""Diagnose why site_B objects are not captured by zones."""
import numpy as np
from pathlib import Path

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

def in_any_zone(x, y, zones):
    for poly in zones.values():
        if point_in_polygon_vec(x, y, poly):
            return True
    return False

# Find site_B files
labels_dir = Path("/root/red-light-prediction/labels")
site_b_files = sorted([f for f in labels_dir.glob("*.txt") if any(d in f.name for d in ("20260123","20260126","20260127"))])
print(f"Site B label files: {len(site_b_files)}")

if site_b_files:
    f = site_b_files[0]
    print(f"Checking: {f.name}")
    total = 0; in_zone = 0
    xs_all, ys_all = [], []
    xs_in, ys_in = [], []
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("###"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            xc = float(parts[1]) * IMG_W
            yc = float(parts[2]) * IMG_H
            total += 1
            if len(xs_all) < 200:
                xs_all.append(xc); ys_all.append(yc)
            if in_any_zone(xc, yc, SITE_B_ZONES):
                in_zone += 1
                if len(xs_in) < 50:
                    xs_in.append(xc); ys_in.append(yc)

    print(f"Total objects: {total}, In zone: {in_zone} ({100*in_zone/max(1,total):.1f}%)")
    print(f"ALL positions X: {min(xs_all):.0f}-{max(xs_all):.0f}, Y: {min(ys_all):.0f}-{max(ys_all):.0f}")
    if xs_in:
        print(f"IN-ZONE positions X: {min(xs_in):.0f}-{max(xs_in):.0f}, Y: {min(ys_in):.0f}-{max(ys_in):.0f}")

    # Print zones for comparison
    for name, poly in SITE_B_ZONES.items():
        print(f"Zone {name}: X {poly[:,0].min():.0f}-{poly[:,0].max():.0f}, Y {poly[:,1].min():.0f}-{poly[:,1].max():.0f}")
