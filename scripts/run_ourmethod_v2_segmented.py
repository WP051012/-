"""Wrapper: run OurMethod v2 in 2-epoch segments to stay within 80GB memory."""
import os, sys

BASE = "/root/miniconda3/bin/python"
SCRIPT = "scripts/run_ourmethod_v2.py"
TOTAL = 10
SEG = 2

print("=" * 60)
print(f"OurMethod v2 — segmented: {TOTAL//SEG} x {SEG} epochs")
print("=" * 60)

for start in range(0, TOTAL, SEG):
    seg_num = start // SEG + 1
    print(f"\n>>> Segment {seg_num}: epochs {start}->{start+SEG}")
    cmd = f"{BASE} {SCRIPT} --segment-start {start} --segment-epochs {SEG} --total-epochs {TOTAL}"
    rc = os.system(cmd)
    rc_val = rc >> 8 if rc > 255 else rc
    print(f">>> Segment {seg_num} exit: {rc_val}")
    if rc_val != 0:
        print("WARNING: non-zero exit, but continuing...")

print("\nDone!")
