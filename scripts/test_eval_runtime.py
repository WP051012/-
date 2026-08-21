"""
Runtime test for the FIXED per-domain FOMAML eval.

Confirms the eval is back to ~baseline speed (not the old ~16 h per-sample
inner loop) by timing the two phases separately:

    Phase A — per-domain inner-loop adaptation (ONE time)  → adapt_time
    Phase B — forward-only best-of-N sampling             → eval it/s

Then extrapolates the full D5 test-set time.

Usage:
    python scripts/test_eval_runtime.py \
        --checkpoint checkpoints/fomaml_v2/best_fomaml.pt \
        --n-timing 600 --num-mc 100 --full-n 128412
"""

import sys, os, time, json, argparse, logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts/ dir

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from eval_fomaml import (
    load_fomaml_model, split_by_domain, filter_candidates,
    parse_geometry, META_TEST, _compute_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test-runtime")

FAIL = False


def main():
    global FAIL
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--perception-ckpt", default="checkpoints/stage1_best.pt")
    ap.add_argument("--flowchain-ckpt", default="checkpoints/flowchain_domain_filtered.pt")
    ap.add_argument("--data-dir", default="data/processed/trajectories")
    ap.add_argument("--label-dir", default="labels/")
    ap.add_argument("--domain-map", default="data/domains/domain_labels_int.json")
    ap.add_argument("--annotations-dir", default="data/annotations")
    ap.add_argument("--num-mc", type=int, default=100)
    ap.add_argument("--max-samples", type=int, default=50000,
                    help="cap total dataset size (0 = full). Bigger → more D5 candidates.")
    ap.add_argument("--max-scene", type=int, default=20000)
    ap.add_argument("--n-timing", type=int, default=600,
                    help="number of test samples to time in Phase B")
    ap.add_argument("--full-n", type=int, default=128412,
                    help="known full D5 candidate count for extrapolation")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------- Model ----------
    with open(args.config) as f:
        config = yaml.safe_load(f)
    model, trainable_params, mod_net, domain_conditions, inner_cfg = \
        load_fomaml_model(config, args.perception_ckpt, args.flowchain_ckpt,
                          args.checkpoint, device)
    n_train = sum(p.numel() for p in trainable_params.values())
    logger.info(f"trainable={n_train:,}  inner_cfg={inner_cfg}")

    # ---------- Geometry + dataset + filter ----------
    junction_roi, crosswalk_roi, stop_line = parse_geometry(args.annotations_dir)
    with open(args.domain_map) as f:
        domain_label_map = json.load(f)

    ds = TrajectoryDataset(
        data_dir=args.data_dir, label_dir=args.label_dir,
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"], mode="with_scene",
        max_scene_samples=args.max_scene, max_samples=args.max_samples,
        domain_label_map=domain_label_map, force_scene=True,
        crossing_region=junction_roi,
    )
    logger.info(f"dataset: {len(ds)} samples")

    test_all = split_by_domain(ds, META_TEST, "test_all(D5)")
    test_idx = filter_candidates(ds, test_all, junction_roi, stop_line,
                                 crosswalk_roi, use_future_gt=True, name="test")
    K = min(args.n_timing, len(test_idx))
    if K == 0:
        logger.error("No D5 candidates — increase --max-samples / --max-scene")
        sys.exit(1)
    logger.info(f"timing on first {K} of {len(test_idx)} D5 candidates")
    idxs = test_idx[:K]

    # ============ Phase A: per-domain adaptation (ONE time) ============
    did = 5
    meta_state = {name: p.data.clone() for name, p in trainable_params.items()}

    cond = domain_conditions.get(did)
    if mod_net is not None and cond is not None:
        delta_flat = mod_net(cond.unsqueeze(0))
        mod_net.apply_delta(delta_flat, trainable_params, sign=+1)

    rng = np.random.RandomState(42)
    arr = np.array(idxs)
    rng.shuffle(arr)
    n_support = max(1, int(len(arr) * 0.7))
    support_idx = arr[:n_support].tolist()

    model.eval()
    loader = DataLoader(
        Subset(ds, support_idx),
        batch_size=min(inner_cfg["batch_size"], len(support_idx)),
        shuffle=True, collate_fn=trajectory_collate_fn, num_workers=0,
    )
    p_init = {name: p.data.clone() for name, p in trainable_params.items()}
    t0 = time.time()
    n_steps = 0
    for k, batch in enumerate(loader):
        if k >= inner_cfg["inner_steps"]:
            break
        obs = batch["obs_trajectory"].to(device)
        target = batch["target_trajectory"].to(device)
        loss = _compute_loss(model, obs, target,
                             inner_cfg["ade_weight"], inner_cfg["lambda_feat"])
        if torch.isfinite(loss):
            loss.backward()
            for name, p in trainable_params.items():
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
                    p.data -= inner_cfg["inner_lr"] * p.grad
                    p.grad = None
                    delta = p.data - p_init[name]
                    dn = delta.norm()
                    if dn > inner_cfg["max_delta_norm"]:
                        delta = delta * (inner_cfg["max_delta_norm"] / dn)
                        p.data = p_init[name] + delta
            n_steps += 1
    adapt_time = time.time() - t0
    logger.info(f"Phase A: {n_steps} support steps on {len(support_idx)} samples "
                f"→ {adapt_time:.2f}s")

    bad = sum(1 for p in trainable_params.values() if not torch.isfinite(p.data).all())
    logger.info(f"  [{('PASS' if bad == 0 else 'FAIL')}] adapted params finite (bad={bad})")
    if bad:
        FAIL = True

    # ============ Phase B: forward-only sampling ============
    model.eval()
    zero_cond = torch.zeros(1, model.condition_dim, device=device)
    t0 = time.time()
    ok, nonfinite = 0, 0
    with torch.no_grad():
        for idx in tqdm(idxs, desc="Phase B timing", leave=False):
            sample = ds[idx]
            obs = sample["obs_trajectory"].to(device).unsqueeze(0)
            pred = model.flow_chain(obs_trajectory=obs, perception_c=zero_cond,
                                    num_samples=args.num_mc)
            samples = pred.get("samples")
            if samples is None:
                continue
            ok += 1
            if not torch.isfinite(samples).all():
                nonfinite += 1
    eval_time = time.time() - t0
    it_s = K / eval_time if eval_time > 0 else float("inf")
    full_min = (args.full_n / it_s) / 60.0 if it_s > 0 else float("inf")

    logger.info(f"Phase B: {ok}/{K} samples sampled in {eval_time:.2f}s "
                f"→ {it_s:.2f} it/s")
    logger.info(f"  [{('PASS' if nonfinite == 0 else 'FAIL')}] samples finite (nonfinite={nonfinite})")
    if nonfinite:
        FAIL = True

    # restore
    for name, p in trainable_params.items():
        p.data.copy_(meta_state[name])

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("RUNTIME TEST — per-domain FOMAML eval (fixed protocol)")
    print("=" * 60)
    print(f"Phase A (adapt, once):      {adapt_time:.2f}s  ({n_steps} steps)")
    print(f"Phase B (sampling):         {it_s:.2f} it/s  (num_mc={args.num_mc})")
    print(f"Extrapolated full D5 eval:  ~{full_min:.1f} min  "
          f"({args.full_n:,} samples @ {it_s:.2f} it/s)")
    print(f"Old per-sample protocol:    ~16 h")
    print(f"Speedup:                    ~{16 * 60 / full_min:.0f}x")
    print("=" * 60)
    print(f"RESULT: {'FAIL' if FAIL else 'PASS'}")

    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
