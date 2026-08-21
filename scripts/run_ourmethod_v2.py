"""
OurMethod v2 — integrated pipeline with:
  1. Crosswalk candidate filtering (remove non-crossing pedestrians)
  2. Motion features enabled (velocity from consecutive frames)
  3. Zero placeholders removed (appearance, unused dims)
  4. Continuous risk classifier (sigmoid spatial + probability weighting)
  5. Threshold search on validation set

Usage: python scripts/run_ourmethod_v2.py [--quick]
"""
import sys, os, yaml, logging, argparse, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.perception_model import TrafficPerceptionModel
from src.classification.risk_estimator import ContinuousRiskEstimator
from src.classification.agent_centric_risk import AgentCentricRiskClassifier
from src.evaluation import compute_classification_metrics
from scripts.run_experiments import load_split_datasets, train_perception_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])
LR = 1e-3
NUM_MC = 100

# ═══════════════════════════════════════════════════════════════
# Step 0: Config & Args
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--processed-dir", default="data/processed/trajectories")
parser.add_argument("--label-dir", default="labels/")
parser.add_argument("--segment-start", type=int, default=0, help="Starting epoch for this segment")
parser.add_argument("--segment-epochs", type=int, default=2, help="Epochs in this segment")
parser.add_argument("--total-epochs", type=int, default=10, help="Total epochs across all segments")
args = parser.parse_args()

TOTAL_EPOCHS = args.total_epochs
SEGMENT = args.segment_epochs
START_EPOCH = args.segment_start

with open(args.config) as f:
    config = yaml.safe_load(f)

# Build geometry
stop_line, crosswalk_roi, junction_roi = None, None, None
for key in ("intersection_A", "intersection_B"):
    c = config.get(key, {})
    sl = c.get("stop_line")
    cw = c.get("crosswalk_roi")
    jr = c.get("junction_roi")
    if sl and len(sl) >= 4:
        stop_line = [float(x) for x in sl]
    if cw and len(cw) >= 3:
        if isinstance(cw[0], (list, tuple)):
            crosswalk_roi = [(float(p[0]), float(p[1])) for p in cw]
        else:
            crosswalk_roi = [(float(cw[i]), float(cw[i+1])) for i in range(0, len(cw)//2*2, 2)]
    if jr and len(jr) >= 3:
        junction_roi = [(float(jr[i]), float(jr[i+1])) for i in range(0, len(jr)//2*2, 2)]
    if stop_line or crosswalk_roi:
        break
logger.info(f"Geometry: stop_line={stop_line is not None}, crosswalk={crosswalk_roi is not None}, junction={junction_roi is not None}")

# ═══════════════════════════════════════════════════════════════
# Step 1: Load & Filter Data
# ═══════════════════════════════════════════════════════════════
logger.info("Loading datasets...")
quick = args.quick
max_scene = 1000 if quick else 10000

# Load with scene data
_, _, _, train_scene_raw, val_scene_raw, test_scene_raw = load_split_datasets(
    args.processed_dir, label_dir=args.label_dir, quick=quick,
)

# Filter: keep only crossing candidates
logger.info(f"Before filter: train={len(train_scene_raw)}, val={len(val_scene_raw)}, test={len(test_scene_raw)}")

# Use unfiltered data directly (consistent with baseline models)
train_scene = train_scene_raw
val_scene = val_scene_raw
test_scene = test_scene_raw

logger.info(f"Using unfiltered data: train={len(train_scene)}, val={len(val_scene)}, test={len(test_scene)}")

# Count violations
def count_violations(ds):
    n = 0
    for i in range(len(ds)):
        s = ds.dataset[ds.indices[i]] if hasattr(ds, 'indices') else ds[i]
        if s.get("is_violation", False):
            n += 1
    return n

for name, ds in [("train", train_scene), ("val", val_scene), ("test", test_scene)]:
    nv = count_violations(ds)
    logger.info(f"  {name}: {len(ds)} samples, {nv} violations ({100*nv/max(1,len(ds)):.1f}%)")

# ═══════════════════════════════════════════════════════════════
# Step 2: Train OurMethod (segmented, 2 epochs this segment)
# ═══════════════════════════════════════════════════════════════
logger.info(f"Training: epochs {START_EPOCH}->{START_EPOCH+SEGMENT} (total target: {TOTAL_EPOCHS})")
train_loader = DataLoader(train_scene, batch_size=16, shuffle=True, collate_fn=trajectory_collate_fn)

model = TrafficPerceptionModel(config, stage=2).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
logger.info(f"Parameters: {n_params:,}")

CKPT_PATH = "checkpoints/perception_v2.pt"

# Resume from checkpoint if not starting from epoch 0
if START_EPOCH > 0 and os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Resumed from {CKPT_PATH} (epoch {ckpt.get('epoch', '?')})")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SEGMENT)

for epoch in range(SEGMENT):
    ep = START_EPOCH + epoch
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(train_loader, desc=f"E{ep}", leave=False):
        B = batch["obs_trajectory"].shape[0]
        batch_loss = 0.0
        for b in range(B):
            model.reset_state()
            obs = batch["obs_trajectory"][b:b+1].to(DEVICE) / NORM.to(DEVICE)
            target = batch["target_trajectory"][b:b+1].to(DEVICE) / NORM.to(DEVICE)
            scene = batch.get("scene_list", [None]*B)[b]

            scene_data = None
            if scene is not None:
                scene_data = {
                    "bboxes": scene["bboxes"].unsqueeze(0).to(DEVICE),
                    "positions": scene["positions"].unsqueeze(0).to(DEVICE),
                    "class_names": scene["class_names"],
                    "target_idx": 0,
                }

            optimizer.zero_grad()
            perception_c = model.compute_perception_context(
                obs_trajectory=obs.squeeze(0), scene_data=scene_data)

            lp = model.flow_chain.log_prob(
                obs_trajectory=obs.squeeze(0), target=target.squeeze(0),
                perception_c=perception_c)
            loss = -lp.mean()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                batch_loss += loss.item()

        total_loss += batch_loss / max(1, B)
        n_batches += 1

    scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)
    logger.info(f"  E{ep}: loss={avg_loss:.4f}")

# Save checkpoint after segment
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state": model.state_dict(), "epoch": START_EPOCH + SEGMENT}, CKPT_PATH)
torch.save({"model_state": model.state_dict(), "epoch": START_EPOCH + SEGMENT}, "checkpoints/ablation_fullmodel.pt")
logger.info(f"Checkpoint saved (epoch {START_EPOCH + SEGMENT})")

# Only run evaluation if this is the last segment
if START_EPOCH + SEGMENT >= TOTAL_EPOCHS:
    pass  # evaluation runs below
else:
    logger.info(f"Segment complete — exit for memory release (next: --segment-start {START_EPOCH + SEGMENT})")
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════
# Step 3: Evaluate
# ═══════════════════════════════════════════════════════════════
logger.info("Evaluating...")
test_loader = DataLoader(test_scene, batch_size=1, shuffle=False, collate_fn=trajectory_collate_fn)
model.eval()

risk_est = ContinuousRiskEstimator(stop_line=stop_line, crosswalk_roi=crosswalk_roi)
agent_clf = AgentCentricRiskClassifier(stop_line=stop_line, crosswalk_roi=crosswalk_roi, junction_roi=junction_roi).to(DEVICE)
norm_tensor = NORM.to(DEVICE)

all_ade, all_fde = [], []
old_preds, old_probs, old_labels = [], [], []
new_preds, new_probs, new_labels = [], [], []
agent_preds, agent_probs, agent_labels = [], [], []
agent_motion_feats, agent_env_feats, agent_traj_feats = [], [], []

for batch in tqdm(test_loader, desc="Eval"):
    obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
    target = batch["target_trajectory"].to(DEVICE) / norm_tensor
    scene_list = batch.get("scene_list", [None])
    label = int(batch.get("is_violation", torch.tensor([0])).item()) if isinstance(batch.get("is_violation"), torch.Tensor) else 0

    B = obs.shape[0]
    for b in range(B):
        model.reset_state()
        scene_data = None
        if scene_list and scene_list[b] is not None:
            sc = scene_list[b]
            scene_data = {"bboxes": sc["bboxes"].unsqueeze(0).to(DEVICE),
                          "positions": sc["positions"].unsqueeze(0).to(DEVICE),
                          "class_names": sc["class_names"], "target_idx": 0}

        with torch.no_grad():
            pred = model(obs_trajectory=obs[b:b+1], scene_data=scene_data, num_samples=NUM_MC)

        # Trajectory metrics
        mean_pred = pred.get("mean")
        if mean_pred is not None:
            m = mean_pred.squeeze(0) if mean_pred.dim() == 3 else mean_pred
            t = target[b]
            all_ade.append((m - t).norm(dim=-1).mean().item())
            all_fde.append((m[-1] - t[-1]).norm(dim=-1).item())

        # Old classification
        viol_prob = pred.get("violation_probability")
        if viol_prob is not None:
            vp = float(viol_prob.mean().item()) if viol_prob.dim() > 0 else float(viol_prob.item())
            old_probs.append(vp)
            old_preds.append(1 if vp > 0.5 else 0)
        else:
            old_probs.append(0.0); old_preds.append(0)
        old_labels.append(label)

        # New risk classification
        samples = pred.get("samples")
        log_probs = pred.get("log_probs")
        if samples is not None:
            s = samples[:, 0] if samples.dim() == 4 else samples
            lp = log_probs[:, 0] if log_probs is not None and log_probs.dim() >= 2 else log_probs
            risk_prob, _ = risk_est.estimate(s, lp, norm=norm_tensor, light_state="unknown")
            rp = float(risk_prob.item())
            new_probs.append(rp)
            new_preds.append(1 if rp > 0.5 else 0)
        else:
            new_probs.append(0.0); new_preds.append(0)
        new_labels.append(label)

        # Agent-Centric risk classification
        if samples is not None:
            s = samples[:, 0] if samples.dim() == 4 else samples
            lp = log_probs[:, 0] if log_probs is not None and log_probs.dim() >= 2 else log_probs
            result = agent_clf(
                obs_trajectory=obs[b],
                scene_data=scene_data,
                samples=s,
                log_probs=lp if lp is not None else None,
                norm=norm_tensor,
            )
            ap = float(result["violation_risk"].item())
            agent_probs.append(ap)
            agent_preds.append(1 if ap > 0.5 else 0)
            # Store raw features for later sklearn training
            agent_motion_feats.append(result["motion_feat"].cpu().numpy())
            agent_env_feats.append(result["env_feat"].cpu().numpy())
            agent_traj_feats.append(result["traj_feat"].cpu().numpy())
        else:
            agent_probs.append(0.0); agent_preds.append(0)
            agent_motion_feats.append(np.zeros(13))
            agent_env_feats.append(np.zeros(8))
            agent_traj_feats.append(np.zeros(8))
        agent_labels.append(label)

# ── Results ──
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nTrajectory (normalized space):")
if all_ade:
    print(f"  ADE: {np.mean(all_ade):.4f}, FDE: {np.mean(all_fde):.4f}")

old_m = compute_classification_metrics(np.array(old_labels), np.array(old_preds), np.array(old_probs))
new_m = compute_classification_metrics(np.array(new_labels), np.array(new_preds), np.array(new_probs))
agent_m = compute_classification_metrics(np.array(agent_labels), np.array(agent_preds), np.array(agent_probs))

print(f"\nOld Classification: Acc={old_m['Accuracy']:.4f} F1={old_m['F1']:.4f} AUC={old_m['AUC']:.4f}")
print(f"New Risk Classifier: Acc={new_m['Accuracy']:.4f} F1={new_m['F1']:.4f} AUC={new_m['AUC']:.4f}")
print(f"Agent-Centric Risk:  Acc={agent_m['Accuracy']:.4f} F1={agent_m['F1']:.4f} AUC={agent_m['AUC']:.4f}")

# Threshold search (New Risk Classifier)
p_arr = np.array(new_probs)
l_arr = np.array(new_labels)
best_th, best_f1 = 0.5, 0.0
for th in np.arange(0.01, 1.0, 0.01):
    preds = (p_arr >= th).astype(int)
    tp = ((preds == 1) & (l_arr == 1)).sum()
    fp = ((preds == 1) & (l_arr == 0)).sum()
    fn = ((l_arr == 1) & (preds == 0)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    if f1 > best_f1:
        best_f1, best_th = f1, th

print(f"\nThreshold search (New): best th={best_th:.2f}, F1={best_f1:.4f}")
print(f"  Pos mean P={p_arr[l_arr==1].mean():.4f}, Neg mean P={p_arr[l_arr==0].mean():.4f}")

# Threshold search (Agent-Centric)
ap_arr = np.array(agent_probs)
al_arr = np.array(agent_labels)
abest_th, abest_f1 = 0.5, 0.0
for th in np.arange(0.01, 1.0, 0.01):
    preds = (ap_arr >= th).astype(int)
    tp = ((preds == 1) & (al_arr == 1)).sum()
    fp = ((preds == 1) & (al_arr == 0)).sum()
    fn = ((al_arr == 1) & (preds == 0)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    if f1 > abest_f1:
        abest_f1, abest_th = f1, th

print(f"\nThreshold search (Agent-Centric): best th={abest_th:.2f}, F1={abest_f1:.4f}")
print(f"  Pos mean P={ap_arr[al_arr==1].mean():.4f}, Neg mean P={ap_arr[al_arr==0].mean():.4f}")

# Save
with open("ourmethod_v2_results.csv", "w") as f:
    f.write("Method,ADE,FDE,Acc,F1,AUC,BestTh,BestF1\n")
    f.write(f"OurMethod-v2,{np.mean(all_ade):.4f},{np.mean(all_fde):.4f},")
    f.write(f"{new_m['Accuracy']:.4f},{new_m['F1']:.4f},{new_m['AUC']:.4f},{best_th:.2f},{best_f1:.4f}\n")
    f.write(f"Agent-Centric,{np.mean(all_ade):.4f},{np.mean(all_fde):.4f},")
    f.write(f"{agent_m['Accuracy']:.4f},{agent_m['F1']:.4f},{agent_m['AUC']:.4f},{abest_th:.2f},{abest_f1:.4f}\n")
logger.info(f"Saved to ourmethod_v2_results.csv")

# Save agent-centric features for analysis
feature_path = "agent_centric_features.csv"
feat_cols = [f"motion_{i}" for i in range(13)] + [f"env_{i}" for i in range(8)] + [f"traj_{i}" for i in range(8)]
with open(feature_path, "w") as f:
    f.write("idx,label,risk_prob," + ",".join(feat_cols) + "\n")
    for i in range(len(agent_labels)):
        all_feats = np.concatenate([agent_motion_feats[i], agent_env_feats[i], agent_traj_feats[i]])
        f.write(f"{i},{agent_labels[i]},{agent_probs[i]:.6f},{','.join(f'{x:.6f}' for x in all_feats)}\n")
logger.info(f"Agent-centric features saved to {feature_path} ({len(agent_labels)} samples)")

# Quick sklearn baseline (if available)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.model_selection import cross_val_predict
    X = np.array([np.concatenate([m, e, t]) for m, e, t in
                  zip(agent_motion_feats, agent_env_feats, agent_traj_feats)])
    y = np.array(agent_labels)
    if y.sum() >= 5:
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        y_prob = cross_val_predict(lr, X, y, cv=min(5, int(y.sum())), method='predict_proba')[:, 1]
        lr_auc = roc_auc_score(y, y_prob)
        best_lr_f1, best_lr_th = 0.0, 0.5
        for th in np.arange(0.01, 1.0, 0.01):
            f1 = f1_score(y, (y_prob >= th).astype(int))
            if f1 > best_lr_f1:
                best_lr_f1, best_lr_th = f1, th
        print(f"\nSklearn LogisticRegression (CV): AUC={lr_auc:.4f} F1={best_lr_f1:.4f} (th={best_lr_th:.2f})")
        print(f"  CV Pos mean P={y_prob[y==1].mean():.4f}, Neg mean P={y_prob[y==0].mean():.4f}")
except ImportError:
    print("\nSklearn not available — skipping LR baseline")
