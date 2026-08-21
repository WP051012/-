"""Threshold search for continuous risk classifier."""
import sys, numpy as np, torch, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.perception_model import TrafficPerceptionModel
from src.classification.risk_estimator import ContinuousRiskEstimator
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

# Load data
ds_test = TrajectoryDataset(
    data_dir="data/processed/trajectories/", label_dir="labels/",
    obs_len=8, pred_len=12, mode="with_scene", max_scene_samples=2000,
)
from scripts.run_experiments import load_split_datasets
_, _, _, _, _, test_scene = load_split_datasets("data/processed/trajectories/", quick=False)
test_loader = DataLoader(test_scene, batch_size=1, shuffle=False, collate_fn=trajectory_collate_fn)

# Load model
model = TrafficPerceptionModel(config, stage=2).to(DEVICE)
ckpt = torch.load("checkpoints/ablation_fullmodel.pt", map_location=DEVICE)
ms = model.state_dict()
loaded = {k: v for k, v in ckpt.get("model_state", ckpt).items()
          if k in ms and v.shape == ms[k].shape}
model.load_state_dict(loaded, strict=False)
model.eval()

# Build geometry
stop_line, crosswalk_roi = None, None
for key in ("intersection_A", "intersection_B"):
    c = config.get(key, {})
    sl = c.get("stop_line")
    cw = c.get("crosswalk_roi")
    if sl and len(sl) >= 4:
        stop_line = [float(x) for x in sl]
    if cw and len(cw) >= 3:
        if isinstance(cw[0], (list, tuple)):
            crosswalk_roi = [(float(p[0]), float(p[1])) for p in cw]
        else:
            crosswalk_roi = [(float(cw[i]), float(cw[i+1]))
                           for i in range(0, len(cw)//2*2, 2)]
    if stop_line or crosswalk_roi:
        break

est = ContinuousRiskEstimator(stop_line=stop_line, crosswalk_roi=crosswalk_roi)
norm_tensor = NORM.to(DEVICE)

all_probs = []
all_labels = []

for batch in tqdm(test_loader, desc="Eval"):
    obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
    scene_list = batch.get("scene_list", [None])
    label_val = batch.get("is_violation")
    label = int(label_val.item()) if isinstance(label_val, torch.Tensor) and label_val.numel() > 0 else 0

    B = obs.shape[0]
    for b in range(B):
        model.reset_state()
        scene_data = scene_list[b] if isinstance(scene_list, list) and b < len(scene_list) else None
        with torch.no_grad():
            pred = model(obs_trajectory=obs[b:b+1], scene_data=scene_data, num_samples=100)
        samples = pred.get("samples")
        log_probs = pred.get("log_probs")
        if samples is not None:
            s = samples[:, 0] if samples.dim() == 4 else samples
            lp = log_probs[:, 0] if log_probs is not None and log_probs.dim() >= 2 else log_probs
            risk_prob, _ = est.estimate(s, lp, norm=norm_tensor, light_state="unknown")
            all_probs.append(float(risk_prob.item()))
        else:
            all_probs.append(0.0)
        all_labels.append(label)

probs = np.array(all_probs)
labels = np.array(all_labels)

# Threshold search
best_th, best_f1 = 0.5, 0.0
results = []
for th in np.arange(0.01, 1.0, 0.01):
    preds = (probs >= th).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((labels == 1) & (preds == 0)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    results.append((th, f1, prec, rec, tp, fp, fn))
    if f1 > best_f1:
        best_f1, best_th = f1, th

print(f"Best threshold: {best_th:.2f}, F1: {best_f1:.4f}")

results.sort(key=lambda x: -x[1])
print("Top 5 thresholds:")
for th, f1, prec, rec, tp, fp, fn in results[:5]:
    print(f"  th={th:.2f}: F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} TP={int(tp)} FP={int(fp)} FN={int(fn)}")

auc = roc_auc_score(labels, probs)
pos_mask = labels == 1
print(f"\nAUC: {auc:.4f}")
print(f"Pos({pos_mask.sum()}): mean={probs[pos_mask].mean():.4f} max={probs[pos_mask].max():.4f}")
print(f"Neg({(~pos_mask).sum()}): mean={probs[~pos_mask].mean():.4f} max={probs[~pos_mask].max():.4f}")
