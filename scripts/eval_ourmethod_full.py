"""Evaluate trained OurMethod checkpoint: trajectory + classification (old & new risk-based)."""
import sys, os, logging, numpy as np, torch, yaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.perception_model import TrafficPerceptionModel
from src.evaluation import compute_classification_metrics
from src.classification.risk_estimator import ContinuousRiskEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])
CKPT = "checkpoints/ablation_fullmodel.pt"
CONFIG_PATH = "configs/default.yaml"

# ── Load config ──
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# ── Load data ──
logger.info("Loading test set...")
ds_test = TrajectoryDataset(
    data_dir="data/processed/trajectories/", label_dir="labels/",
    obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
    target_classes=["pedestrian"], mode="with_scene", max_scene_samples=2000,
)
from scripts.run_experiments import load_split_datasets
_, _, _, _, _, test_scene = load_split_datasets("data/processed/trajectories/", quick=False)
test_loader = DataLoader(test_scene, batch_size=1, shuffle=False, collate_fn=trajectory_collate_fn)
logger.info(f"Test samples: {len(test_scene)}")

# ── Load model ──
logger.info(f"Loading checkpoint: {CKPT}")
model = TrafficPerceptionModel(config, stage=2).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE)
model_state = model.state_dict()
loaded = {k: v for k, v in ckpt.get("model_state", ckpt).items()
          if k in model_state and v.shape == model_state[k].shape}
model.load_state_dict(loaded, strict=False)
logger.info(f"Loaded {len(loaded)}/{len(model_state)} params")
model.eval()

# ── Build geometric components from config ──
def build_geometry(cfg):
    stop_line, crosswalk = None, None
    for key in ("intersection_A", "intersection_B"):
        c = cfg.get(key, {})
        sl = c.get("stop_line", None)
        cw = c.get("crosswalk_roi", None)
        if sl and len(sl) >= 4:
            stop_line = [float(x) for x in sl]
        if cw and len(cw) >= 3:
            if isinstance(cw[0], (list, tuple)):
                crosswalk = [(float(p[0]), float(p[1])) for p in cw]
            else:
                crosswalk = [(float(cw[i]), float(cw[i+1])) for i in range(0, len(cw)//2*2, 2)]
        if stop_line or crosswalk:
            break
    return stop_line, crosswalk

stop_line, crosswalk = build_geometry(config)
logger.info(f"Geometry: stop_line={stop_line is not None}, crosswalk={crosswalk is not None}")

# ── Risk estimator (new) ──
risk_est = ContinuousRiskEstimator(stop_line=stop_line, crosswalk_roi=crosswalk)

# ===================================================================
# Evaluation
# ===================================================================

all_ade, all_fde = [], []
old_preds, old_probs, old_labels = [], [], []
new_preds, new_probs, new_labels = [], [], []

norm_tensor = NORM.to(DEVICE)
num_samples = 100

for batch in tqdm(test_loader, desc="Eval"):
    obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
    target = batch["target_trajectory"].to(DEVICE) / norm_tensor
    scene_list = batch.get("scene_list", [None])
    label_val = batch.get("is_violation")
    if isinstance(label_val, torch.Tensor):
        label = int(label_val.item()) if label_val.numel() > 0 else 0
    else:
        label = 0

    B = obs.shape[0]
    for b in range(B):
        model.reset_state()
        scene_data = scene_list[b] if isinstance(scene_list, list) and b < len(scene_list) else None

        with torch.no_grad():
            pred = model(obs_trajectory=obs[b:b+1], scene_data=scene_data, num_samples=num_samples)

        # -- Trajectory metrics --
        mean_pred = pred.get("mean")
        if mean_pred is not None:
            if mean_pred.dim() == 3:
                mean_pred = mean_pred.squeeze(0)
            tgt = target[b] if target.dim() == 3 else target
            ade = (mean_pred - tgt).norm(dim=-1).mean().item()
            fde = (mean_pred[-1] - tgt[-1]).norm(dim=-1).item()
            all_ade.append(ade)
            all_fde.append(fde)

        samples = pred.get("samples")  # (N, 1, T, 2) or (N, T, 2)
        log_probs = pred.get("log_probs")

        # -- Old classification (from model built-in) --
        viol_prob = pred.get("violation_probability")
        if viol_prob is not None:
            if viol_prob.dim() > 0:
                viol_prob = viol_prob.mean()
            old_probs.append(float(viol_prob.item()))
            old_preds.append(1 if float(viol_prob.item()) > 0.5 else 0)
        else:
            old_probs.append(0.0)
            old_preds.append(0)
        old_labels.append(label)

        # -- New risk-based classification --
        if samples is not None:
            if samples.dim() == 4:
                s = samples[:, 0]  # (N, T, 2)
            else:
                s = samples
            lp = log_probs[:, 0] if log_probs is not None and log_probs.dim() >= 2 else log_probs
            risk_prob, risk_stats = risk_est.estimate(s, lp, norm=norm_tensor, light_state="unknown")
            rp = float(risk_prob.item())
            new_probs.append(rp)
            new_preds.append(1 if rp > 0.5 else 0)
        else:
            new_probs.append(0.0)
            new_preds.append(0)
        new_labels.append(label)

# ---- Results ----
print("\n" + "=" * 60)
print("OurMethod Trajectory Metrics (Stage 2)")
print("=" * 60)
if all_ade:
    print(f"  ADE: {np.mean(all_ade):.4f} px  =  {np.mean(all_ade)/50:.4f} m")
    print(f"  FDE: {np.mean(all_fde):.4f} px  =  {np.mean(all_fde)/50:.4f} m")
else:
    print("  No trajectory metrics (mean_pred missing)")

print("\n" + "=" * 60)
print("Old Classification (hard geometric check)")
print("=" * 60)
old_m = compute_classification_metrics(np.array(old_labels), np.array(old_preds), np.array(old_probs))
for k, v in old_m.items():
    print(f"  {k}: {v:.4f}")

print("\n" + "=" * 60)
print("New Classification (continuous risk)")
print("=" * 60)
new_m = compute_classification_metrics(np.array(new_labels), np.array(new_preds), np.array(new_probs))
for k, v in new_m.items():
    print(f"  {k}: {v:.4f}")

# Risk distribution
print("\n" + "=" * 60)
print("Risk Distribution (new method)")
print("=" * 60)
p_arr = np.array(new_probs)
l_arr = np.array(new_labels)
pos_p = p_arr[l_arr == 1]
neg_p = p_arr[l_arr == 0]
print(f"  Pos samples: {len(pos_p)}, mean P={np.mean(pos_p):.4f}, max={np.max(pos_p):.4f}")
print(f"  Neg samples: {len(neg_p)}, mean P={np.mean(neg_p):.4f}, max={np.max(neg_p):.4f}")
print(f"  P=0 fraction (pos): {np.mean(pos_p < 1e-6):.3f}")
print(f"  P=0 fraction (neg): {np.mean(neg_p < 1e-6):.3f}")

# Write CSV
with open("ourmethod_eval_results.csv", "w") as f:
    f.write("Method,ADE,FDE,NLL\n")
    f.write(f"OurMethod-Traj,{np.mean(all_ade):.4f},{np.mean(all_fde):.4f},N/A\n")
    f.write("Method,Accuracy,Precision,Recall,F1,AUC\n")
    f.write(f"OurMethod-Old,{old_m['Accuracy']:.4f},{old_m['Precision']:.4f},{old_m['Recall']:.4f},{old_m['F1']:.4f},{old_m['AUC']:.4f}\n")
    f.write(f"OurMethod-Risk,{new_m['Accuracy']:.4f},{new_m['Precision']:.4f},{new_m['Recall']:.4f},{new_m['F1']:.4f},{new_m['AUC']:.4f}\n")
logger.info("Results saved to ourmethod_eval_results.csv")
