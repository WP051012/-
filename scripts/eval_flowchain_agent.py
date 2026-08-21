"""
Evaluate Agent-Centric Risk Classifier with FlowChain trajectory predictor.

FlowChain has better trajectory prediction (ADE=20px) than the overwritten
epoch-2 OurMethod, so its trajectory features should be more informative.

Usage: python scripts/eval_flowchain_agent.py [--quick]
"""
import sys, os, logging, argparse, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.baselines.baseline_models import FlowChainBase
from src.classification.risk_estimator import ContinuousRiskEstimator, _signed_dist_to_line
from src.classification.agent_centric_risk import AgentCentricRiskClassifier
from src.evaluation import compute_classification_metrics
from scripts.run_experiments import load_split_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--processed-dir", default="data/processed/trajectories")
parser.add_argument("--label-dir", default="labels/")
args = parser.parse_args()

import yaml
with open(args.config) as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])
NUM_MC = 100

# Build geometry from config
stop_line = None
crosswalk_roi = None
junction_roi = None
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
# Load FlowChain model
# ═══════════════════════════════════════════════════════════════
logger.info("Loading FlowChain model...")
flowchain = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)
ckpt = torch.load("checkpoints/flowchain_best.pt", map_location=DEVICE)
flowchain.load_state_dict(ckpt)
flowchain.eval()
logger.info(f"FlowChain loaded ({sum(p.numel() for p in flowchain.parameters()):,} params)")

# ═══════════════════════════════════════════════════════════════
# Load test dataset (with scene data)
# ═══════════════════════════════════════════════════════════════
logger.info("Loading datasets...")
_, _, _, _, _, test_scene = load_split_datasets(
    args.processed_dir, label_dir=args.label_dir, quick=args.quick,
)
logger.info(f"Test set: {len(test_scene)} samples")

# Count violations
nv = 0
for i in range(len(test_scene)):
    s = test_scene.dataset[test_scene.indices[i]] if hasattr(test_scene, 'indices') else test_scene[i]
    if s.get("is_violation", False):
        nv += 1
logger.info(f"  Violations: {nv} ({100*nv/max(1,len(test_scene)):.1f}%)")

# ═══════════════════════════════════════════════════════════════
# Classifiers
# ═══════════════════════════════════════════════════════════════
risk_est = ContinuousRiskEstimator(stop_line=stop_line, crosswalk_roi=crosswalk_roi)
agent_clf = AgentCentricRiskClassifier(stop_line=stop_line, crosswalk_roi=crosswalk_roi, junction_roi=junction_roi).to(DEVICE)
norm_tensor = NORM.to(DEVICE)

# ═══════════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════════
logger.info("Evaluating...")
test_loader = DataLoader(test_scene, batch_size=1, shuffle=False, collate_fn=trajectory_collate_fn)

all_ade, all_fde = [], []
risk_probs, risk_preds, risk_labels = [], [], []
agent_probs, agent_preds, agent_labels = [], [], []
agent_motion, agent_env, agent_traj = [], [], []
geo_feat = []  # raw geometric features (from stop-line distance)

for batch in tqdm(test_loader, desc="Eval"):
    obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
    target = batch["target_trajectory"].to(DEVICE) / norm_tensor
    scene_list = batch.get("scene_list", [None])
    label = int(batch.get("is_violation", torch.tensor([0])).item()) if isinstance(batch.get("is_violation"), torch.Tensor) else 0

    B = obs.shape[0]
    for b in range(B):
        # Build scene_data
        scene_data = None
        if scene_list and scene_list[b] is not None:
            sc = scene_list[b]
            scene_data = {
                "bboxes": sc["bboxes"].unsqueeze(0).to(DEVICE),
                "positions": sc["positions"].unsqueeze(0).to(DEVICE),
                "class_names": sc["class_names"],
                "target_idx": 0,
            }

        with torch.no_grad():
            pred = flowchain(obs_trajectory=obs[b:b+1], num_samples=NUM_MC)

        # Trajectory metrics (pixel space, Best-of-N)
        samples = pred.get("samples")   # (N, B, 12, 2)
        if samples is not None and "mean" in pred:
            samples_px = samples * norm_tensor
            target_px = (target[b:b+1] * norm_tensor).unsqueeze(0)
            diff = samples_px - target_px
            l2 = torch.sqrt((diff ** 2).sum(dim=-1))
            ade_per_sample = l2.mean(dim=-1)  # (N, B)
            best_idx = ade_per_sample.argmin(dim=0)  # (B,)
            B_idx = best_idx[0]
            all_ade.append(float(ade_per_sample[B_idx, 0]))
            fde_sample = l2[:, :, -1]
            all_fde.append(float(fde_sample[B_idx, 0]))

        # ContinuousRiskEstimator baseline (with motion)
        if samples is not None:
            s = samples[:, 0] if samples.dim() == 4 else samples
            lp = pred.get("log_probs")
            lp_s = lp[:, 0] if lp is not None and lp.dim() >= 2 else lp
            obs_traj_px = (obs[b] * norm_tensor).cpu().numpy()  # (8, 2) pixel coords
            risk_prob, _ = risk_est.estimate(s, lp_s, norm=norm_tensor, obs_trajectory=obs_traj_px)
            rp = float(risk_prob.item())
            risk_probs.append(rp)
            risk_preds.append(1 if rp > 0.5 else 0)
        else:
            risk_probs.append(0.0); risk_preds.append(0)
        risk_labels.append(label)

        # Agent-Centric risk
        if samples is not None:
            s = samples[:, 0] if samples.dim() == 4 else samples
            lp = pred.get("log_probs")
            lp_s = lp[:, 0] if lp is not None and lp.dim() >= 2 else lp
            result = agent_clf(
                obs_trajectory=obs[b],
                scene_data=scene_data,
                samples=s,
                log_probs=lp_s if lp_s is not None else None,
                norm=norm_tensor,
            )
            ap = float(result["violation_risk"].item())
            agent_probs.append(ap)
            agent_preds.append(1 if ap > 0.5 else 0)
            agent_motion.append(result["motion_feat"].cpu().numpy())
            agent_env.append(result["env_feat"].cpu().numpy())
            agent_traj.append(result["traj_feat"].cpu().numpy())
        else:
            agent_probs.append(0.0); agent_preds.append(0)
            agent_motion.append(np.zeros(13)); agent_env.append(np.zeros(8)); agent_traj.append(np.zeros(8))
        agent_labels.append(label)

        # Raw geometric features (stop-line distance from obs + pred trajectory)
        if stop_line is not None and samples is not None:
            obs_px = (obs[b] * norm_tensor).cpu().numpy()  # (8, 2) pixel coords
            # Signed distances over 8 obs frames
            d_obs = np.array([_signed_dist_to_line(obs_px[t, 0], obs_px[t, 1], stop_line) for t in range(8)])
            d_last = d_obs[-1] / 3840.0                     # normalized
            d_first = d_obs[0] / 3840.0
            d_trend = np.polyfit(np.arange(8), d_obs, 1)[0] / 3840.0  # slope per frame, normalized
            obs_crossed = float((d_obs > 0).any())           # any obs frame crossed the line
            d_obs_min_abs = np.abs(d_obs).min() / 3840.0
            # Predicted trajectory (mean):
            if "mean" in pred:
                mean_traj_px = (pred["mean"][0] * norm_tensor).cpu().numpy()  # (12, 2)
                d_pred = np.array([_signed_dist_to_line(mean_traj_px[t, 0], mean_traj_px[t, 1], stop_line) for t in range(12)])
                d_pred_min_abs = np.abs(d_pred).min() / 3840.0
                pred_crossed = float((d_pred > 0).any())
            else:
                d_pred_min_abs = 1.0; pred_crossed = 0.0
            # MC samples: fraction that cross the stop line
            s_np = (s * norm_tensor).cpu().numpy()  # (N, 12, 2)
            n_crossed = 0
            for n in range(s_np.shape[0]):
                for t in range(s_np.shape[1]):
                    if _signed_dist_to_line(s_np[n, t, 0], s_np[n, t, 1], stop_line) > 0:
                        n_crossed += 1
                        break
            mc_crossed_frac = n_crossed / s_np.shape[0]
            geo_feat.append(np.array([
                d_last, d_first, d_trend, obs_crossed, d_obs_min_abs,
                d_pred_min_abs, pred_crossed, mc_crossed_frac,
            ], dtype=np.float32))
        else:
            geo_feat.append(np.zeros(8, dtype=np.float32))

# ═══════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FLOWCHAIN + AGENT-CENTRIC RISK RESULTS")
print("=" * 60)

if all_ade:
    ade_arr = np.array(all_ade)
    fde_arr = np.array(all_fde)
    print(f"\nTrajectory (pixel, Best-of-N):")
    print(f"  ADE: median={np.median(ade_arr):.2f}px, mean={ade_arr.mean():.2f}px")
    print(f"  FDE: median={np.median(fde_arr):.2f}px, mean={fde_arr.mean():.2f}px")

# ContinuousRiskEstimator
risk_arr = np.array(risk_probs)
risk_lbl = np.array(risk_labels)
risk_m = compute_classification_metrics(risk_lbl, np.array(risk_preds), risk_arr)
print(f"\nContinuousRiskEstimator: AUC={risk_m['AUC']:.4f} F1={risk_m['F1']:.4f}")

# Agent-Centric (untrained MLP)
agent_arr = np.array(agent_probs)
agent_lbl = np.array(agent_labels)
agent_m = compute_classification_metrics(agent_lbl, np.array(agent_preds), agent_arr)
print(f"Agent-Centric (untrained): AUC={agent_m['AUC']:.4f} F1={agent_m['F1']:.4f}")

# Threshold search for ContinuousRiskEstimator
best_th, best_f1 = 0.5, 0.0
for th in np.arange(0.01, 1.0, 0.01):
    preds = (risk_arr >= th).astype(int)
    tp = ((preds == 1) & (risk_lbl == 1)).sum()
    fp = ((preds == 1) & (risk_lbl == 0)).sum()
    fn = ((risk_lbl == 1) & (preds == 0)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    if f1 > best_f1:
        best_f1, best_th = f1, th
print(f"\n  Risk Est th search: best th={best_th:.2f}, F1={best_f1:.4f}")
print(f"  Pos mean P={risk_arr[risk_lbl==1].mean():.4f}, Neg mean P={risk_arr[risk_lbl==0].mean():.4f}")

# Threshold search for Agent-Centric
abest_th, abest_f1 = 0.5, 0.0
for th in np.arange(0.01, 1.0, 0.01):
    preds = (agent_arr >= th).astype(int)
    tp = ((preds == 1) & (agent_lbl == 1)).sum()
    fp = ((preds == 1) & (agent_lbl == 0)).sum()
    fn = ((agent_lbl == 1) & (preds == 0)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    if f1 > abest_f1:
        abest_f1, abest_th = f1, th
print(f"\n  Agent-Centric th search: best th={abest_th:.2f}, F1={abest_f1:.4f}")
print(f"  Pos mean P={agent_arr[agent_lbl==1].mean():.4f}, Neg mean P={agent_arr[agent_lbl==0].mean():.4f}")

# ═══════════════════════════════════════════════════════════════
# Sklearn LR on extracted features
# motion(13d) + traj(8d) + geo(8d) = 29d
# ═══════════════════════════════════════════════════════════════
print("\n--- Sklearn LogisticRegression ---")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
    from sklearn.model_selection import cross_val_predict

    X = np.array([np.concatenate([m, t, g]) for m, t, g in
                  zip(agent_motion, agent_traj, geo_feat)])
    y = agent_lbl

    if y.sum() >= 5:
        lr = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1)
        y_prob = cross_val_predict(lr, X, y, cv=min(5, int(y.sum())), method='predict_proba')[:, 1]
        lr_auc = roc_auc_score(y, y_prob)
        lr_ap = average_precision_score(y, y_prob)

        best_lr_f1, best_lr_th = 0.0, 0.5
        for th in np.arange(0.01, 1.0, 0.01):
            f1 = f1_score(y, (y_prob >= th).astype(int))
            if f1 > best_lr_f1:
                best_lr_f1, best_lr_th = f1, th

        print(f"  motion+traj+geo (29d): AUC={lr_auc:.4f}  AP={lr_ap:.4f}  F1={best_lr_f1:.4f} (th={best_lr_th:.2f})")
        print(f"  Pos mean P={y_prob[y==1].mean():.4f}, Neg mean P={y_prob[y==0].mean():.4f}")

        # Per-feature-group ablation
        for name, cols in [
            ("motion-only (13d)", list(range(13))),
            ("traj-only (8d)", list(range(13, 21))),
            ("geo-only (8d)", list(range(21, 29))),
            ("motion+traj (21d)", list(range(21))),
            ("traj+geo (16d)", list(range(13, 29))),
            ("motion+geo (21d)", list(range(13)) + list(range(21, 29))),
        ]:
            X_sub = X[:, cols]
            y_prob_sub = cross_val_predict(
                LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1),
                X_sub, y, cv=min(5, int(y.sum())), method='predict_proba'
            )[:, 1]
            auc_sub = roc_auc_score(y, y_prob_sub)
            print(f"    {name}: AUC={auc_sub:.4f}")

        # Save features
        feat_path = "flowchain_agent_features.csv"
        feat_cols = ([f"motion_{i}" for i in range(13)] +
                     [f"traj_{i}" for i in range(8)] +
                     [f"geo_{i}" for i in range(8)])
        with open(feat_path, "w") as f:
            f.write("label," + ",".join(feat_cols) + "\n")
            for i in range(len(y)):
                all_f = np.concatenate([agent_motion[i], agent_traj[i], geo_feat[i]])
                f.write(f"{y[i]}," + ",".join(f"{x:.6f}" for x in all_f) + "\n")
        logger.info(f"Features saved to {feat_path}")

except ImportError:
    print("  sklearn not available")
