"""Compare original vs fine-tuned FlowChain P_cross — lightweight version.
Loads only test-scene subset directly, avoiding full dataset load."""
import sys, os, logging, torch, yaml, numpy as np
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, '.')
from data.dataset import TrajectoryDataset
from src.baselines.baseline_models import FlowChainBase
from src.classification.crossing_probability import CrossingProbabilityEstimator, compute_signal_factor

logging.basicConfig(level=logging.INFO)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NORM = torch.tensor([3840.0, 2160.0])

with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)
junction_roi = [(962.0, 792.0), (1930.0, 792.0), (1930.0, 2077.0), (962.0, 2077.0)]

# Test dates from run_experiments.py
TEST_DATES = {'2026_01_27'}

# Load full dataset but only use scene subset
print('Loading data...')
ds = TrajectoryDataset(
    'data/processed/trajectories', label_dir='labels/',
    obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
    target_classes=['pedestrian'], mode='with_scene', max_scene_samples=10000,
)
scene_indices = ds.with_scene_subset()
print(f'Scene samples: {len(scene_indices)}')

# Filter to test dates
test_scene = []
for i in scene_indices:
    s = ds.samples[i]
    video = s.get('video', '')
    if any(d.replace('_', '') in video for d in TEST_DATES):
        # Get the full sample with scene data
        sample = ds[i]
        test_scene.append(sample)
    if len(test_scene) >= 1010:
        break

print(f'Test samples: {len(test_scene)}')
viol_count = sum(1 for s in test_scene if s.get('is_violation', False))
print(f'Violations in test: {viol_count}')

p_cross_est = CrossingProbabilityEstimator(crossing_region=junction_roi)

models = {
    'Original FlowChain': 'checkpoints/flowchain_best.pt',
    'Fine-tuned FlowChain': 'checkpoints/flowchain_best_finetuned.pt',
}

for name, ckpt_path in models.items():
    print(f'\n=== {name} ===')
    if not os.path.exists(ckpt_path):
        print(f'  Checkpoint not found: {ckpt_path}')
        continue

    model = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()

    viol_p_cross = []
    nonviol_p_cross = []
    viol_signals = []

    for s in tqdm(test_scene, desc=name):
        obs = s['obs_trajectory'].to(DEVICE).unsqueeze(0) / NORM.to(DEVICE)
        scene = s.get('scene', {})
        tl_states = scene.get('traffic_light_states', [])
        is_viol = s.get('is_violation', False)

        with torch.no_grad():
            pred = model(obs_trajectory=obs, num_samples=100)

        samples = pred.get('samples', pred.get('best_sample'))
        if samples.dim() == 4:
            samples = samples[:, 0, :, :]
        # Denormalize before computing P_cross
        samples_px = samples * NORM.to(samples.device)
        p_cross = float(p_cross_est.compute_p_cross(samples_px))
        sig = compute_signal_factor(tl_states)

        if is_viol:
            viol_p_cross.append(p_cross)
            viol_signals.append(sig)
        else:
            nonviol_p_cross.append(p_cross)

    viol_pc = np.array(viol_p_cross)
    nonviol_pc = np.array(nonviol_p_cross)

    print(f'  P_cross (violations):   mean={viol_pc.mean():.4f}, median={np.median(viol_pc):.4f}')
    print(f'  P_cross (non-violations): mean={nonviol_pc.mean():.4f}, median={np.median(nonviol_pc):.4f}')
    print(f'  P_cross > 0 violations: {(viol_pc > 0).sum()}/{len(viol_pc)} ({(viol_pc > 0).mean()*100:.1f}%)')
    print(f'  P_cross > 0.5 violations: {(viol_pc > 0.5).sum()}/{len(viol_pc)} ({(viol_pc > 0.5).mean()*100:.1f}%)')

    for label, val in [('green', 0.0), ('unknown', 0.5), ('yellow', 0.7), ('red', 1.0)]:
        count = (np.abs(np.array(viol_signals) - val) < 0.01).sum()
        print(f'  Violations at {label}: {count}/{len(viol_signals)}')

print('\nDone!')
