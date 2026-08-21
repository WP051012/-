"""Quick diagnostic: what do fine-tuned FlowChain predictions actually look like?"""
import sys, os, torch, yaml, numpy as np
sys.path.insert(0, '.')
from data.dataset import TrajectoryDataset
from src.baselines.baseline_models import FlowChainBase
from src.classification.crossing_probability import CrossingProbabilityEstimator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NORM = torch.tensor([3840.0, 2160.0])
TEST_DATES = {'2026_01_27'}

with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)
junction_roi = [(962.0, 792.0), (1930.0, 792.0), (1930.0, 2077.0), (962.0, 2077.0)]

print('Loading data...')
ds = TrajectoryDataset('data/processed/trajectories', label_dir='labels/',
    obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
    target_classes=['pedestrian'], mode='with_scene', max_scene_samples=10000)
scene_indices = ds.with_scene_subset()

test_viols = []
for i in scene_indices:
    s = ds.samples[i]
    if any(d.replace('_','') in s['video'] for d in TEST_DATES) and s.get('is_violation', False):
        sample = ds[i]
        test_viols.append(sample)
print(f'Test violations with scene data: {len(test_viols)}')

p_cross_est = CrossingProbabilityEstimator(crossing_region=junction_roi)

for name, ckpt in [('Original', 'checkpoints/flowchain_best.pt'),
                    ('Fine-tuned', 'checkpoints/flowchain_best_finetuned.pt')]:
    model = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()

    print(f'\n=== {name} FlowChain ===')
    all_pc = []
    endpoints_junction = 0
    total_samples = 0

    for idx, s in enumerate(test_viols[:10]):
        obs = s['obs_trajectory'].to(DEVICE).unsqueeze(0) / NORM.to(DEVICE)
        tl = s.get('scene', {}).get('traffic_light_states', [])
        sig_state = tl[-1] if tl else '?'

        with torch.no_grad():
            pred = model(obs_trajectory=obs, num_samples=100)
        samples = pred['samples']
        if samples.dim() == 4:
            samples = samples[:, 0, :, :]
        samples_px = samples * NORM.to(samples.device)
        pc = float(p_cross_est.compute_p_cross(samples_px))
        all_pc.append(pc)

        # Last predicted point for each sample
        endpoints = samples_px[:, -1, :].cpu().numpy()  # (100, 2)

        # Count how many endpoints are inside junction
        from data.dataset import _point_in_polygon_px
        ep_in_junc = sum(1 for i in range(100) if _point_in_polygon_px(endpoints[i,0], endpoints[i,1], junction_roi))

        # Mean trajectory direction: start→end vector
        mean_end = endpoints.mean(axis=0)

        # GT last position
        tgt = s['target_trajectory']
        gt_end = tgt[-1].numpy()

        print(f'  Viol#{idx} [{sig_state}]: P_cross={pc:.3f}, '
              f'endpoints_in_junction={ep_in_junc}/100, '
              f'mean_end=({mean_end[0]:.0f},{mean_end[1]:.0f}), '
              f'GT_end=({gt_end[0]:.0f},{gt_end[1]:.0f})')

    print(f'  ---')
    print(f'  Mean P_cross: {np.mean(all_pc):.4f}')
    print(f'  P_cross > 0: {(np.array(all_pc) > 0).sum()}/{len(all_pc)}')

print('\nDone!')
