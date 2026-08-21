"""Diagnose dataset condition matching"""
import sys, json, torch
sys.path.insert(0, '/root/red-light-prediction')
from data.dataset import TrajectoryDataset

cm = torch.load('data/gat_conditions.pt', map_location='cpu', weights_only=False)
print(f'Condition map: {len(cm)} videos')

with open('data/domains/domain_labels_int.json') as f:
    dmap = json.load(f)

ds = TrajectoryDataset(
    data_dir='data/processed/trajectories', label_dir='labels',
    obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
    target_classes=['pedestrian'], mode='trajectory_only',
    max_samples=1000, domain_label_map=dmap, condition_map=cm,
)
print(f'Dataset: {len(ds)} samples')

# Check first 10 samples
for i in range(min(10, len(ds))):
    s = ds[i]
    emb = s.get('cond_embedding')
    v = s.get('video', '?')
    tid = s.get('track_id', '?')
    obs_start = s.get('obs_start', '?')
    key = f'{tid}__{obs_start}'
    if emb is not None:
        print(f'  [{i}] video={v}, key={key}, norm={emb.norm().item():.6f}')

# Check matching detail for sample 0
s0 = ds[0]
v0 = s0.get('video', '')
tid0 = s0.get('track_id', '')
obs0 = s0.get('obs_start', '')
key0 = f'{tid0}__{obs0}'
print(f'\nSample 0: video={v0}, key={key0}')
print(f'  video in cm? {v0 in cm}')
if v0 in cm:
    print(f'  key in cm[video]? {key0 in cm[v0]}')
    if key0 in cm[v0]:
        print(f'  raw emb norm: {cm[v0][key0].norm().item():.6f}')
    else:
        print(f'  Available keys (first 5): {list(cm[v0].keys())[:5]}')
else:
    print(f'  Available videos (first 3): {list(cm.keys())[:3]}')

# Full stats
nz = sum(1 for i in range(len(ds)) if ds[i].get('cond_embedding') is not None and ds[i]['cond_embedding'].norm().item() > 0)
print(f'\nNonzero cond: {nz}/{len(ds)}')
