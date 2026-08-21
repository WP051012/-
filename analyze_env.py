"""Analyze env feature distributions."""
import numpy as np
data = np.loadtxt("flowchain_agent_features.csv", delimiter=",", skiprows=1)
labels = data[:, 0]
env = data[:, 14:22]
print(f"Total samples: {len(labels)}")
print(f"Violations: {int(labels.sum())}")
for i in range(8):
    col = env[:, i]
    print(f"env[{i}]: mean={col.mean():.4f}, std={col.std():.4f}, nonzero={(col!=0).sum()}/{len(col)}")
print()
any_nonzero = (env != 0).any(axis=1).sum()
print(f"Samples with any non-zero env feature: {any_nonzero}/{len(env)}")
print()
for i in range(8):
    col = env[:, i]
    print(f"env[{i}] pos(mean)={col[labels==1].mean():.4f} neg(mean)={col[labels==0].mean():.4f}")
