"""Wrap old-format checkpoint with epoch marker for resume."""
import torch

ck = torch.load("checkpoints/flowchain_domain_filtered.pt", map_location="cpu", weights_only=False)

if isinstance(ck, dict) and "epoch" not in ck:
    # Old format: raw state_dict — wrap with epoch=5
    new_ck = {"model_state": ck, "epoch": 5, "best_val_loss": float("inf")}
    torch.save(new_ck, "checkpoints/flowchain_domain_filtered.pt")
    print("Wrapped -> epoch=5")
elif isinstance(ck, dict) and "epoch" in ck:
    print(f"Already has epoch={ck['epoch']}")
else:
    print(f"Unknown format: {type(ck)}")
