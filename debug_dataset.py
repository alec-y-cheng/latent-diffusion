import torch
import numpy as np
from torch.utils.data import DataLoader
from ldm.data.cfd_data import CFDConditionalDataset

def test_shapes():
    # Use a dummy path to trigger the mock data generation
    ds = CFDConditionalDataset("dummy_path.npz")
    print(f"Dataset length: {len(ds)}")
    
    item = ds[0]
    print("Item Keys:", item.keys())
    print("Image Shape:", item['image'].shape)
    print("Cond Shape:", item['cond'].shape)
    
    dl = DataLoader(ds, batch_size=2)
    batch = next(iter(dl))
    
    img = batch['image']
    cond = batch['cond']
    
    print("\nBatch Shapes (Channels Last expected):")
    print(f"Image: {img.shape}")
    print(f"Cond: {cond.shape}")
    
    # Simulate DDPM rearrange
    from einops import rearrange
    img_re = rearrange(img, 'b h w c -> b c h w')
    cond_re = rearrange(cond, 'b h w c -> b c h w')
    
    print("\nAfter Rearrange (Channels First):")
    print(f"Image: {img_re.shape}")
    print(f"Cond: {cond_re.shape}")

if __name__ == "__main__":
    test_shapes()
