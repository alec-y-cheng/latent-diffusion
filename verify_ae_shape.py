import sys
import os
import torch
from omegaconf import OmegaConf
from ldm.models.autoencoder import AutoencoderKL

# Add current directory to path so ldm imports work
sys.path.append(os.getcwd())

def verify():
    # Load default config
    config_path = "configs/autoencoder/autoencoder_kl_32x32x4.yaml"
    print(f"Loading config from {config_path}")
    conf = OmegaConf.load(config_path)

    # Modify for 1 channel input
    print("Modifying config: in_channels=1, out_ch=1")
    conf.model.params.ddconfig.in_channels = 1
    conf.model.params.ddconfig.out_ch = 1
    
    # Instantiate model
    print("Instantiating AutoencoderKL...")
    try:
        model = AutoencoderKL(**conf.model.params)
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return

    # Create dummy input: (Batch, Channels, Height, Width)
    # Using batch size 2 to save memory, user has 2000
    B, C, H, W = 2, 1, 504, 504
    x = torch.randn(B, C, H, W)
    print(f"Created dummy input with shape: {x.shape}")

    # Test Forward Pass
    print("Running forward pass...")
    try:
        dec, posterior = model(x)
        print("Forward pass successful!")
        print(f"Output shape: {dec.shape}")
        
        if dec.shape == (B, C, H, W):
            print("SUCCESS: Input and Output shapes match.")
        else:
            print(f"WARNING: Output shape {dec.shape} does not match input {x.shape}")
            
        # Check latent shape
        # Encode
        post = model.encode(x)
        z = post.mode()
        print(f"Latent shape: {z.shape}")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
