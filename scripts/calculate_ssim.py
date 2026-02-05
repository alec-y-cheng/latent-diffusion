import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchvision import transforms

def load_image_as_tensor(path):
    img = Image.open(path).convert('L') # Convert to grayscale
    transform = transforms.Compose([
        transforms.ToTensor() # Converts to [0, 1]
    ])
    return transform(img).unsqueeze(0) # Add batch dimension: (1, 1, H, W)

def calculate_folder_ssim(gt_folder, pred_folder, extension="png"):
    pred_files = sorted(glob.glob(os.path.join(pred_folder, f"*.{extension}")))
    
    if not pred_files:
        print(f"No files found in {pred_folder} with extension {extension}")
        return

    ssim_values = []
    
    print(f"Found {len(pred_files)} files. Calculating SSIM...")
    
    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        gt_path = os.path.join(gt_folder, filename)
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found for {filename}. Skipping.")
            continue
            
        try:
            pred_tensor = load_image_as_tensor(pred_path)
            gt_tensor = load_image_as_tensor(gt_path)
            
            # Ensure shapes match
            if pred_tensor.shape != gt_tensor.shape:
                print(f"Warning: Shape mismatch for {filename}: {pred_tensor.shape} vs {gt_tensor.shape}. Skipping.")
                continue

            # Calculate SSIM
            # data_range=1.0 because ToTensor scales to [0, 1]
            val = ssim(pred_tensor, gt_tensor, data_range=1.0)
            ssim_values.append(val.item())
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if ssim_values:
        avg_ssim = sum(ssim_values) / len(ssim_values)
        print(f"\nProcessed {len(ssim_values)} pairs.")
        print(f"Average SSIM: {avg_ssim:.4f}")
    else:
        print("No valid pairs processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SSIM between two folders of images.")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth folder")
    parser.add_argument("--pred", type=str, required=True, help="Path to predictions folder")
    parser.add_argument("--ext", type=str, default="png", help="Image extension (default: png)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gt):
        print(f"Error: Ground truth folder {args.gt} does not exist.")
        exit(1)
    if not os.path.exists(args.pred):
        print(f"Error: Predictions folder {args.pred} does not exist.")
        exit(1)
        
    calculate_folder_ssim(args.gt, args.pred, args.ext)
