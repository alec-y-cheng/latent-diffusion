
import numpy as np
import sys

def verify_channels(file_path):
    print(f"Loading {file_path}...")
    try:
        data = np.load(file_path)
        if 'X' in data:
            X = data['X']
        elif 'arr_0' in data:
            X = data['arr_0']
        else:
            print("Could not find 'X' or 'arr_0' in npz")
            return

        print(f"Data Shape: {X.shape}")
        # Expected: (N, 8, 504, 504) or (N, 1, 8, 504, 504)
        
        if X.ndim == 5:
            X = X[:, 0, :, :, :]
            
        # Channels 6 and 7
        ch6 = X[:, 6, :, :].flatten()
        ch7 = X[:, 7, :, :].flatten()
        
        print("\n--- Statistics ---")
        print(f"Channel 6 (Supposed Sin): Mean={np.mean(ch6):.4f}, Std={np.std(ch6):.4f}, Min={np.min(ch6):.4f}, Max={np.max(ch6):.4f}")
        print(f"Channel 7 (Supposed Cos): Mean={np.mean(ch7):.4f}, Std={np.std(ch7):.4f}, Min={np.min(ch7):.4f}, Max={np.max(ch7):.4f}")
        
        # Hypothesis Check
        # If flow is mostly Left-to-Right (0 degrees), Cos should be ~1.0, Sin ~0.0
        print("\n--- Hypothesis Check ---")
        if np.mean(ch7) > 0.5:
             print("Channel 7 has high positive mean -> Consistent with Cos(0) = 1 (Left-to-Right Flow)")
        else:
             print("Channel 7 mean is low or negative -> Inconsistent with naive Left-to-Right Cosine assumption")
             
        if abs(np.mean(ch6)) < 0.2:
             print("Channel 6 mean is near 0 -> Consistent with Sin(0) = 0")
        else:
             print("Channel 6 mean is significant -> Inconsistent or flow has strong Y component")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    path = "data/full_dataset.npz"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    verify_channels(path)
