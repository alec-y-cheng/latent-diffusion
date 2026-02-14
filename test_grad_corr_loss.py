import torch
from ldm.modules.losses.grad_corr import GradCorrLoss

def test_grad_corr():
    criterion = GradCorrLoss()
    
    # 1. Identity Case (Perfect Correlation)
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    y = x.clone() # Identical
    
    loss = criterion(x, y)
    print(f"Identity Loss (Expect 0.0): {loss.item()}")
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    
    # 2. Inverse Case (Perfect Negative Correlation)
    y_inv = -x.clone()
    loss_inv = criterion(x, y_inv)
    print(f"Inverse Loss (Expect 2.0): {loss_inv.item()}")
    assert torch.isclose(loss_inv, torch.tensor(2.0), atol=1e-6)
    
    # 3. Gradient Propagation
    loss.backward()
    print(f"Input Gradient Norm: {x.grad.norm().item()}")
    assert x.grad is not None
    assert x.grad.norm() > 0 or loss.item() == 0 # derivative at local min might be 0? 
    # Actually for identity, corr=1. derivative of (1-corr) might be 0 if it's a stable maximum of corr.
    
    # Let's try non-perfect case for grad check
    x2 = torch.randn(2, 1, 32, 32, requires_grad=True)
    y2 = torch.randn(2, 1, 32, 32) # Random target
    loss2 = criterion(x2, y2)
    loss2.backward()
    print(f"Random Loss: {loss2.item()}")
    print(f"Random Input Grad Norm: {x2.grad.norm().item()}")
    assert x2.grad.norm() > 0

    print("All tests passed!")

if __name__ == "__main__":
    test_grad_corr()
