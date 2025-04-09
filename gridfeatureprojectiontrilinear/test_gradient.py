import torch
import torch.nn as nn
from grid_feature_projection_trilinear import GridFeatureProjectionTrilinear, grid_feature_projection_trilinear_pytorch

def test_gradient():
    # Set up device
    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')

    print(f"Using device: {device}")

    # Create test data with requires_grad=True
    batch_size = 2
    num_vertices = 100
    num_features = 16
    volume_size = 16

    # Create vertices and features with requires_grad=True
    vertices = torch.randn(batch_size, num_vertices, 3, device=device, requires_grad=True)
    features = torch.randn(batch_size, num_vertices, num_features, device=device, requires_grad=True)

    # Create the projection module
    projection = GridFeatureProjectionTrilinear(volume_size)

    # Reference forward pass
    vertices_ref = vertices.clone().detach().requires_grad_(True)
    features_ref = features.clone().detach().requires_grad_(True)
    volume_ref = grid_feature_projection_trilinear_pytorch(vertices_ref, volume_size, features_ref)
    volume_ref.retain_grad()  # Retain gradients for intermediate tensor

    # Forward pass
    volume = projection(vertices, features)
    volume.retain_grad()  # Retain gradients for intermediate tensor
    print(f"Volume shape: {volume.shape}")

    # Check forward pass results
    if not torch.allclose(volume, volume_ref, rtol=1e-4, atol=1e-4):
        print("Forward pass mismatch:")
        print(f"MSE volume: {torch.nn.functional.mse_loss(volume, volume_ref)}")
        print(f"MSE features: {torch.nn.functional.mse_loss(features, features_ref)}")
        print(f"MSE vertices: {torch.nn.functional.mse_loss(vertices, vertices_ref)}")
        assert False, "Forward pass results don't match"

    # Compute a loss that will exercise all elements
    loss_ref = torch.nn.functional.mse_loss(volume_ref, torch.ones_like(volume_ref))
    loss = torch.nn.functional.mse_loss(volume, torch.ones_like(volume))

    # Verify losses match
    if not torch.allclose(loss, loss_ref, rtol=1e-4, atol=1e-4):
        print(f"Loss mismatch: {loss.item()} vs {loss_ref.item()}")
        assert False, "Losses don't match"

    # Clear any existing gradients
    vertices.grad = None if vertices.grad is not None else None
    features.grad = None if features.grad is not None else None
    vertices_ref.grad = None if vertices_ref.grad is not None else None
    features_ref.grad = None if features_ref.grad is not None else None

    # Backward pass
    loss_ref.backward(retain_graph=True)
    loss.backward()

    # Ensure gradients were computed
    assert vertices_ref.grad is not None, "Reference vertices gradient is None"
    assert features_ref.grad is not None, "Reference features gradient is None"
    assert vertices.grad is not None, "Vertices gradient is None"
    assert features.grad is not None, "Features gradient is None"

    # Clone gradients for comparison
    vertices_grad_ref = vertices_ref.grad.clone()
    features_grad_ref = features_ref.grad.clone()
    vertices_grad = vertices.grad.clone()
    features_grad = features.grad.clone()

    # Print gradient information
    print(f"Vertices grad shape: {vertices_grad.shape}")
    print(f"Features grad shape: {features_grad.shape}")
    print(f"Vertices grad max diff: {(vertices_grad - vertices_grad_ref).abs().max().item()}")
    print(f"Features grad max diff: {(features_grad - features_grad_ref).abs().max().item()}")

    # Check if gradients match
    grad_match = (torch.allclose(vertices_grad, vertices_grad_ref, rtol=1e-4, atol=1e-2) and
                 torch.allclose(features_grad, features_grad_ref, rtol=1e-4, atol=1e-2))

    if not grad_match:
        print("\nGradient check failed:")
        print(f"Vertices grad MSE: {torch.nn.functional.mse_loss(vertices_grad, vertices_grad_ref)}")
        print(f"Features grad MSE: {torch.nn.functional.mse_loss(features_grad, features_grad_ref)}")
        assert False, "Gradients don't match"

    return True

if __name__ == "__main__":
    success = test_gradient()
    print(f"Gradient test {'passed' if success else 'failed'}")