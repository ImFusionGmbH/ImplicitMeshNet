import os
import sys
import platform
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from grid_feature_projection_trilinear import (
    grid_feature_projection_trilinear_pytorch,
    grid_feature_projection_trilinear,
    CUDA_AVAILABLE,
    METAL_AVAILABLE,
    IS_MACOS
)

def reference_trilinear_interpolation(vertices: np.ndarray, features: np.ndarray, volume_size: int) -> np.ndarray:
    """
    Reference implementation of trilinear interpolation for testing correctness.
    This is a pure NumPy implementation that computes the exact expected output.

    Args:
        vertices: (N, 3) array of vertex coordinates (-1 to 1)
        features: (N, C) array of vertex features
        volume_size: Size of the cubic volume grid

    Returns:
        volume: (1, C, D, H, W) feature volume
    """
    num_vertices = vertices.shape[0]
    num_features = features.shape[1]

    # Create output volume
    volume: np.ndarray = np.zeros((1, num_features, volume_size, volume_size, volume_size), dtype=np.float32)

    # Scale vertices to [-1, 1] if they aren't already
    scaled_vertices = np.clip(vertices, -1, 1)

    # Scale vertex coordinates to volume indices (floating point)
    indices = ((scaled_vertices + 1) * (volume_size - 1) / 2)

    for i in range(num_vertices):
        # Get the floating-point indices
        idx_x, idx_y, idx_z = indices[i]

        # Get floor indices
        x0 = int(np.floor(idx_x))
        y0 = int(np.floor(idx_y))
        z0 = int(np.floor(idx_z))

        # Clamp to valid range
        x0 = max(0, min(x0, volume_size - 2))
        y0 = max(0, min(y0, volume_size - 2))
        z0 = max(0, min(z0, volume_size - 2))

        # Get ceiling indices
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # Compute interpolation weights
        wx = idx_x - x0
        wy = idx_y - y0
        wz = idx_z - z0

        # Compute the 8 weights for trilinear interpolation
        w000 = (1 - wx) * (1 - wy) * (1 - wz)
        w001 = (1 - wx) * (1 - wy) * wz
        w010 = (1 - wx) * wy * (1 - wz)
        w011 = (1 - wx) * wy * wz
        w100 = wx * (1 - wy) * (1 - wz)
        w101 = wx * (1 - wy) * wz
        w110 = wx * wy * (1 - wz)
        w111 = wx * wy * wz

        # Add weighted features to the 8 surrounding voxels
        # Swap x and z to match the Metal implementation
        # Original: volume has shape [C, D, H, W] with indices [c, z, y, x]
        # Modified: volume has shape [C, D, H, W] with indices [c, x, y, z]
        for c in range(num_features):
            feature_val = features[i, c]
            volume[0, c, x0, y0, z0] += feature_val * w000
            volume[0, c, x0, y0, z1] += feature_val * w001
            volume[0, c, x0, y1, z0] += feature_val * w010
            volume[0, c, x0, y1, z1] += feature_val * w011
            volume[0, c, x1, y0, z0] += feature_val * w100
            volume[0, c, x1, y0, z1] += feature_val * w101
            volume[0, c, x1, y1, z0] += feature_val * w110
            volume[0, c, x1, y1, z1] += feature_val * w111

    return volume

def test_correctness(volume_size: int=32, num_vertices: int=10, num_features: int=3, tolerance: float=1e-5) -> bool:
    """
    Test the correctness of the grid feature projection implementations.

    Args:
        volume_size: Size of the cubic volume grid
        num_vertices: Number of test vertices
        num_features: Number of feature channels
        tolerance: Maximum allowed difference between implementations

    Returns:
        success: True if all tests pass, False otherwise
    """
    print(f"Testing correctness with {num_vertices} vertices, {num_features} features, {volume_size}³ volume")

    # Create deterministic test data
    np.random.seed(42)
    vertices_np = np.random.uniform(-0.9, 0.9, (num_vertices, 3)).astype(np.float32)
    features_np = np.random.randn(num_vertices, num_features).astype(np.float32)

    # Convert to PyTorch tensors
    vertices = torch.from_numpy(vertices_np)
    features = torch.from_numpy(features_np)

    # Compute reference result with NumPy
    reference_volume = reference_trilinear_interpolation(vertices_np, features_np, volume_size)
    print(f"Reference volume shape: {reference_volume.shape}")

    # Compute result with PyTorch implementation
    pytorch_volume = grid_feature_projection_trilinear_pytorch(vertices, volume_size, features)
    pytorch_volume_np = pytorch_volume.detach().cpu().numpy()

    # Compare PyTorch implementation with reference
    pytorch_diff = np.abs(pytorch_volume_np - reference_volume).max()
    print(f"PyTorch implementation max difference: {pytorch_diff}")
    pytorch_correct = pytorch_diff < tolerance
    print(f"PyTorch implementation correct: {pytorch_correct}")

    # Test CUDA implementation if available
    cuda_correct = True
    if CUDA_AVAILABLE and torch.cuda.is_available() and not IS_MACOS:
        print("Testing CUDA implementation...")
        vertices_cuda = vertices.cuda()
        features_cuda = features.cuda()

        # Compute result with CUDA implementation
        cuda_volume = grid_feature_projection_trilinear(vertices_cuda, volume_size, features_cuda)
        cuda_volume_np = cuda_volume.detach().cpu().numpy()

        # Compare CUDA implementation with reference
        cuda_diff = np.abs(cuda_volume_np - reference_volume).max()
        print(f"CUDA implementation max difference: {cuda_diff}")
        cuda_correct = cuda_diff < tolerance
        print(f"CUDA implementation correct: {cuda_correct}")
    elif not IS_MACOS:
        print("CUDA not available, skipping CUDA test")

    # Test Metal implementation if available
    metal_correct = True
    if METAL_AVAILABLE and IS_MACOS:
        print("Testing Metal implementation...")

        # Compute result with Metal implementation
        metal_volume = grid_feature_projection_trilinear(vertices.to("mps"), volume_size, features.to("mps"))
        metal_volume_np = metal_volume.detach().cpu().numpy()

        # Compare Metal implementation with reference
        metal_diff = np.abs(metal_volume_np - reference_volume).max()
        print(f"Metal implementation max difference: {metal_diff}")
        metal_correct = metal_diff < tolerance
        print(f"Metal implementation correct: {metal_correct}")
    elif IS_MACOS:
        print("Metal not available, skipping Metal test")

    # Visualize results for the first feature channel
    visualize_results(reference_volume, pytorch_volume_np, volume_size)

    # Return overall success
    return pytorch_correct and cuda_correct and metal_correct

def test_gradient_correctness(volume_size=16, tolerance=1e-5):
    """
    Test the correctness of gradient computation in the grid feature projection implementations.

    Args:
        volume_size: Size of the cubic volume grid
        tolerance: Maximum allowed difference between implementations

    Returns:
        success: True if all gradient tests pass, False otherwise
    """
    print(f"\nTesting gradient computation with {volume_size}³ volume")

    # Create a simple test case with a few vertices
    vertices_np = np.array([
        [0.0, 0.0, 0.0],    # Center of the volume
        [0.5, 0.5, 0.5],    # Between 8 voxels
        [-0.5, -0.5, -0.5]  # Between 8 voxels in negative direction
    ], dtype=np.float32)

    # Simple features: each vertex has a unique value
    features_np = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)

    # Convert to PyTorch tensors and require gradients
    vertices = torch.from_numpy(vertices_np)[None,].requires_grad_(True)
    features = torch.from_numpy(features_np)[None,].requires_grad_(True)

    # Forward pass with PyTorch implementation
    pytorch_volume = grid_feature_projection_trilinear_pytorch(vertices, volume_size, features)

    # Create a simple loss: sum of all elements in the volume
    pytorch_loss = (pytorch_volume - torch.ones_like(pytorch_volume)).abs().sum()

    # Backward pass to compute gradients
    pytorch_loss.backward()

    # Store the gradients
    pytorch_grad_vertices = vertices.grad.clone()
    pytorch_grad_features = features.grad.clone()

    print(f"PyTorch gradient shapes - vertices: {pytorch_grad_vertices.shape}, features: {pytorch_grad_features.shape}")


    # Test CUDA implementation if available
    cuda_correct = True
    if CUDA_AVAILABLE and torch.cuda.is_available() and not IS_MACOS:
        print("Testing CUDA gradient computation...")
        vertices.grad.zero_()
        features.grad.zero_()

        # Forward pass with CUDA implementation
        cuda_volume = grid_feature_projection_trilinear(vertices.to("cuda"), volume_size, features.to("cuda"))

        # Create a simple loss: sum of all elements in the volume
        cuda_loss = (cuda_volume - torch.ones_like(cuda_volume)).abs().sum()

        # Backward pass to compute gradients
        cuda_loss.backward()

        # Get the gradients
        cuda_grad_vertices = vertices.grad.cpu()
        cuda_grad_features = features.grad.cpu()

        vertices_grad_diff = torch.abs(cuda_grad_vertices - pytorch_grad_vertices).max().item()
        features_grad_diff = torch.abs(cuda_grad_features - pytorch_grad_features).max().item()

        print(f"CUDA vertices gradient max difference: {vertices_grad_diff}")
        print(f"CUDA features gradient max difference: {features_grad_diff}")

        cuda_correct = vertices_grad_diff < tolerance and features_grad_diff < tolerance
        print(f"CUDA gradient computation correct: {cuda_correct}")

        # Print some gradient values for inspection
        print("\nSample gradient values (first vertex):")
        print(f"PyTorch vertices grad: {pytorch_grad_vertices[0]}")
        print(f"CUDA vertices grad: {cuda_grad_vertices[0]}")
    elif not IS_MACOS:
        print("CUDA not available, skipping CUDA gradient test")

    # Test Metal implementation if available
    metal_correct = True
    if METAL_AVAILABLE and IS_MACOS:
        print("Testing Metal gradient computation...")

        # Reset gradients for Metal test
        vertices.grad.zero_()
        features.grad.zero_()

        # Forward pass with Metal implementation
        metal_volume = grid_feature_projection_trilinear(vertices.to("mps"), volume_size, features.to("mps"))

        # Create a simple loss: sum of all elements in the volume
        metal_loss = (metal_volume - torch.ones_like(metal_volume)).abs().sum()

        # Backward pass to compute gradients
        metal_loss.backward()

        # Get the gradients
        metal_grad_vertices = vertices.grad
        metal_grad_features = features.grad

        vertices_grad_diff = torch.abs(metal_grad_vertices - pytorch_grad_vertices).max().item()
        features_grad_diff = torch.abs(metal_grad_features - pytorch_grad_features).max().item()

        print(f"Metal vertices gradient max difference: {vertices_grad_diff}")
        print(f"Metal features gradient max difference: {features_grad_diff}")

        metal_correct = vertices_grad_diff < tolerance and features_grad_diff < tolerance
        print(f"Metal gradient computation correct: {metal_correct}")

        # Print some gradient values for inspection
        print("\nSample gradient values (first vertex):")
        print(f"PyTorch vertices grad: {pytorch_grad_vertices[0]}")
        print(f"Metal vertices grad: {metal_grad_vertices[0]}")
    elif IS_MACOS:
        print("Metal not available, skipping Metal gradient test")

    # Visualize the gradients
    visualize_gradients(pytorch_grad_vertices.numpy(), pytorch_grad_features.numpy())

    return cuda_correct and metal_correct

def visualize_gradients(grad_vertices, grad_features):
    """
    Visualize the gradients for vertices and features.

    Args:
        grad_vertices: Gradients for vertices
        grad_features: Gradients for features
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot vertex gradients
    axes[0].bar(range(grad_vertices.size), grad_vertices.flatten())
    axes[0].set_title("Vertex Gradients")
    axes[0].set_xlabel("Gradient Component")
    axes[0].set_ylabel("Gradient Value")

    # Plot feature gradients
    axes[1].bar(range(grad_features.size), grad_features.flatten())
    axes[1].set_title("Feature Gradients")
    axes[1].set_xlabel("Gradient Component")
    axes[1].set_ylabel("Gradient Value")

    plt.tight_layout()
    plt.savefig("grid_projection_gradients.png")
    print("Saved gradient visualization to grid_projection_gradients.png")

def visualize_results(reference_volume, pytorch_volume, volume_size):
    """
    Visualize the reference and PyTorch implementation results.

    Args:
        reference_volume: Reference volume from NumPy implementation
        pytorch_volume: Volume from PyTorch implementation
        volume_size: Size of the cubic volume grid
    """
    # Create a figure to visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Get the middle slice for visualization
    mid_slice = volume_size // 2

    # Plot reference volume
    axes[0, 0].imshow(reference_volume[0, 0, mid_slice, :, :])
    axes[0, 0].set_title("Reference (XY slice)")
    axes[0, 1].imshow(reference_volume[0, 0, :, mid_slice, :])
    axes[0, 1].set_title("Reference (ZX slice)")
    axes[0, 2].imshow(reference_volume[0, 0, :, :, mid_slice])
    axes[0, 2].set_title("Reference (ZY slice)")

    # Plot PyTorch volume
    axes[1, 0].imshow(pytorch_volume[0, 0, mid_slice, :, :])
    axes[1, 0].set_title("PyTorch (XY slice)")
    axes[1, 1].imshow(pytorch_volume[0, 0, :, mid_slice, :])
    axes[1, 1].set_title("PyTorch (ZX slice)")
    axes[1, 2].imshow(pytorch_volume[0, 0, :, :, mid_slice])
    axes[1, 2].set_title("PyTorch (ZY slice)")

    # Save the figure
    plt.tight_layout()
    plt.savefig("grid_projection_correctness.png")
    print("Saved visualization to grid_projection_correctness.png")

def test_with_specific_vertices():
    """
    Test with a small set of specific vertices for which we can manually verify the results.
    """
    print("\nTesting with specific vertices...")

    # Create a simple test case with 3 vertices
    vertices_np = np.array([
        [0.0, 0.0, 0.0],    # Center of the volume
        [0.5, 0.5, 0.5],    # Between 8 voxels
        [-0.5, -0.5, -0.5]  # Between 8 voxels in negative direction
    ], dtype=np.float32)

    # Simple features: each vertex has a unique value
    features_np = np.array([
        [1.0, 0.0, 0.0],    # Red
        [0.0, 1.0, 0.0],    # Green
        [0.0, 0.0, 1.0]     # Blue
    ], dtype=np.float32)

    volume_size = 8  # Small volume for easy verification

    # Convert to PyTorch tensors
    vertices = torch.from_numpy(vertices_np)
    features = torch.from_numpy(features_np)

    # Compute reference result with NumPy
    reference_volume = reference_trilinear_interpolation(vertices_np, features_np, volume_size)

    # Compute result with PyTorch implementation
    pytorch_volume = grid_feature_projection_trilinear_pytorch(vertices, volume_size, features)
    pytorch_volume_np = pytorch_volume.detach().cpu().numpy()

    # Print expected values for specific voxels
    print("\nExpected values at specific voxels:")

    # Center vertex (should be at [3, 3, 3] in a 8x8x8 volume)
    center_idx = (volume_size - 1) // 2
    print(f"Center voxel [{center_idx}, {center_idx}, {center_idx}]:")
    print(f"  Reference: {reference_volume[0, :, center_idx, center_idx, center_idx]}")
    print(f"  PyTorch:   {pytorch_volume_np[0, :, center_idx, center_idx, center_idx]}")

    # Voxels around the second vertex (should be at [5, 5, 5] in a 8x8x8 volume)
    second_idx = int((0.5 + 1) * (volume_size - 1) / 2)
    print(f"\nSecond vertex voxel [{second_idx}, {second_idx}, {second_idx}]:")
    print(f"  Reference: {reference_volume[0, :, second_idx, second_idx, second_idx]}")
    print(f"  PyTorch:   {pytorch_volume_np[0, :, second_idx, second_idx, second_idx]}")

    # Voxels around the third vertex (should be at [1, 1, 1] in a 8x8x8 volume)
    third_idx = int((-0.5 + 1) * (volume_size - 1) / 2)
    print(f"\nThird vertex voxel [{third_idx}, {third_idx}, {third_idx}]:")
    print(f"  Reference: {reference_volume[0, :, third_idx, third_idx, third_idx]}")
    print(f"  PyTorch:   {pytorch_volume_np[0, :, third_idx, third_idx, third_idx]}")

    # Visualize the results
    visualize_specific_results(reference_volume, pytorch_volume_np, volume_size)

    # Test GPU implementations if available
    if CUDA_AVAILABLE and torch.cuda.is_available() and not IS_MACOS:
        print("\nTesting CUDA implementation with specific vertices...")
        vertices_cuda = vertices.cuda()
        features_cuda = features.cuda()

        # Compute result with CUDA implementation
        cuda_volume = grid_feature_projection_trilinear(vertices_cuda, volume_size, features_cuda)
        cuda_volume_np = cuda_volume.detach().cpu().numpy()

        print(f"Center voxel CUDA: {cuda_volume_np[0, :, center_idx, center_idx, center_idx]}")
        print(f"Second vertex voxel CUDA: {cuda_volume_np[0, :, second_idx, second_idx, second_idx]}")
        print(f"Third vertex voxel CUDA: {cuda_volume_np[0, :, third_idx, third_idx, third_idx]}")

    if METAL_AVAILABLE and IS_MACOS:
        print("\nTesting Metal implementation with specific vertices...")

        # Compute result with Metal implementation
        metal_volume = grid_feature_projection_trilinear(vertices.to("mps"), volume_size, features.to("mps"))
        metal_volume_np = metal_volume.detach().cpu().numpy()

        print(f"Center voxel Metal: {metal_volume_np[0, :, center_idx, center_idx, center_idx]}")
        print(f"Second vertex voxel Metal: {metal_volume_np[0, :, second_idx, second_idx, second_idx]}")
        print(f"Third vertex voxel Metal: {metal_volume_np[0, :, third_idx, third_idx, third_idx]}")

def visualize_specific_results(reference_volume, pytorch_volume, volume_size):
    """
    Visualize the results for the specific test case.
    """
    # Create a figure to visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot reference volume
    for c in range(3):  # RGB channels
        axes[0, c].imshow(reference_volume[0, c, :, :, volume_size//2])
        axes[0, c].set_title(f"Reference (Channel {c})")

        axes[1, c].imshow(pytorch_volume[0, c, :, :, volume_size//2])
        axes[1, c].set_title(f"PyTorch (Channel {c})")

    # Save the figure
    plt.tight_layout()
    plt.savefig("grid_projection_specific_test.png")
    print("Saved specific test visualization to grid_projection_specific_test.png")

if __name__ == "__main__":
    print(f"Platform: {platform.system()}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print(f"Metal available: {METAL_AVAILABLE}")

    # Run the general correctness test
    forward_success = test_correctness(volume_size=32, num_vertices=10, num_features=3)

    # Run the specific test case
    test_with_specific_vertices()

    # Run the gradient test
    gradient_success = test_gradient_correctness(volume_size=16)

    if forward_success and gradient_success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)