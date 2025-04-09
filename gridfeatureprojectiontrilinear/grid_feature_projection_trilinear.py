import os
import sys
import platform
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Callable
from torch.utils.cpp_extension import load

# Get the directory of the current file
sources_prefix = os.path.dirname(os.path.abspath(__file__))

# Check for MPS availability
IS_MACOS = platform.system() == 'Darwin'
MPS_AVAILABLE = IS_MACOS and hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# Initialize flags
CUDA_AVAILABLE = False
METAL_AVAILABLE = False
EXTENSION_LOADED = False
grid_feature_projection_cuda: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]] = None
grid_feature_projection_metal: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]] = None

# Define source files and flags for CUDA
if torch.cuda.is_available() and not IS_MACOS:
    os.environ['TORCH_CUDA_ARCH_LIST'] = "7.0 7.5 8.0 8.6 8.7 8.9 9.0"
    # CUDA implementation
    grid_feature_projection_sources = [
        os.path.join(sources_prefix, "grid_feature_projection_cuda.cpp")
    ]
    grid_feature_projection_cflags = ["-O3"]
    grid_feature_projection_cflags.append("-D WITH_CUDA")
    grid_feature_projection_sources.append(
        os.path.join(sources_prefix, "grid_feature_projection_cuda_kernel.cu")
    )
    CUDA_AVAILABLE = True

    # Try to load the CUDA extension
    try:
        grid_feature_projection_cuda = load(
            name="grid_feature_projection_cuda",
            sources=grid_feature_projection_sources,
            extra_cflags=grid_feature_projection_cflags,
            verbose=False
        )
        EXTENSION_LOADED = True
        print("CUDA extension for GridFeatureProjectionTrilinear loaded successfully")
    except Exception as e:
        print(f"Failed to load CUDA extension: {e}")
        CUDA_AVAILABLE = False
        EXTENSION_LOADED = False
elif IS_MACOS:
    # Metal implementation for macOS
    try:
        # Check if Metal is available
        import subprocess
        result = subprocess.run(["xcrun", "--find", "metal"], capture_output=True, text=True)
        if result.returncode == 0:
            METAL_AVAILABLE = True

            # Define Metal source files
            grid_feature_projection_metal_sources = [
                os.path.join(sources_prefix, "grid_feature_projection_metal.mm")
            ]

            # Define Metal compilation flags
            grid_feature_projection_metal_cflags = ["-O3"]
            grid_feature_projection_metal_ldflags = [
                "-framework", "Metal",
                "-framework", "Foundation",
                "-framework", "MetalPerformanceShaders"
            ]

            # Try to load the Metal extension
            try:
                grid_feature_projection_metal = load(
                    name="grid_feature_projection_metal",
                    sources=grid_feature_projection_metal_sources,
                    extra_cflags=grid_feature_projection_metal_cflags,
                    extra_ldflags=grid_feature_projection_metal_ldflags,
                    verbose=False
                )
                EXTENSION_LOADED = True
                print("Metal extension for GridFeatureProjectionTrilinear loaded successfully")
            except Exception as e:
                print(f"Failed to load Metal extension: {e}")
                METAL_AVAILABLE = False
                EXTENSION_LOADED = False
    except Exception as e:
        print(f"Metal not available: {e}")
        METAL_AVAILABLE = False
else:
    print("Neither CUDA nor Metal is available. Using PyTorch implementation.")


class GridFeatureProjectionTrilinearFunction(torch.autograd.Function):
    """
    Autograd function for grid feature projection with trilinear interpolation.
    """
    @staticmethod
    def forward(ctx: Any, vertices: torch.Tensor, features: torch.Tensor, volume_size: int) -> torch.Tensor:
        if not EXTENSION_LOADED:
            raise RuntimeError("No GPU extension available. Use the PyTorch implementation instead.")

        # Add batch dimension if not present
        original_dim = vertices.dim()
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
            features = features.unsqueeze(0)

        # Save original dimensions and tensors for backward
        ctx.save_for_backward(vertices, features)
        ctx.volume_size = volume_size
        ctx.original_dim = original_dim

        # Check device type and use appropriate implementation
        if vertices.is_cuda and CUDA_AVAILABLE:
            # CUDA implementation for NVIDIA GPUs
            volume = grid_feature_projection_cuda.forward(vertices, features, volume_size)
        elif METAL_AVAILABLE and (vertices.device.type == 'mps'):
            # Metal implementation for Apple Silicon
            volume = grid_feature_projection_metal.forward(vertices, features, volume_size)
        else:
            raise RuntimeError("No suitable GPU extension available for the current device")

        return volume

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        vertices, features = ctx.saved_tensors
        volume_size = ctx.volume_size
        original_dim = ctx.original_dim

        # Ensure grad_output has the right shape
        if grad_output.dim() != vertices.dim() + 2:  # vertices is [B, N, 3], grad_output should be [B, C, D, H, W]
            raise RuntimeError(f"Expected grad_output to have {vertices.dim() + 2} dimensions, got {grad_output.dim()}")

        if vertices.is_cuda and CUDA_AVAILABLE:
            # CUDA implementation for NVIDIA GPUs
            grad_vertices, grad_features = grid_feature_projection_cuda.backward(
                grad_output, vertices, features, volume_size)
        elif METAL_AVAILABLE and (vertices.device.type == 'mps'):
            # Metal implementation for Apple Silicon
            grad_vertices, grad_features = grid_feature_projection_metal.backward(
                grad_output, vertices, features, volume_size)
        else:
            raise RuntimeError("No suitable GPU extension available for the current device")

        # Remove batch dimension if the original input didn't have it
        if original_dim == 2:
            grad_vertices = grad_vertices.squeeze(0)
            grad_features = grad_features.squeeze(0)

        return grad_vertices, grad_features, None


def grid_feature_projection_trilinear_pytorch(
    vertices: torch.Tensor,
    volume_size: int,
    features: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    PyTorch implementation of grid feature projection with trilinear interpolation.
    This is a fallback for when the GPU extensions are not available.

    Args:
        vertices: (B, N, 3) or (N, 3) tensor of vertex coordinates (-1 to 1)
        volume_size: Size of the cubic volume grid
        features: (B, N, C) or (N, C) tensor of vertex features

    Returns:
        volume: (B, C, D, H, W) or (1, C, D, H, W) feature volume
    """
    # Handle batch dimension
    original_dim = vertices.dim()
    if original_dim == 2:
        vertices = vertices.unsqueeze(0)
        if features is None:
            features = vertices
        else:
            features = features.unsqueeze(0)
    elif features is None:
        features = vertices

    # Ensure batch dimensions match
    assert vertices.dim() == 3, f"Expected vertices to have 3 dimensions (B, N, 3), got {vertices.shape}"
    assert features.dim() == 3, f"Expected features to have 3 dimensions (B, N, C), got {features.shape}"
    assert vertices.size(0) == features.size(0), f"Batch size mismatch: vertices {vertices.size(0)} vs features {features.size(0)}"

    batch_size = vertices.size(0)
    feature_dim = features.shape[2]

    # Create a feature volume for each batch
    volumes = []

    for b in range(batch_size):
        # Create a feature volume for this batch
        volume = torch.zeros(
            (1, feature_dim, volume_size, volume_size, volume_size),
            device=vertices.device
        )

        # Get vertices and features for this batch
        batch_vertices = vertices[b]
        batch_features = features[b]

        # Scale vertices to [-1, 1] if they aren't already
        scaled_vertices = batch_vertices.clamp(-1, 1)

        # Scale vertex coordinates to volume indices (floating point)
        indices = ((scaled_vertices + 1) * (volume_size - 1) / 2)

        # For each vertex, compute the 8 surrounding voxels and weights
        floor_indices = torch.floor(indices).long().clamp(0, volume_size - 2)
        ceil_indices = (floor_indices + 1).clamp(0, volume_size - 1)

        # Compute interpolation weights
        weights = indices - floor_indices.float()

        # For each vertex, distribute its features to the 8 surrounding voxels
        for i in range(batch_vertices.shape[0]):
            # Get the 8 corner indices
            x0, y0, z0 = floor_indices[i]
            x1, y1, z1 = ceil_indices[i]

            # Get the interpolation weights
            wx, wy, wz = weights[i]

            # Compute the 8 weights for trilinear interpolation
            w000 = (1 - wx) * (1 - wy) * (1 - wz)
            w001 = (1 - wx) * (1 - wy) * wz
            w010 = (1 - wx) * wy * (1 - wz)
            w011 = (1 - wx) * wy * wz
            w100 = wx * (1 - wy) * (1 - wz)
            w101 = wx * (1 - wy) * wz
            w110 = wx * wy * (1 - wz)
            w111 = wx * wy * wz

            # Get the feature vector for this vertex
            feature = batch_features[i]

            # Add weighted features to the 8 surrounding voxels
            volume[0, :, x0, y0, z0] += feature * w000
            volume[0, :, x0, y0, z1] += feature * w001
            volume[0, :, x0, y1, z0] += feature * w010
            volume[0, :, x0, y1, z1] += feature * w011
            volume[0, :, x1, y0, z0] += feature * w100
            volume[0, :, x1, y0, z1] += feature * w101
            volume[0, :, x1, y1, z0] += feature * w110
            volume[0, :, x1, y1, z1] += feature * w111

        volumes.append(volume)

    # Stack along batch dimension if we have multiple batches
    if batch_size > 1:
        return torch.cat(volumes, dim=0)
    else:
        # If the original input didn't have a batch dimension, keep the output consistent
        if original_dim == 2:
            return volumes[0]
        else:
            return volumes[0]


def grid_feature_projection_trilinear(
    vertices: torch.Tensor,
    volume_size: int,
    features: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Project mesh vertex features into a regular 3D grid using trilinear interpolation.
    Uses GPU implementation if available, otherwise falls back to PyTorch.

    Args:
        vertices: (B, N, 3) or (N, 3) tensor of vertex coordinates (-1 to 1)
        volume_size: Size of the cubic volume grid
        features: (B, N, C) or (N, C) tensor of vertex features

    Returns:
        volume: (B, C, D, H, W) or (1, C, D, H, W) feature volume
    """
    if features is None:
        features = vertices

    # Add batch dimension if not present
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
        features = features.unsqueeze(0)

    # Ensure batch dimensions match
    assert vertices.dim() == 3, f"Expected vertices to have 3 dimensions (B, N, 3), got {vertices.shape}"
    assert features.dim() == 3, f"Expected features to have 3 dimensions (B, N, C), got {features.shape}"
    assert vertices.size(0) == features.size(0), f"Batch size mismatch: vertices {vertices.size(0)} vs features {features.size(0)}"

    try:
        if EXTENSION_LOADED and (
            (vertices.is_cuda and CUDA_AVAILABLE) or
            (vertices.device.type == 'mps' and METAL_AVAILABLE)
        ):
            # Use the autograd function to ensure gradients are properly tracked
            return GridFeatureProjectionTrilinearFunction.apply(vertices, features, volume_size)
    except Exception as e:
        print(f"GPU implementation failed, falling back to PyTorch: {e}")

    # Fall back to PyTorch implementation
    batch_size = vertices.size(0)
    volumes = []

    for b in range(batch_size):
        volume_b = grid_feature_projection_trilinear_pytorch(
            vertices[b], volume_size, features[b]
        )
        volumes.append(volume_b)

    # Stack along batch dimension
    if batch_size > 1:
        return torch.cat(volumes, dim=0)
    else:
        return volumes[0]


class GridFeatureProjectionTrilinear(nn.Module):
    """
    Module wrapper for grid feature projection with trilinear interpolation.
    """
    def __init__(self, volume_size):
        super().__init__()
        self.volume_size = volume_size

    def forward(self, vertices, features=None):
        """
        Project mesh vertex features into a regular 3D grid using trilinear interpolation

        Args:
            vertices: (B, N, 3) or (N, 3) tensor of vertex coordinates (-1 to 1)
            features: (B, N, C) or (N, C) tensor of vertex features

        Returns:
            volume: (B, C, D, H, W) or (1, C, D, H, W) feature volume
        """
        return grid_feature_projection_trilinear(vertices, self.volume_size, features)


def test_grid_projection_trilinear():
    """Test the trilinear feature projection"""
    import numpy as np
    import trimesh
    from matplotlib import pyplot as plt

    # Check available implementations
    print(f"Platform: {platform.system()}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print(f"Metal available: {METAL_AVAILABLE}")
    print(f"MPS available: {MPS_AVAILABLE}")
    print(f"Extension loaded: {EXTENSION_LOADED}")

    # Create a test mesh
    mesh = trimesh.creation.torus(major_radius=0.5, minor_radius=0.25, major_sections=32, minor_sections=32)
    vertices = torch.from_numpy(mesh.vertices).float()
    normals = torch.from_numpy(np.array(mesh.vertex_normals)).float()
    features = torch.concat((torch.ones_like(vertices), vertices, normals), dim=1)

    # Move to appropriate device
    if torch.cuda.is_available() and not IS_MACOS:
        vertices = vertices.cuda()
        features = features.cuda()
        print("Using CUDA")
    elif MPS_AVAILABLE:
        vertices = vertices.to('mps')
        features = features.to('mps')
        print("Using MPS")
    else:
        print("Using CPU")

    # Create projection module
    size = 128
    proj = GridFeatureProjectionTrilinear(volume_size=size)

    # Project features
    with torch.no_grad():
        volume = proj(vertices, features)

    # Move back to CPU for visualization
    volume_np = volume.detach().cpu().numpy()
    vertices_np = (vertices.cpu().numpy() + 1.0) * (size - 1) / 2

    # Visualize
    fig, ax = plt.subplots(3, 3, figsize=(12, 10))

    # Now the visualization should match the SDF implementation without rotation or flipping
    ax[0, 0].imshow(volume_np[0, 0, size//2, :, :], cmap='gray')
    ax[0, 0].plot(vertices_np[:, 2], vertices_np[:, 1], 'r.', alpha=0.5)
    ax[0, 1].imshow(volume_np[0, 1, size//2, :, :], cmap='gray')
    ax[0, 2].imshow(volume_np[0, 2, size//2, :, :], cmap='gray')

    ax[1, 0].imshow(volume_np[0, 3, :, size//2, :], cmap='gray')
    ax[1, 0].plot(vertices_np[:, 2], vertices_np[:, 0], 'r.', alpha=0.5)
    ax[1, 1].imshow(volume_np[0, 4, :, size//2, :], cmap='gray')
    ax[1, 2].imshow(volume_np[0, 5, :, size//2, :], cmap='gray')

    ax[2, 0].imshow(volume_np[0, 6, :, :, size//2], cmap='gray')
    ax[2, 0].plot(vertices_np[:, 1], vertices_np[:, 0], 'r.', alpha=0.5)
    ax[2, 1].imshow(volume_np[0, 7, :, :, size//2], cmap='gray')
    ax[2, 2].imshow(volume_np[0, 8, :, :, size//2], cmap='gray')

    fig.savefig("grid_projection_trilinear.png")
    plt.show()
    print("Saved visualization to grid_projection_trilinear.png")


if __name__ == "__main__":
    test_grid_projection_trilinear()
