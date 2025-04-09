import os
import platform
import torch
import torch.nn as nn
from typing import Any
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
mesh2sdf_cuda = None
mesh2sdf_metal = None

# Define CUDA implementation for non-macOS systems
if torch.cuda.is_available() and not IS_MACOS:
    os.environ['TORCH_CUDA_ARCH_LIST'] = "7.0 7.5 8.0 8.6 8.7 8.9 9.0"
    # CUDA implementation
    mesh2sdf_cuda_sources = [
        os.path.join(sources_prefix, "mesh2sdf_cuda.cpp"),
        os.path.join(sources_prefix, "mesh2sdf_cuda_kernel.cu")
    ]
    mesh2sdf_cuda_cflags = ["-O3", "-D WITH_CUDA"]
    CUDA_AVAILABLE = True

    # Try to load the CUDA extension
    try:
        mesh2sdf_cuda = load(
            name="mesh2sdf_cuda",
            sources=mesh2sdf_cuda_sources,
            extra_cflags=mesh2sdf_cuda_cflags,
            verbose=False
        )
        EXTENSION_LOADED = True
        print("CUDA extension for Mesh2SDF loaded successfully")
    except Exception as e:
        print(f"Failed to load CUDA extension: {e}")
        CUDA_AVAILABLE = False

# Metal implementation for macOS
if IS_MACOS:
    try:
        METAL_AVAILABLE = True

        # Define Metal source files
        mesh2sdf_metal_sources = [
            os.path.join(sources_prefix, "mesh2sdf_metal.mm")
        ]

        # Define Metal compilation flags
        mesh2sdf_metal_cflags = ["-O3"]
        mesh2sdf_metal_ldflags = [
            "-framework", "Metal",
            "-framework", "Foundation",
            "-framework", "MetalPerformanceShaders"
        ]

        # Try to load the Metal extension
        try:
            mesh2sdf_metal = load(
                name="mesh2sdf_metal",
                sources=mesh2sdf_metal_sources,
                extra_cflags=mesh2sdf_metal_cflags,
                extra_ldflags=mesh2sdf_metal_ldflags,
                verbose=False
            )
            EXTENSION_LOADED = True
            print("Metal extension for Mesh2SDF loaded successfully")
        except Exception as e:
            print(f"Failed to load Metal extension: {e}")
            METAL_AVAILABLE = False
    except Exception as e:
        print(f"Metal not available: {e}")
        METAL_AVAILABLE = False


class Mesh2SDFFunction(torch.autograd.Function):
    """
    Autograd function for mesh to SDF conversion.
    """
    @staticmethod
    def forward(ctx: Any, vertices: torch.Tensor, faces: torch.Tensor, volume_size: int) -> torch.Tensor:
        if not EXTENSION_LOADED:
            raise RuntimeError("No GPU extension available. Use the PyTorch implementation instead.")

        # Save original dimensions and tensors for backward
        ctx.save_for_backward(vertices, faces)
        ctx.volume_size = volume_size

        # Check device type and use appropriate implementation
        if vertices.is_cuda and CUDA_AVAILABLE:
            # CUDA implementation for NVIDIA GPUs
            sdf_volume = mesh2sdf_cuda.forward(vertices, faces, volume_size)
        elif METAL_AVAILABLE and (vertices.device.type == 'mps'):
            # Metal implementation for Apple Silicon
            sdf_volume = mesh2sdf_metal.forward(vertices, faces, volume_size)
        else:
            raise RuntimeError("No suitable GPU extension available for the current device")

        return sdf_volume


def mesh2sdf(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    volume_size: int
) -> torch.Tensor:
    """
    Convert a mesh to a signed distance field.

    Args:
        vertices: (N, 3) tensor of vertex coordinates (-1 to 1)
        faces: (F, 3) tensor of face indices
        volume_size: Size of the cubic volume grid

    Returns:
        sdf_volume: (D, H, W) SDF volume
    """
    return Mesh2SDFFunction.apply(vertices, faces, volume_size)


class Mesh2SDF(nn.Module):
    """
    PyTorch module for mesh to SDF conversion.
    """
    def __init__(self, volume_size: int):
        super().__init__()
        self._volume_size = volume_size

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        return mesh2sdf(vertices, faces, self._volume_size)


def test_mesh2sdf():
    """
    Test the mesh to SDF conversion.
    """
    # Create a simple cube mesh
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt # pylint: disable=import-outside-toplevel
    import trimesh # pylint: disable=import-outside-toplevel
    import numpy as np # pylint: disable=import-outside-toplevel

    mesh = trimesh.creation.torus(minor_radius=0.25, major_radius=0.75)
    # mesh = trimesh.creation.icosphere(radius=0.5)
    # mesh = trimesh.creation.capsule(radius=0.5, height=0.5)
    # mesh = trimesh.creation.cylinder(radius=0.5, height=0.5)

    volume_size = 64
    vertices = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
    faces = torch.tensor(np.array(mesh.faces), dtype=torch.int32)
    vertices_np = (vertices.cpu().numpy() + 1.0) / 2.0 * (volume_size - 1)

    # Convert to appropriate device
    if torch.cuda.is_available() and not IS_MACOS:
        vertices = vertices.cuda()
        faces = faces.cuda()
        print("Using CUDA")
    elif MPS_AVAILABLE:
        vertices = vertices.to('mps')
        faces = faces.to('mps')
        print("Using MPS")
    else:
        print("Using CPU")

    # Create SDF
    print(f"Vertices shape: {vertices.shape}")
    print(f"Faces shape: {faces.shape}")
    mesh2sdf_ = Mesh2SDF(volume_size)
    sdf = mesh2sdf_(vertices, faces).cpu().numpy()

    print(f"SDF shape: {sdf.shape}")
    print(f"Min SDF value: {sdf.min().item()}")
    print(f"Max SDF value: {sdf.max().item()}")


    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(sdf[volume_size//2, :, :], vmin=-1, vmax=1, cmap="seismic")
    ax[1].imshow(sdf[:, volume_size//2, :], vmin=-1, vmax=1, cmap="seismic")
    ax[2].imshow(sdf[:, :, volume_size//2], vmin=-1, vmax=1, cmap="seismic")
    ax[0].plot(vertices_np[:, 2], vertices_np[:, 1], 'g.', markersize=0.7)
    ax[1].plot(vertices_np[:, 2], vertices_np[:, 0], 'g.', markersize=0.7)
    ax[2].plot(vertices_np[:, 1], vertices_np[:, 0], 'g.', markersize=0.7)
    plt.savefig("mesh2sdf.png")


if __name__ == "__main__":
    test_mesh2sdf()
