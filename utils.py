import os
import subprocess
import torch


def check_for_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes in the repository.
    If there are, print an error message and return True.
    If there are no uncommitted changes, return False.
    """
    # Check if we're in a git repository
    try:
        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            check=True
        )

        if result.stdout.strip():
            print("Error: There are uncommitted changes in the repository.")
            print("Commit your changes or use --ignore_uncommitted_changes to proceed anyway.")
            return True
        return False
    except subprocess.CalledProcessError:
        print("Warning: Not running from a git repository or git is not installed.")
        return True


def binary_voxel_erosion(x: torch.Tensor) -> torch.Tensor:
    """
    Perform one voxel of erosion on a binary voxel image using 3D convolution.

    Parameters
    ----------
    x : torch.Tensor
        Binary voxel image of shape (B, 1, H, W, D) or (B, H, W, D)

    Returns
    -------
    torch.Tensor
        Eroded binary voxel image of same shape as input
    """
    # Ensure input has channel dimension
    if len(x.shape) == 4:
        x = x.unsqueeze(1)

    # Create 3x3x3 kernel for erosion
    kernel = torch.ones(1, 1, 3, 3, 3, device=x.device)

    # Perform 3D convolution
    eroded = torch.nn.functional.conv3d(x, kernel, padding=1)

    # A voxel is kept only if all 27 neighbors (including itself) are 1
    return (eroded == 27).float()
