from typing import Optional

import torch
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.utils import get_laplacian

import trimesh


class GraphLaplacianLoss(torch.nn.Module):
    """
    Graph Laplacian Loss module for mesh vertices.
    """
    def __init__(self, edge_index: torch.Tensor, num_vertices: Optional[int]=None, normalization: str='sym'):
        """
        Initialize the graph Laplacian Loss module using PyTorch Geometric.

        Args:
            edge_index (torch.Tensor): Edge indices of shape (2, E) representing the mesh connectivity.
            normalization (str): Normalization type for the Laplacian ('sym' for symmetric, None for unnormalized).
        """
        super().__init__()

        # Convert to PyG format if needed
        if edge_index.size(0) != 2:
            self.edge_index = edge_index.t().contiguous()
        else:
            self.edge_index = edge_index

        # Determine number of vertices from edge indices
        if num_vertices is None:
            self.num_vertices = int(self.edge_index.max().item() + 1)
        else:
            self.num_vertices = num_vertices

        self.normalization = normalization
        self.cached_lap_edge_index = None
        self.cached_lap_edge_weight = None

        # Always compute and cache the Laplacian
        self._compute_laplacian(self.edge_index, self.num_vertices)

    def _compute_laplacian(self, edge_index, num_nodes):
        """
        Compute the Laplacian edge index and edge weights.

        Args:
            edge_index (torch.Tensor): Edge indices (2, E)
            num_nodes (int): Number of nodes

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Laplacian edge index and edge weights
        """
        # Get the Laplacian using PyTorch Geometric
        lap_edge_index, lap_edge_weight = get_laplacian(
            edge_index,
            normalization=self.normalization,
            num_nodes=num_nodes
        )

        self.cached_lap_edge_index = lap_edge_index
        self.cached_lap_edge_weight = lap_edge_weight

        return lap_edge_index, lap_edge_weight

    def _apply_laplacian(self, x, edge_index, edge_weight=None):
        """
        Apply the Laplacian operator to a feature vector.

        Args:
            x (torch.Tensor): Feature vector of shape (V, C) or (V,)
            edge_index (torch.Tensor): Laplacian edge indices
            edge_weight (torch.Tensor, optional): Laplacian edge weights

        Returns:
            torch.Tensor: Laplacian applied to x
        """

        # Message passing to perform Lx
        row, col = edge_index

        if edge_weight is None:
            edge_weight = torch.ones_like(row, dtype=torch.float)

        # Vectorized implementation that handles all channels at once
        # Gather features for all source nodes and channels: (E, C)
        source_features = x[row]

        # Multiply by weights: (E, C)
        weighted_features = source_features * edge_weight.view(-1, 1)

        # Scatter to targets for all channels at once
        output = torch.zeros_like(x)
        for i in range(x.size(1)):  # Still need this loop since scatter_add_ is per-dimension
            output[:, i].scatter_add_(0, col, weighted_features[:, i])

        return output

    def forward(self, features, target_features=None):
        """
        Compute the graph Laplacian loss on mesh features (can be vertex positions or displacements).

        This function supports two distinct approaches for Laplacian regularization:

        1. Zero Laplacian regularization (when target_features=None):
           - Minimizes ||L·x||² where L is the Laplacian matrix and x is your feature vector
           - Encourages locally smooth features where each vertex is close to the average of its neighbors
           - This is the standard approach for mesh smoothness regularization

        2. Laplacian preservation (when target_features is provided):
           - Minimizes ||L·x - L·y||² where y is your target feature vector
           - Instead of pushing toward zero Laplacian, it preserves the Laplacian coordinates
             of a reference mesh
           - Useful for detail preservation or non-uniform smoothness
           - Helps maintain geometric details from a reference shape while allowing
             for global deformation

        Args:
            features (torch.Tensor): Predicted vertex features of shape (B, V, C)
            target_features (torch.Tensor, optional): Target features of shape (B, V, C).
                                                     If None, uses zero Laplacian regularization.

        Returns:
            torch.Tensor: Laplacian loss
        """
        # Reshape to (B, V, C) for simplicity since N=1 in your case
        batch_size, num_vertices, num_channels = features.shape

        # Use cached Laplacian
        lap_edge_index = self.cached_lap_edge_index
        lap_edge_weight = self.cached_lap_edge_weight

        # Process all batches
        total_loss = 0
        for b in range(batch_size):
            feat = features[b]  # (V, C)

            # Apply Laplacian to features
            lap_feat = self._apply_laplacian(feat, lap_edge_index, lap_edge_weight)

            if target_features is not None:
                # Laplacian preservation approach
                target_feat = target_features.reshape(
                    batch_size, num_vertices, num_channels)[b]
                target_lap_feat = self._apply_laplacian(
                    target_feat, lap_edge_index, lap_edge_weight)
                # Compare Laplacians
                batch_loss = F.mse_loss(lap_feat, target_lap_feat)
            else:
                # Zero Laplacian regularization approach
                batch_loss = torch.mean(lap_feat ** 2)

            total_loss += batch_loss

        # Average across batches
        return total_loss / batch_size


def example_usage(device="cpu"):
    """
    Example usage of GraphLaplacianLoss with a real sphere mesh from trimesh.
    Demonstrates the effect of the graph Laplacian loss on clean vs. noisy meshes.
    """
    # Create a sphere mesh using trimesh
    torch.manual_seed(0)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

    device = torch.device(device)

    # Extract vertices and edges
    vertices = torch.tensor(sphere.vertices, dtype=torch.float32, device=device)
    edge_index = []
    for edge in sphere.edges_unique:  # pylint: disable=not-an-iterable
        edge_index.append([edge[0], edge[1]])

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()

    # Initialize the Laplacian loss
    lap_loss = GraphLaplacianLoss(
        edge_index=edge_index,
        num_vertices=vertices.shape[0],
        normalization=None  # Use symmetric normalization
    )

    # Create a batch with the original sphere vertices
    batch_size = 2

    # Create two versions: original and with noise
    clean_vertices = vertices[None,].repeat(batch_size, 1, 1)

    # Add random noise to create noisy vertices
    noise_level = 0.05
    noise = torch.randn_like(clean_vertices) * noise_level
    noisy_vertices = clean_vertices + noise

    # Calculate Laplacian loss for both
    clean_loss = lap_loss(clean_vertices)
    noisy_loss = lap_loss(noisy_vertices)

    print(f"Laplacian loss for clean sphere: {clean_loss.item():.6f}")
    print(f"Laplacian loss for noisy sphere: {noisy_loss.item():.6f}")
    print(f"Ratio (noisy/clean): {noisy_loss.item() / clean_loss.item():.2f}x")

    # test with targets
    target_loss_clean = lap_loss(clean_vertices, clean_vertices)
    target_loss_noisy = lap_loss(noisy_vertices, clean_vertices)
    print(f"Laplacian loss for clean sphere with target: {target_loss_clean.item():.6f}")
    print(f"Laplacian loss for noisy sphere with target: {target_loss_noisy.item():.6f}")


def example_usage_mesh_laplacian(device="cpu"):
    """
    Example usage of MeshLaplacianLoss with a real sphere mesh.
    Demonstrates the effect of the mesh Laplacian loss on clean vs. deformed meshes.
    """
    # Create a sphere mesh using trimesh
    torch.manual_seed(0)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

    device = torch.device(device)

    # Extract vertices and faces
    vertices = torch.tensor(sphere.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(sphere.faces, dtype=torch.long, device=device).t()  # Convert to (3, F) format

    # Initialize the Mesh Laplacian loss
    mesh_lap_loss = MeshLaplacianLoss(
        faces=faces,
        num_vertices=vertices.shape[0],
        normalization='cotangens'  # Use cotangent weights
    )

    # Create a batch with the original sphere vertices
    batch_size = 2
    clean_vertices = vertices[None,].repeat(batch_size, 1, 1)

    # Add random noise to create deformed vertices
    noise_level = 0.05
    noise = torch.randn_like(clean_vertices) * noise_level
    deformed_vertices = clean_vertices + noise

    # Calculate Mesh Laplacian loss for both
    clean_loss = mesh_lap_loss(clean_vertices)
    deformed_loss = mesh_lap_loss(deformed_vertices)

    print("\nMesh Laplacian Loss Results:")
    print(f"Mesh Laplacian loss for clean sphere: {clean_loss.item():.6f}")
    print(f"Mesh Laplacian loss for deformed sphere: {deformed_loss.item():.6f}")
    print(f"Ratio (deformed/clean): {deformed_loss.item() / clean_loss.item():.2f}x")

    # Test with targets
    target_loss_clean = mesh_lap_loss(clean_vertices, clean_vertices)
    target_loss_deformed = mesh_lap_loss(deformed_vertices, clean_vertices)
    print(f"Mesh Laplacian loss for clean sphere with target: {target_loss_clean.item():.6f}")
    print(f"Mesh Laplacian loss for deformed sphere with target: {target_loss_deformed.item():.6f}")


if __name__ == "__main__":
    example_usage()
    example_usage_mesh_laplacian()
