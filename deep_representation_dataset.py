import os

import numpy as np
import torch
import trimesh

from sdffunctions import Mesh2SDF
from gcn import create_edge_index


class DeepRepresentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        volume_size: int,
        subdivisions: int = 4,
        augmentation_frequency: float = 0.9,
        advanced_augmentation_frequency: float = 0.9,
        noise_augmentation_frequency: float = 0.1,
        augmentation_magnitude: float = 1.0,
        num_control_points: int = 16,
        max_displacement: float = 0.2,
        device: str | torch.device = "cpu",
        repeat: int = 1,
    ):
        """Initialize the Mesh2VoxelDataset for training deep SDF models.

        This dataset creates meshes (spheres of different sizes) and their corresponding
        signed distance fields (SDFs). It provides various augmentation methods including:
        1. Basic affine transformations (rotation, scaling, translation, shearing)
        2. Advanced deformations using control points with nonlinear functions
        3. B-spline based smooth displacement fields
        4. Random noise perturbations

        Parameters
        ----------
        volume_size : int
            Size of the cubic volume to generate (volume_size x volume_size x volume_size)
        subdivisions : int, optional
            Number of subdivisions for the icosphere, by default 4
        augmentation_frequency : float, optional
            Probability of applying basic affine augmentations, by default 0.9
        advanced_augmentation_frequency : float, optional
            Probability of applying advanced deformation augmentations, by default 0.9
        noise_augmentation_frequency : float, optional
            Probability of applying random noise perturbations, by default 0.1
        augmentation_magnitude : float, optional
            Scaling factor for all augmentation intensities, by default 1.0
        num_control_points : int, optional
            Number of control points for advanced deformation, by default 16
        max_displacement : float, optional
            Maximum displacement magnitude for advanced deformations, by default 0.2
        device : str or torch.device, optional
            Device to run computations on, by default "cpu"
        repeat: int, optional
            Number of times to repeat the dataset, by default 1

        Notes
        -----
        The dataset generates four icospheres with different radii (0.95, 0.75, 0.5, 0.25)
        and computes their signed distance fields. These can be augmented with various
        deformation methods to create a diverse training set for deep SDF models.

        The advanced augmentation methods include:
        - Control point displacement: Places control points in the volume and applies
          sinusoidal displacement fields with varying frequencies and magnitudes
        - B-spline displacement: Uses a 4x4x4 grid of control points with random
          displacements and applies B-spline interpolation for smooth deformation

        Examples
        --------
        >>> dataset = Voxel2MeshDataset(
        ...     volume_size=64,
        ...     device="cuda",
        ...     advanced_augmentation_frequency=1.0,
        ...     max_displacement=0.3
        ... )
        >>> sdf, vertices, faces, edge_index = dataset[0]
        """

        super().__init__()
        self._augmentation_frequency = augmentation_frequency
        self._advanced_augmentation_frequency = advanced_augmentation_frequency
        self._noise_augmentation_frequency = noise_augmentation_frequency
        self._augmentation_magnitude = augmentation_magnitude
        self._device = torch.device(device)
        self._volume_size = volume_size
        self._mesh2sdf = Mesh2SDF(volume_size)
        self._num_control_points = num_control_points
        self._max_displacement = max_displacement
        self._repeat = repeat

        # Pre-compute grid for faster B-spline interpolation
        grid_size = 4  # 4x4x4 control grid for B-splines
        self._grid_coords = torch.linspace(-1, 1, grid_size)

        # Create a meshgrid for the control points
        grid_x, grid_y, grid_z = torch.meshgrid(
            self._grid_coords, self._grid_coords, self._grid_coords, indexing='ij'
        )
        self._grid_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1).to(self._device)

        self._meshes: list[trimesh.Trimesh] = []
        self._meshes.append(trimesh.creation.icosphere(radius=0.8, subdivisions=subdivisions))
        self._meshes.append(trimesh.creation.icosphere(radius=0.7, subdivisions=subdivisions))
        self._meshes.append(trimesh.creation.icosphere(radius=0.6, subdivisions=subdivisions))
        self._meshes.append(trimesh.creation.icosphere(radius=0.5, subdivisions=subdivisions))
        self._meshes.append(trimesh.creation.icosphere(radius=0.4, subdivisions=subdivisions))
        self._meshes.append(trimesh.creation.icosphere(radius=0.3, subdivisions=subdivisions))
        self._meshes.append(trimesh.creation.icosphere(radius=0.2, subdivisions=subdivisions))

        self._vertices = []
        self._faces = []
        self._edge_indices = []
        for mesh in self._meshes:
            self._vertices.append(torch.from_numpy(mesh.vertices).float())
            self._faces.append(torch.from_numpy(mesh.faces).to(torch.int32))
            self._edge_indices.append(create_edge_index(self._vertices[-1], self._faces[-1]))

    def __len__(self) -> int:
        return len(self._meshes) * self._repeat

    @property
    def volume_size(self) -> int:
        return self._volume_size

    def _generate_control_point_displacement(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Generate a smooth displacement field using control points with nonlinear functions.
        Optimized version using vectorized operations.

        Args:
            vertices: Tensor of shape [N, 3] containing vertex coordinates

        Returns:
            Tensor of shape [N, 3] containing displacement vectors for each vertex
        """
        # Create control points in a grid pattern
        grid_size = int(np.ceil(self._num_control_points ** (1/3)))

        # Generate control points in normalized space [-1, 1]
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    pos = torch.tensor(
                        [
                            i / (grid_size - 1) * 2 - 1,
                            j / (grid_size - 1) * 2 - 1,
                            k / (grid_size - 1) * 2 - 1
                        ],
                        device=self._device
                    )
                    positions.append(pos)

        # Limit to desired number of control points and add jitter
        positions = torch.stack(positions[:self._num_control_points])
        jitter = (torch.rand_like(positions) * 0.2 - 0.1)
        control_points = positions + jitter

        # Generate random parameters for displacement functions
        freqs = torch.rand(self._num_control_points, 3, device=self._device) * 2 + 0.5  # 0.5 to 2.5
        phases = torch.rand(self._num_control_points, 3, device=self._device) * 2 * np.pi
        magnitudes = (torch.rand(self._num_control_points, 3, device=self._device) * 2 - 1) * self._max_displacement

        # Compute all pairwise distances between vertices and control points
        # Shape: [num_vertices, num_control_points]
        vertices_expanded = vertices.unsqueeze(1)  # [N, 1, 3]
        control_points_expanded = control_points.unsqueeze(0)  # [1, num_control_points, 3]
        dists = torch.norm(vertices_expanded - control_points_expanded, dim=2)

        # Compute weights based on distance
        # Shape: [num_vertices, num_control_points]
        weights = 1.0 / (1.0 + dists * dists * 5)
        weights_sum = weights.sum(dim=1, keepdim=True)
        normalized_weights = weights / (weights_sum + 1e-8)

        # Compute displacements for each control point influence
        # Shape: [num_vertices, num_control_points, 3]
        dists_expanded = dists.unsqueeze(-1)  # [num_vertices, num_control_points, 1]
        freqs_expanded = freqs.unsqueeze(0)  # [1, num_control_points, 3]
        phases_expanded = phases.unsqueeze(0)  # [1, num_control_points, 3]
        magnitudes_expanded = magnitudes.unsqueeze(0)  # [1, num_control_points, 3]

        # Compute sin waves with broadcasting
        sin_values = torch.sin(freqs_expanded * dists_expanded + phases_expanded) * magnitudes_expanded

        # Apply weights and sum
        # Shape: [num_vertices, num_control_points, 3] -> [num_vertices, 3]
        normalized_weights_expanded = normalized_weights.unsqueeze(-1)  # [num_vertices, num_control_points, 1]
        vertex_displacements = (sin_values * normalized_weights_expanded).sum(dim=1)

        return vertex_displacements

    def _generate_bspline_displacement_field(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Generate a smooth displacement field using 3D B-splines.
        Optimized version using PyTorch's grid_sample.

        Args:
            vertices: Tensor of shape [N, 3] containing vertex coordinates

        Returns:
            Tensor of shape [N, 3] containing displacement vectors for each vertex
        """
        # Define control grid size (coarse grid for B-spline control points)
        grid_size = 4  # 4x4x4 control grid

        # Create control points with random displacements
        control_displacements = torch.randn(1, 3, grid_size, grid_size, grid_size, device=self._device) * self._max_displacement

        # Normalize vertices to [-1, 1] for grid_sample
        # PyTorch's grid_sample expects coordinates in range [-1, 1]
        vertices_normalized = vertices.clone()  # Already in [-1, 1] range

        # Reshape vertices for grid_sample: [N, 3] -> [1, N, 1, 1, 3]
        grid_vertices = vertices_normalized.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # Use grid_sample to interpolate displacements
        # Mode 'bilinear' is faster than 'bicubic' and still gives smooth results
        displacements = torch.nn.functional.grid_sample(
            control_displacements,
            grid_vertices,
            mode='bilinear',  # Use bilinear for speed
            padding_mode='border',
            align_corners=True
        )

        # Reshape output: [1, 3, 1, 1, N] -> [N, 3]
        displacements = displacements.squeeze().permute(1, 0)

        return displacements

    def _advanced_augment(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Apply advanced augmentation using control points and smooth displacement fields.

        Args:
            vertices: Tensor of shape [N, 3] containing vertex coordinates

        Returns:
            Augmented vertices
        """
        # Choose between different augmentation methods
        method = torch.randint(0, 2, (1,)).item()

        if method == 0:
            # Control point based displacement
            displacements = self._generate_control_point_displacement(vertices)
        else:
            # B-spline based displacement
            displacements = self._generate_bspline_displacement_field(vertices)

        # Apply displacements
        augmented_vertices = vertices + displacements

        return augmented_vertices

    def _augment(
        self,
        vertices: torch.Tensor,
        augmentation_magnitude: float = 1.0,
    ):
        if (torch.rand(1) > (1-self._noise_augmentation_frequency)).item():
            vertices = vertices + torch.randn_like(vertices) * 0.005 * augmentation_magnitude

        # Use advanced augmentation if enabled
        if (torch.rand(1) > (1-self._advanced_augmentation_frequency)).item():
            vertices = self._advanced_augment(vertices)

        if (torch.rand(1) > (1-self._augmentation_frequency)).item():
            # translation
            trans_x = (torch.rand(1)*2-1) * 0.2 * augmentation_magnitude
            trans_z = (torch.rand(1)*2-1) * 0.2 * augmentation_magnitude
            trans_y = (torch.rand(1)*2-1) * 0.2 * augmentation_magnitude

            # rotation
            angle_x = (torch.rand(1)*2-1) * 0.2 * augmentation_magnitude
            angle_y = (torch.rand(1)*2-1) * 0.2 * augmentation_magnitude
            angle_z = (torch.rand(1)*2-1) * 0.2 * augmentation_magnitude

            # scaling
            scale_x = (torch.rand(1)*2-1) * 0.2 * augmentation_magnitude
            scale_y = (torch.rand(1)*2-1) * 0.2 * augmentation_magnitude
            scale_z = (torch.rand(1)*2-1) * 0.2 * augmentation_magnitude

            # shearing
            shear_xy = (torch.rand(1)*2-1) * 0.1 * augmentation_magnitude
            shear_xz = (torch.rand(1)*2-1) * 0.1 * augmentation_magnitude
            shear_yx = (torch.rand(1)*2-1) * 0.1 * augmentation_magnitude
            shear_yz = (torch.rand(1)*2-1) * 0.1 * augmentation_magnitude
            shear_zx = (torch.rand(1)*2-1) * 0.1 * augmentation_magnitude
            shear_zy = (torch.rand(1)*2-1) * 0.1 * augmentation_magnitude

            theta1_s = torch.tensor([
                [1*(1+scale_x), 0, 0, 0],
                [0, 1*(1+scale_y), 0, 0],
                [0, 0, 1*(1+scale_z), 0],
                [0, 0, 0, 1]
            ])

            theta1_shear = torch.tensor([
                [1, shear_xy, shear_xz, 0],
                [shear_yx, 1, shear_yz, 0],
                [shear_zx, shear_zy, 1, 0],
                [0, 0, 0, 1]
            ])

            theta1_x = torch.tensor([
                [1, 0, 0, 0],
                [0, torch.cos(angle_x), -torch.sin(angle_x), 0],
                [0, torch.sin(angle_x),  torch.cos(angle_x), 0],
                [0, 0, 0, 1]
            ])
            theta1_y = torch.tensor([
                [torch.cos(angle_y), 0,  torch.sin(angle_y), 0],
                [0, 1, 0, 0],
                [-torch.sin(angle_y), 0, torch.cos(angle_y), 0],
                [0, 0, 0, 1]
            ])
            theta1_z = torch.tensor([
                [torch.cos(angle_z), -torch.sin(angle_z), 0, 0],
                [torch.sin(angle_z),  torch.cos(angle_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            theta1_trans = torch.tensor([
                [1, 0, 0, trans_x],
                [0, 1, 0, trans_y],
                [0, 0, 1, trans_z],
                [0, 0, 0, 1]
            ])

            theta1_ = theta1_trans @ theta1_z @ theta1_y @ theta1_x @ theta1_shear @ theta1_s
            theta1 = theta1_[:3][None,]

            theta2 = torch.linalg.inv(theta1_)
            theta2 = theta2.to(vertices.device)
            vertices = (torch.nn.functional.pad(vertices, pad=(0,1), value=1.0) @ theta2.T)[:,:3]

        return vertices.contiguous()

    def __getitem__(self, idx: int):
        idx = idx % len(self._meshes)
        vertices = self._vertices[idx].to(self._device)
        faces = self._faces[idx].to(self._device)
        edge_index = self._edge_indices[idx].to(self._device)
        with torch.no_grad():
            vertices = self._augment(vertices, self._augmentation_magnitude)
            sdf = self._mesh2sdf(vertices, faces)[None,]

        return sdf, vertices, faces, edge_index


def smoke_test():
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    plot_out_dir = "./plot_test"
    os.makedirs(plot_out_dir, exist_ok=True)

    dataset = DeepRepresentationDataset(
        volume_size=64,
        subdivisions=4,
        augmentation_frequency=0.9,
        advanced_augmentation_frequency=0.9,
        noise_augmentation_frequency=0.1,
        num_control_points=16,
        max_displacement=0.2,
        device="mps",
        repeat=2
    )

    if len(dataset) == 0:
        raise RuntimeError("No images found")

    print("len(dataset):", len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for num_img, (sdf, vertices, faces, edge_index) in enumerate(dataloader):
        sdf = sdf.detach().cpu().numpy()
        print("seg.shape:", sdf.shape)

        print("seg", sdf.min(), sdf.max())

        print("vertices.shape:", vertices.shape)
        print("faces.shape:", faces.shape)
        print("edge_index.shape:", edge_index.shape)

        fig, ax = plt.subplots(3, 2, figsize=(6, 8))
        s = dataset.volume_size//2
        ax[0,0].imshow(sdf[0, 0, s], vmin=-1, vmax=1, interpolation=None, cmap="seismic")
        ax[1,0].imshow(sdf[0, 0, :, s], vmin=-1, vmax=1, interpolation=None, cmap="seismic")
        ax[2,0].imshow(sdf[0, 0, :, :, s], vmin=-1, vmax=1, interpolation=None, cmap="seismic")

        ax[0,1].imshow(sdf[0, 0, s], vmin=-1, vmax=1, interpolation=None, cmap="seismic")
        ax[1,1].imshow(sdf[0, 0, :, s], vmin=-1, vmax=1, interpolation=None, cmap="seismic")
        ax[2,1].imshow(sdf[0, 0, :, :, s], vmin=-1, vmax=1, interpolation=None, cmap="seismic")

        mesh = trimesh.Trimesh(
            vertices=(vertices[0].detach().cpu().numpy() + 1) * (dataset.volume_size-1)/2,
            faces=faces[0].detach().cpu().numpy(),
            process=False
        )

        ax[0,0].plot(mesh.vertices[:,2], mesh.vertices[:,1], 'k.', markersize=1, alpha=0.5)
        ax[1,0].plot(mesh.vertices[:,2], mesh.vertices[:,0], 'k.', markersize=1, alpha=0.5)
        ax[2,0].plot(mesh.vertices[:,1], mesh.vertices[:,0], 'k.', markersize=1, alpha=0.5)

        fig.tight_layout()
        fig.savefig(os.path.join(plot_out_dir, f"{num_img}.png"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    smoke_test()
