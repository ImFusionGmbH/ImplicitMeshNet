import os
from glob import glob
from typing import Optional, Sequence

import numpy as np
import SimpleITK as sitk
import torch


def apply_window_level(image: torch.Tensor, window: float, level: float) -> torch.Tensor:
    """Apply window/level to CT image.

    Args:
        image: Input image tensor
        window: Window width (range of HU values to show)
        level: Window center (center HU value)

    Returns:
        Normalized image tensor in range [-1, 1]
    """
    min_value = level - window / 2

    # Linear mapping from [min_value, max_value] to [0, 1]
    image_normalized = (image - min_value) / window
    image_normalized = image_normalized.clip(0, 1)

    # Map from [0, 1] to [-1, 1]
    image_normalized = image_normalized * 2 - 1
    return image_normalized


def resample_image(
    sitk_image: sitk.Image,
    target_spacing: tuple[float, float, float],
    interpolator: int
) -> sitk.Image:
    """Resample image to target spacing using linear interpolation."""
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    # Calculate new size based on target spacing
    new_size = [int(round(original_size[i] * (original_spacing[i] / target_spacing[i]))) for i in range(3)]

    # Create resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(sitk_image.GetPixelIDValue())
    resample.SetInterpolator(interpolator)

    return resample.Execute(sitk_image)


class ImplicitMeshNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        repeat: int = 1,
        target_spacing: tuple[float, float, float] = (1.3, 1.3, 1.3),
        crop_size: tuple[int, int, int] = (128, 128, 128),
        window: int = 2048,
        level: int = 0,
        use_labels: Optional[Sequence[int]] = None,
        device: str | torch.device = "cpu",
        augmentation_frequency: float = 0.0,
        advanced_augmentation_frequency: float = 0.0,
        noise_augmentation_frequency: float = 0.0,
        augmentation_magnitude: float = 1.0,
        num_control_points: int = 16,
        max_displacement: float = 0.1,
        sample_indices: Optional[Sequence[int]] = None,
    ):
        self._repeat = repeat
        files = glob(os.path.join(data_dir, "*.nii.gz"))
        self._images_fnames = sorted(list(filter(lambda x: "_image" in x, files)))
        self._labels_fnames = sorted(list(filter(lambda x: "_label" in x, files)))

        if len(self._images_fnames) == 0:
            raise ValueError("No images found in the data directory")

        if len(self._images_fnames) != len(self._labels_fnames):
            raise ValueError("Number of images and labels must be the same")

        if sample_indices is not None:
            self._images_fnames = [self._images_fnames[i] for i in sample_indices]
            self._labels_fnames = [self._labels_fnames[i] for i in sample_indices]

        self._original_images = []
        self._original_labels = []

        # Store parameters needed for normalization/processing
        self._window = window
        self._level = level
        self._use_labels = use_labels

        # Augmentation parameters
        self._augmentation_frequency = augmentation_frequency
        self._advanced_augmentation_frequency = advanced_augmentation_frequency
        self._noise_augmentation_frequency = noise_augmentation_frequency
        self._augmentation_magnitude = augmentation_magnitude
        self._device = torch.device(device)
        self._num_control_points = num_control_points
        self._max_displacement = max_displacement

        # Pre-compute grid for faster B-spline interpolation
        grid_size = 4  # 4x4x4 control grid for B-splines
        self._grid_coords = torch.linspace(-1, 1, grid_size)

        for img_fname, lbl_fname in zip(self._images_fnames, self._labels_fnames):
            sitk_image = sitk.ReadImage(img_fname)
            sitk_label = sitk.ReadImage(lbl_fname)

            resampled_image = self._resample_image(sitk_image, target_spacing)
            resampled_label = self._resample_label(sitk_label, target_spacing)

            image_array = sitk.GetArrayFromImage(resampled_image)
            label_array = sitk.GetArrayFromImage(resampled_label)

            image_array = self._center_crop(image_array, crop_size)
            label_array = self._center_crop(label_array, crop_size)

            self._original_images.append(torch.from_numpy(image_array).float())
            self._original_labels.append(torch.from_numpy(label_array).long())

    def _apply_augmentation(self, image, label):
        """Apply augmentation to image and label tensors."""
        # Create coordinate grid for the volume
        d, h, w = image.shape
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, d),
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        )
        grid = torch.stack([x, y, z], dim=-1).to(self._device)

        # Apply noise augmentation
        if self._noise_augmentation_frequency > 0 and torch.rand(1).item() < self._noise_augmentation_frequency:
            noise = torch.randn_like(image) * 0.05 * self._augmentation_magnitude
            image = image + noise

        # Apply advanced augmentation (control point based)
        if self._advanced_augmentation_frequency > 0 and torch.rand(1).item() < self._advanced_augmentation_frequency:
            # Choose between different augmentation methods
            method = torch.randint(0, 2, (1,)).item()

            if method == 0:
                # Control point based displacement
                displacement_field = self._generate_control_point_displacement(grid.reshape(-1, 3)).reshape(d, h, w, 3)
            else:
                # B-spline based displacement
                displacement_field = self._generate_bspline_displacement_field(grid.reshape(-1, 3)).reshape(d, h, w, 3)

            # Apply displacement to grid
            deformed_grid = grid + displacement_field

            # Sample image and label using the deformed grid
            image = self._sample_volume(image[None, None], deformed_grid)[0, 0]
            label = self._sample_volume(label[None, None].float(), deformed_grid, mode='nearest')[0, 0].long()

        # Apply basic affine augmentation
        if self._augmentation_frequency > 0 and torch.rand(1).item() < self._augmentation_frequency:
            # Create affine transformation matrix
            theta = self._create_affine_matrix(self._augmentation_magnitude)

            # Apply affine transformation to grid
            grid_flat = grid.reshape(-1, 3)
            grid_homogeneous = torch.nn.functional.pad(grid_flat, pad=(0, 1), value=1.0)
            transformed_grid = (grid_homogeneous @ theta.T)[:, :3].reshape(d, h, w, 3)

            # Sample image and label using the transformed grid
            image = self._sample_volume(image[None, None], transformed_grid)[0, 0]
            label = self._sample_volume(label[None, None].float(), transformed_grid, mode='nearest')[0, 0].long()

        return image, label

    def _sample_volume(self, volume, grid, mode='bilinear'):
        """Sample a volume using a coordinate grid."""
        # Add batch and channel dimensions if needed
        if volume.dim() == 3:
            volume = volume[None, None]
        elif volume.dim() == 4:
            volume = volume[None]

        # Normalize grid to [-1, 1] for grid_sample
        grid_normalized = grid.clone()

        # Reshape grid for grid_sample: [d, h, w, 3] -> [1, d, h, w, 3]
        grid_normalized = grid_normalized.unsqueeze(0)

        # Use grid_sample to interpolate
        sampled = torch.nn.functional.grid_sample(
            volume,
            grid_normalized,
            mode=mode,
            padding_mode='zeros',
            align_corners=True
        )

        return sampled

    def _create_affine_matrix(self, magnitude=1.0):
        """Create a random affine transformation matrix."""
        # translation
        trans_x = (torch.rand(1)*2-1) * 0.2 * magnitude
        trans_z = (torch.rand(1)*2-1) * 0.2 * magnitude
        trans_y = (torch.rand(1)*2-1) * 0.2 * magnitude

        # rotation
        angle_x = (torch.rand(1)*2-1) * 0.2 * magnitude
        angle_y = (torch.rand(1)*2-1) * 0.2 * magnitude
        angle_z = (torch.rand(1)*2-1) * 0.2 * magnitude

        # scaling
        scale_x = (torch.rand(1)*2-1) * 0.2 * magnitude
        scale_y = (torch.rand(1)*2-1) * 0.2 * magnitude
        scale_z = (torch.rand(1)*2-1) * 0.2 * magnitude

        # shearing
        shear_xy = (torch.rand(1)*2-1) * 0.1 * magnitude
        shear_xz = (torch.rand(1)*2-1) * 0.1 * magnitude
        shear_yx = (torch.rand(1)*2-1) * 0.1 * magnitude
        shear_yz = (torch.rand(1)*2-1) * 0.1 * magnitude
        shear_zx = (torch.rand(1)*2-1) * 0.1 * magnitude
        shear_zy = (torch.rand(1)*2-1) * 0.1 * magnitude

        theta1_s = torch.tensor([
            [1*(1+scale_x), 0, 0, 0],
            [0, 1*(1+scale_y), 0, 0],
            [0, 0, 1*(1+scale_z), 0],
            [0, 0, 0, 1]
        ], device=self._device)

        theta1_shear = torch.tensor([
            [1, shear_xy, shear_xz, 0],
            [shear_yx, 1, shear_yz, 0],
            [shear_zx, shear_zy, 1, 0],
            [0, 0, 0, 1]
        ], device=self._device)

        theta1_x = torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(angle_x), -torch.sin(angle_x), 0],
            [0, torch.sin(angle_x),  torch.cos(angle_x), 0],
            [0, 0, 0, 1]
        ], device=self._device)

        theta1_y = torch.tensor([
            [torch.cos(angle_y), 0,  torch.sin(angle_y), 0],
            [0, 1, 0, 0],
            [-torch.sin(angle_y), 0, torch.cos(angle_y), 0],
            [0, 0, 0, 1]
        ], device=self._device)

        theta1_z = torch.tensor([
            [torch.cos(angle_z), -torch.sin(angle_z), 0, 0],
            [torch.sin(angle_z),  torch.cos(angle_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=self._device)

        theta1_trans = torch.tensor([
            [1, 0, 0, trans_x],
            [0, 1, 0, trans_y],
            [0, 0, 1, trans_z],
            [0, 0, 0, 1]
        ], device=self._device)

        theta1_ = theta1_trans @ theta1_z @ theta1_y @ theta1_x @ theta1_shear @ theta1_s

        return theta1_

    def _generate_control_point_displacement(self, vertices):
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

    def _generate_bspline_displacement_field(self, vertices):
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

    @staticmethod
    def _resample_image(sitk_image: sitk.Image, target_spacing: tuple[float, float, float]) -> sitk.Image:
        return resample_image(sitk_image, target_spacing, sitk.sitkLinear)

    @staticmethod
    def _resample_label(sitk_label: sitk.Image, target_spacing: tuple[float, float, float]) -> sitk.Image:
        return resample_image(sitk_label, target_spacing, sitk.sitkNearestNeighbor)

    @staticmethod
    def _center_crop(array: np.ndarray, crop_size: tuple[int, int, int]) -> np.ndarray:
        """Perform center crop on a 3D array."""
        # Get current dimensions
        d, h, w = array.shape

        # Calculate start indices for the crop
        d_start = max(0, d // 2 - crop_size[0] // 2)
        h_start = max(0, h // 2 - crop_size[1] // 2)
        w_start = max(0, w // 2 - crop_size[2] // 2)

        # Perform the crop
        d_end = min(d_start + crop_size[0], d)
        h_end = min(h_start + crop_size[1], h)
        w_end = min(w_start + crop_size[2], w)

        cropped = array[d_start:d_end, h_start:h_end, w_start:w_end]

        # If the cropped array is smaller than the target size, pad it
        if cropped.shape != crop_size:
            result = np.zeros(crop_size, dtype=array.dtype)
            d_pad, h_pad, w_pad = cropped.shape

            # Calculate padding offsets to center the content
            d_offset = (crop_size[0] - d_pad) // 2
            h_offset = (crop_size[1] - h_pad) // 2
            w_offset = (crop_size[2] - w_pad) // 2

            # Place the cropped array in the center of the result array
            result[d_offset:d_offset+d_pad,
                   h_offset:h_offset+h_pad,
                   w_offset:w_offset+w_pad] = cropped
            return result

        return cropped

    def __len__(self):
        return len(self._original_images) * self._repeat

    def __getitem__(self, idx):
        idx = idx % len(self._original_images)

        # Get original image and label
        image = self._original_images[idx].to(self._device)
        label = self._original_labels[idx].to(self._device)

        # Apply augmentation if enabled
        if (self._advanced_augmentation_frequency > 0 or
            self._augmentation_frequency > 0 or
            self._noise_augmentation_frequency > 0):
            image, label = self._apply_augmentation(image, label)

        # normalize image to [-1, 1]
        image_normalized = apply_window_level(image.clone(), self._window, self._level)

        # convert label values if needed
        processed_label = label.clone()

        # Apply label mapping (original labels are not 0, 1, 2, ...)
        unique_values = torch.unique(processed_label)
        for i, j in enumerate(unique_values):
            processed_label[processed_label == j] = i

        # Filter labels if use_labels is specified
        if self._use_labels is not None:
            processed_label = torch.isin(processed_label, torch.tensor(self._use_labels, device=self._device))

        # Add channel dimension to image
        image_normalized = image_normalized.unsqueeze(0)

        return image_normalized, processed_label.float()


def smoke_test():
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    dataset = ImplicitMeshNetDataset(
        data_dir="mmwhs",
        use_labels=[2, 3],
        augmentation_frequency=0.0,
        advanced_augmentation_frequency=1.0
    )
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for image, label in dataloader:
        print(image.shape, label.shape)
        print(image.min(), image.max())
        print(label.unique())

        fig, axs = plt.subplots(2, 3, figsize=(8, 6))
        axs[0, 0].imshow(image[0, 0, 64, :, :], cmap="gray")
        axs[0, 1].imshow(image[0, 0, :, 64, :], cmap="gray")
        axs[0, 2].imshow(image[0, 0, :, :, 64], cmap="gray")
        axs[1, 0].imshow(label[0, 64, :, :])
        axs[1, 1].imshow(label[0, :, 64, :])
        axs[1, 2].imshow(label[0, :, :, 64])
        fig.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    smoke_test()
