from typing import Optional

import torch
from unet3d import UNet3D
from gcn import GCN


class ShapeEstimationNet(torch.nn.Module):
    def __init__(
        self,
        unet_num_classes: int,
        gcn_out_channels: int = 3,
        gcn_amplification_factor: float = 0.1,
        unet_dropout_p: float = 0.1,
    ):
        """
        A model that combines a 3D UNet for segmentation and a graph convolutional network for mesh generation.

        Parameters
        ----------
        unet_num_classes : int
            Number of classes in the segmentation task.
        gcn_out_channels : int, default=3
            Number of output channels for the mesh generation task.
        gcn_amplification_factor : float, default=0.1
            Factor by which the mesh is amplified.
        unet_dropout_p : float, default=0.1
            Dropout probability.
        """
        super().__init__()
        self._unet = UNet3D(num_classes=unet_num_classes, dropout_p=unet_dropout_p)
        self._gcn = GCN(amplification_factor=gcn_amplification_factor, out_channels=gcn_out_channels)

    def forward(
        self,
        img: torch.Tensor,
        mesh_template: torch.Tensor,
        edge_index: torch.Tensor = None,
        get_segmentation: bool = True
    ) -> tuple[list[torch.Tensor], Optional[torch.Tensor]]:
        features_enc = self._unet.encoder(img)
        mesh_list = self._gcn(features_enc, mesh_template, edge_index)
        segmentation = self._unet.decoder(features_enc) if get_segmentation else None
        return mesh_list, segmentation


def smoke_test():
    # pylint: disable=import-outside-toplevel
    from time import perf_counter
    import trimesh
    from gcn import create_edge_index

    torch.manual_seed(0)
    sphere = trimesh.creation.icosphere(radius=1.0, subdivisions=4)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    batch_size = 1
    num_meshes = 1
    vertices = torch.from_numpy(sphere.vertices).float()[None,None,].repeat(batch_size, num_meshes, 1, 1).to(device)
    edge_index = create_edge_index(torch.from_numpy(sphere.vertices), torch.from_numpy(sphere.faces)).to(device)

    combined = ShapeEstimationNet(unet_num_classes=2, gcn_amplification_factor=0.1).to(device)

    img = torch.randn(batch_size, 1, 128, 128, 128).to(device)
    meshes, segmentation = combined(img, vertices, edge_index, get_segmentation=True)
    [print(mesh.shape) for mesh in meshes]
    print(segmentation.shape)

    losses = [torch.nn.functional.mse_loss(mesh, vertices) for mesh in meshes]
    loss = sum(losses)
    loss.backward()
    print(loss)

    start = perf_counter()
    for _ in range(10):
        meshes, segmentation = combined(img, vertices, edge_index, get_segmentation=True)
    end = perf_counter()
    print(f"Time taken: {end - start} seconds")


if __name__ == "__main__":
    smoke_test()
