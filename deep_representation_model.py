import torch
import torch_geometric as pyg

from gridfeatureprojectiontrilinear import GridFeatureProjectionTrilinear
from gcn import create_edge_index


class MeshEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_stages: int = 3):
        super().__init__()
        self.conv0= pyg.nn.GCNConv(in_channels, hidden_channels)

        self.convs_1 = torch.nn.ModuleList()
        for _ in range(num_stages):
            self.convs_1.append(pyg.nn.GCNConv(hidden_channels, hidden_channels))

        self.convs_2 = torch.nn.ModuleList()
        for _ in range(num_stages):
            self.convs_2.append(pyg.nn.GCNConv(hidden_channels, hidden_channels))

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Initial features could be vertex positions, normals, etc.
        y0 = self.conv0(x, edge_index).relu()

        skips = []

        for conv_1, conv_2 in zip(self.convs_1, self.convs_2):
            y = self.dropout(conv_1(y0, edge_index).relu())
            y = conv_2(y, edge_index)
            y = (y0 + y).relu()
            skips.append(y)
            y0 = y

        return tuple(skips)


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 16):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = torch.nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = torch.nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.act_fn = torch.nn.SiLU()

        if in_channels != out_channels:
            self.residual = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.residual = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gn1(self.conv1(x))
        y = self.act_fn(y)
        y = self.gn2(self.conv2(y))
        return self.residual(x) + y


class UpsampleBlock(torch.nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = torch.nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class VoxelDecoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        skip_channels: tuple[int, int, int],
        out_channels: int,
        volume_size: int,
        num_stages: int = 3,
        num_groups: int = 16,
        num_channels: int = 128,
    ):
        super().__init__()
        self.volume_size = volume_size
        self.res_blocks = torch.nn.ModuleList()
        self.upsamples = torch.nn.ModuleList()
        self.conv_finals = torch.nn.ModuleList()

        self.res_blocks.append(ResBlock(hidden_channels, num_channels, num_groups))
        self.upsamples.append(UpsampleBlock(num_channels))
        self.conv_finals.append(torch.nn.Conv3d(num_channels, out_channels, kernel_size=1, stride=1))

        for i in range(num_stages-1):
            self.res_blocks.append(ResBlock(num_channels//2**i + skip_channels[i], num_channels//2**(i+1), num_groups))
            self.upsamples.append(UpsampleBlock(num_channels//2**(i+1)))
            self.conv_finals.append(torch.nn.Conv3d(num_channels//2**(i+1), out_channels, kernel_size=1, stride=1))

        # Refinement block
        self.res_blocks.append(
            ResBlock(num_channels//2**(num_stages-1) + skip_channels[-1], num_channels//2**(num_stages-1), num_groups)
        )

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        y_list = []

        y = self.res_blocks[0](x_list[0])
        y = self.upsamples[0](y)
        if self.training:
            y1 = self.conv_finals[0](y)
            y1 = torch.nn.functional.interpolate(y1, size=(self.volume_size, self.volume_size, self.volume_size),
                                                 mode="trilinear", align_corners=False)
            y_list.append(y1)

        for i in range(1, len(self.res_blocks)-2):
            y = self.res_blocks[i](torch.cat([y, x_list[i]], dim=1))
            y = self.upsamples[i](y)
            if self.training:
                y2 = self.conv_finals[i](y)
                y2 = torch.nn.functional.interpolate(y2, size=(self.volume_size, self.volume_size, self.volume_size),
                                                    mode="trilinear", align_corners=False)
                y_list.append(y2)

        y = self.res_blocks[-2](torch.cat([y, x_list[-2]], dim=1))
        y = self.upsamples[-1](y)

        y = self.res_blocks[-1](torch.cat([y, x_list[-1]], dim=1))
        y = self.conv_finals[-1](y)

        if self.training:
            y_list.append(y)
            return y_list
        return [y]


class DeepRepresentationNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        out_channels: int = 1,
        volume_size: int = 64,
        num_stages: int = 3
    ):
        super().__init__()
        if num_stages < 3:
            raise ValueError("num_stages must be at least 3")

        self.mesh_encoder = MeshEncoder(in_channels, hidden_channels, num_stages)
        self.num_stages = num_stages

        self.feature_projections = torch.nn.ModuleList()

        for i in range(self.num_stages+1):
            self.feature_projections.append(GridFeatureProjectionTrilinear(volume_size//(2**i)))

        self.voxel_decoder = VoxelDecoder(
            hidden_channels,
            (hidden_channels,)*(num_stages-1) + (3,),
            out_channels,
            volume_size,
            num_stages
        )

    def forward(self, vertices: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Encode mesh features
        encoded_features = self.mesh_encoder(vertices, edge_index)

        # Project features using grid_sample
        projected_features = [self.feature_projections[0](vertices, torch.ones_like(vertices))]

        for i, e in enumerate(encoded_features):
            projected_features.append(self.feature_projections[i+1](vertices, e))

        return self.voxel_decoder(projected_features[::-1])


def smoke_test():
    import trimesh # pylint: disable=import-outside-toplevel

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    mesh2voxel = DeepRepresentationNet().to(device)

    sphere = trimesh.creation.icosphere(radius=0.5, subdivisions=3)
    vertices = torch.from_numpy(sphere.vertices).float().to(device)
    faces = torch.from_numpy(sphere.faces).to(device)
    edge_index = create_edge_index(vertices, faces).to(device)
    vertices = vertices[None,]

    y_list = mesh2voxel(vertices, edge_index)
    [print(y.shape) for y in y_list]

    seg = torch.ones_like(y_list[-1])
    loss = sum(torch.nn.functional.mse_loss(y, seg) for y in y_list)
    loss.backward()
    print(loss.item())


if __name__ == "__main__":
    smoke_test()
