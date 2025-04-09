from typing import Union, Optional

import torch
import torch_geometric as pyg

import trimesh
import networkx


def create_edge_index(vertices: torch.Tensor, faces: torch.Tensor) -> torch.LongTensor:
    """
    Computes the edge index matrix for the mesh defined by `vertices` and `faces`.

    Parameters
    ----------
    vertices : torch.Tensor
        The vertices with shape (N, 3)

    faces: torch.Tensor
        The faces indices with shape (F, 3)

    Returns
    -------
    edge_index : torch.Tensor
        A tensor with shape (2, N) containing the vertex indices defining an edge in the mesh.
    """
    m = trimesh.Trimesh(
        vertices=vertices.cpu().numpy(),
        faces=faces.cpu().numpy(),
        process=False
    )

    adj_matrix = networkx.adjacency_matrix(
        m.vertex_adjacency_graph,
        nodelist=range(m.vertices.shape[0])  # this parameter is very important!
    ).tocoo()

    return pyg.utils.from_scipy_sparse_matrix(adj_matrix)[0]


class GCN(torch.nn.Module):
    def __init__(
        self,
        amplification_factor: float = 0.1,
        normalization: Optional[str] = "sym",
        feature_channels: tuple[int, ...] = (16, 32, 64, 128, 256),
        out_channels: int = 3
    ):
        super().__init__()
        self._feature_level = [[3, 4], [1, 2], [0, 1]]
        self._mesh_feat_merge_num = [288, 96, 32]

        self._init_gconv = torch.nn.ModuleList([
            pyg.nn.ChebConv(3, 384, K=2, normalization=normalization),
            pyg.nn.ChebConv(self._mesh_feat_merge_num[0], 144, K=2, normalization=normalization),
            pyg.nn.ChebConv(self._mesh_feat_merge_num[1], 64, K=2, normalization=normalization)
        ])

        self._gc_block_list = torch.nn.ModuleList([
            GraphConvBlock(
                in_channels=384+feature_channels[-1]+feature_channels[-2],
                hidden_channels=self._mesh_feat_merge_num[0],
                out_channels=out_channels
            ),
            GraphConvBlock(
                in_channels=144+feature_channels[-3]+feature_channels[-4],
                hidden_channels=self._mesh_feat_merge_num[1],
                out_channels=out_channels
            ),
            GraphConvBlock(
                in_channels=64+feature_channels[-4]+feature_channels[-5],
                hidden_channels=self._mesh_feat_merge_num[2],
                out_channels=out_channels
            ),
        ])
        self.register_buffer("_amplification_factor", torch.tensor(amplification_factor))

    def forward(
        self,
        features: list[torch.Tensor],
        mesh_template: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        out_mesh = []

        curr_mesh = mesh_template.float()
        curr_feat = curr_mesh.clone()

        for i in (0, 1, 2):
            curr_feat = self._init_gconv[i](x=curr_feat, edge_index=edge_index).relu()
            proj = projection(features, curr_mesh, feature_levels=self._feature_level[i])
            curr_feat = torch.concat([curr_feat, proj], dim=-1)
            curr_mesh_res, curr_feat = self._gc_block_list[i](curr_feat, edge_index)
            curr_mesh = curr_mesh + self._amplification_factor*curr_mesh_res
            out_mesh.append(curr_mesh)

        return out_mesh


class GraphConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        normalization: Optional[str] = None
    ) -> None:
        super().__init__()

        self._gconv_init = pyg.nn.ChebConv(in_channels, hidden_channels, K=2, normalization=normalization)
        self._gresconv_list = torch.nn.ModuleList([
            GraphResBlock(hidden_channels, hidden_channels)
        for _ in range(num_blocks)])
        self._gconv_final = pyg.nn.ChebConv(hidden_channels, out_channels, K=2, normalization=normalization)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor]:
        y = self._gconv_init(x=x, edge_index=edge_index).relu()

        for block in self._gresconv_list:
            y = block(x=y, edge_index=edge_index)

        y2 = self._gconv_final(x=y, edge_index=edge_index)

        return y2, y


class GraphResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = None
    ) -> None:
        super().__init__()
        self._gconv0 = pyg.nn.ChebConv(in_channels, out_channels, K=2, normalization=normalization)
        self._gconv1 = pyg.nn.ChebConv(out_channels, out_channels, K=2, normalization=normalization)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        y = self._gconv0(x=x, edge_index=edge_index).relu()
        y = self._gconv1(x=y, edge_index=edge_index).relu()
        return (x + y)/2.0


def projection(
    features: list[torch.Tensor],
    vertices: torch.Tensor,
    feature_levels: list[int] = (1,)
) -> torch.Tensor:
    """
    Projects the features in `features` selected by `feature_levels` onto the mesh `vertices` using trilinear
    interpolation of the feature volume at sub-voxel vertex location.

    This implementation uses grid_sample.

    Parameters
    ----------
    features : list[torch.Tensor]
        List of feature tensors from different levels, each with shape [B, C, D, H, W]
    vertices : torch.Tensor
        Vertex coordinates with shape [B, 1, V, 3] where V is the number of vertices
    feature_levels : list[int]
        Indices of feature maps to use from the features list

    Returns
    -------
    torch.Tensor
        The interpolated features for each vertex, shape [B, 1, V, F] where F is the total number
        of feature channels across all selected feature levels
    """
    has_channels = vertices.ndim == 4
    if not has_channels:
        vertices = vertices[:,None]

    if vertices.ndim != 4:
        raise ValueError(f"Expected vertices tensor with 3 or 4 dimensions, got shape {vertices.shape}")

    t = vertices[:,None]

    # For 5D inputs, grid[n, d, h, w] specifies the x, y, z pixel locations for interpolating output[n, :, d, h, w].
    out = []
    for fl in feature_levels:
        out.append(torch.nn.functional.grid_sample(features[fl], t, align_corners=False, mode="bilinear")[:, :, 0])

    proj = torch.cat(out, dim=1).permute(0, 2, 3, 1)

    if not has_channels:
        proj = proj[:,0]

    return proj


def smoke_test():
    # pylint: disable=import-outside-toplevel
    import trimesh
    from unet3d import UNet3D

    sphere = trimesh.creation.icosphere(radius=0.8, subdivisions=4)

    vertices = sphere.vertices
    vert = torch.from_numpy(vertices).float()[None,None,].repeat(1, 1, 1, 1)

    edge_index = create_edge_index(vert[0,0], torch.from_numpy(sphere.faces))

    unet = UNet3D()
    gcn = GCN()

    features = unet.encoder(torch.randn(1, 1, 128, 128, 128))
    pred_list = gcn(features=features, mesh_template=vert, edge_index=edge_index)
    losses = [torch.nn.functional.mse_loss(pred, vert) for pred in pred_list]

    loss = sum(losses)
    loss.backward()
    print(loss)

if __name__ == "__main__":
    smoke_test()
