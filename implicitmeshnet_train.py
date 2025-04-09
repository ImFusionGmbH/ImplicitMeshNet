import os
import json
import subprocess
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

import numpy as np
import torch
import trimesh
from fire import Fire
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt

from implicitmeshnet_dataset import ImplicitMeshNetDataset
from shapeestimationnet import ShapeEstimationNet
from deep_representation_model import DeepRepresentationNet
from deep_representation_pretrain import DeepRepresentationTrainingConfig
from gcn import create_edge_index
from utils import check_for_uncommitted_changes, binary_voxel_erosion
from sdffunctions import Mesh2SDF
from laplacian_loss import GraphLaplacianLoss
from gridfeatureprojectiontrilinear import GridFeatureProjectionTrilinear


def get_mesh_template(
    mesh_template_radius: float,
    mesh_template_subdivisions: int,
    batch_size: int = 1,
    device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    template_mesh = trimesh.creation.icosphere(
        radius=mesh_template_radius,
        subdivisions=mesh_template_subdivisions
    )
    template_vertices = torch.from_numpy(template_mesh.vertices).float().to(device)
    template_faces = torch.from_numpy(template_mesh.faces).to(torch.int32).to(device)
    template_edge_indices = create_edge_index(template_vertices, template_faces).to(device)
    template_vertices = template_vertices[None,].repeat(batch_size, 1, 1)

    return template_vertices, template_faces, template_edge_indices


def surface_loss(projected_mesh: torch.Tensor, label: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute the surface loss between a projected mesh and a label.

    Parameters
    ----------
    projected_mesh : torch.Tensor (B, 1, H, W, D)
        The projected mesh.
    label : torch.Tensor (B, 1, H, W, D)
        The label.
    weight : torch.Tensor (B, 1, H, W, D)
        Positional weights for the loss.

    Returns
    -------
    loss_surface : torch.Tensor
        The surface loss.
    """
    loss_surface = torch.nn.functional.mse_loss(projected_mesh, label, weight=weight)

    # if projected_mesh is all zeros, the loss becomes NaN
    if loss_surface.isnan().any():
        loss_surface = torch.nn.functional.mse_loss(
            projected_mesh, label, weight=weight+1e-6
        )

    return loss_surface


def cosine_annealing(epoch: int, num_epochs: int, start: float, end: float) -> float:
    """
    Cosine annealing schedule.

    Parameters
    ----------
    epoch : int
        The current epoch. If epoch-1 > num_epochs, the value will be end.
    num_epochs : int
        The total number of epochs for the cosine annealing schedule.
    start : float
        The starting value.
    end : float
        The ending value.

    Returns
    -------
    value : float
        The value at the current epoch.
    """
    if epoch-1 > num_epochs:
        return end
    return start + (end - start) * (1 + np.cos(np.pi * epoch / (num_epochs-1) + np.pi)) / 2


@dataclass
class ImplicitMeshNetTrainConfig:
    git_commit: str = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
    ).decode("utf-8").strip()
    start_time: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    artifacts_dir: str = "implicitmeshnet"
    snapshot_path: Optional[str] = None
    tensorboard_dir: Optional[str] = None
    config_dir: Optional[str] = None
    data_dir: str = "mmwhs"
    use_labels: tuple[int, ...] = (2, 3)
    dataset_repeat: int = 1

    mesh_template_radius: float = 0.5
    mesh_template_subdivisions: int = 4

    target_spacing: tuple[float, float, float] = (1.3, 1.3, 1.3)
    crop_size: tuple[int, int, int] = (128, 128, 128)
    window: int = 1000
    level: int = 200

    mesh2voxel_config_path: Optional[str] = None
    mesh2voxel_hidden_channels: int = 128
    mesh2voxel_out_channels: int = 1
    mesh2voxel_volume_size: int = 64
    mesh2voxel_num_stages: int = 3
    voxel2mesh_lr: float = 1e-4
    mesh2voxel_lr: float = 1e-4
    voxel2mesh_lr_min: float = 1e-5
    weight_decay: float = 1e-6

    batch_size: int = 1
    num_epochs: int = 1200
    device: str = "cpu"

    segmentation_loss_weight: float = 0.1
    laplacian_loss_weight_start: float = 1e3
    laplacian_loss_weight_end: float = 1e-2
    laplacian_loss_weight_max_epoch: Optional[int] = None
    surface_loss_weight: float = 1.0

    unet_num_classes: int = 2
    gcn_out_channels: int = 3
    gcn_amplification_factor: float = 0.1
    unet_dropout_p: float = 0.1

    augmentation_frequency: float = 0.5
    advanced_augmentation_frequency: float = 0.8
    noise_augmentation_frequency: float = 0.0
    augmentation_magnitude: float = 1.0

    num_control_points: int = 16
    max_displacement: float = 0.1

    sigmoid_temperature: float = 100.0

    def __post_init__(self):
        if self.laplacian_loss_weight_max_epoch is None:
            self.laplacian_loss_weight_max_epoch = self.num_epochs

        if self.snapshot_path is None:
            self.snapshot_path = f"{self.artifacts_dir}/snapshots_implicitmeshnet/model_{self.start_time}.pth"

        if self.tensorboard_dir is None:
            self.tensorboard_dir = f"{self.artifacts_dir}/tensorboard_implicitmeshnet/{self.start_time}"

        if self.config_dir is None:
            self.config_dir = f"{self.artifacts_dir}/configs_implicitmeshnet"


def train(
    mesh2voxel_config_path: Optional[str] = None,
    data_dir: str = "mmwhs",
    artifacts_dir: str = ".",
    use_labels: tuple[int, ...] = (2, 3),
    dataset_repeat: int = 2,
    mesh_template_radius: float = 0.5,
    mesh_template_subdivisions: int = 4,
    target_spacing: tuple[float, float, float] = (2.5, 2.5, 2.5),
    crop_size: tuple[int, int, int] = (64, 64, 64),
    window: int = 1000,
    level: int = 200,
    voxel2mesh_lr: float = 1e-4,
    mesh2voxel_lr: float = 3e-5,
    voxel2mesh_lr_min: float = 1e-5,
    weight_decay: float = 1e-6,
    mesh2voxel_hidden_channels: int = 128,
    mesh2voxel_out_channels: int = 1,
    mesh2voxel_volume_size: int = 64,
    mesh2voxel_num_stages: int = 3,
    segmentation_loss_weight: float = 0.1,
    laplacian_loss_weight_start: float = 1e3,
    laplacian_loss_weight_end: float = 1e-2,
    laplacian_loss_weight_max_epoch: Optional[int] = None,
    surface_loss_weight: float = 1.0,
    batch_size: int = 2,
    num_epochs: int = 1200,
    device: str = "cpu",
    unet_num_classes: int = 2,
    gcn_out_channels: int = 3,
    gcn_amplification_factor: float = 0.01,
    unet_dropout_p: float = 0.1,
    augmentation_frequency: float = 0.5,
    advanced_augmentation_frequency: float = 0.8,
    noise_augmentation_frequency: float = 0.0,
    sigmoid_temperature: float = 100.0,
    ignore_uncommitted_changes: bool = False
):
    matplotlib.use("Agg")

    if not ignore_uncommitted_changes and check_for_uncommitted_changes():
        return

    cfg = ImplicitMeshNetTrainConfig(
        mesh2voxel_config_path=mesh2voxel_config_path,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        use_labels=use_labels,
        dataset_repeat=dataset_repeat,
        mesh_template_radius=mesh_template_radius,
        mesh_template_subdivisions=mesh_template_subdivisions,
        target_spacing=target_spacing,
        crop_size=crop_size,
        window=window,
        level=level,
        voxel2mesh_lr=voxel2mesh_lr,
        mesh2voxel_lr=mesh2voxel_lr,
        voxel2mesh_lr_min=voxel2mesh_lr_min,
        weight_decay=weight_decay,
        mesh2voxel_hidden_channels=mesh2voxel_hidden_channels,
        mesh2voxel_out_channels=mesh2voxel_out_channels,
        mesh2voxel_volume_size=mesh2voxel_volume_size,
        mesh2voxel_num_stages=mesh2voxel_num_stages,
        segmentation_loss_weight=segmentation_loss_weight,
        laplacian_loss_weight_start=laplacian_loss_weight_start,
        laplacian_loss_weight_end=laplacian_loss_weight_end,
        laplacian_loss_weight_max_epoch=laplacian_loss_weight_max_epoch,
        surface_loss_weight=surface_loss_weight,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        unet_num_classes=unet_num_classes,
        gcn_out_channels=gcn_out_channels,
        gcn_amplification_factor=gcn_amplification_factor,
        unet_dropout_p=unet_dropout_p,
        augmentation_frequency=augmentation_frequency,
        advanced_augmentation_frequency=advanced_augmentation_frequency,
        noise_augmentation_frequency=noise_augmentation_frequency,
        sigmoid_temperature=sigmoid_temperature,
    )

    dataset = ImplicitMeshNetDataset(
        repeat=cfg.dataset_repeat,
        data_dir=cfg.data_dir,
        use_labels=cfg.use_labels,
        target_spacing=cfg.target_spacing,
        crop_size=cfg.crop_size,
        window=cfg.window,
        level=cfg.level,
        augmentation_frequency=cfg.augmentation_frequency,
        advanced_augmentation_frequency=cfg.advanced_augmentation_frequency,
        noise_augmentation_frequency=cfg.noise_augmentation_frequency,
        augmentation_magnitude=cfg.augmentation_magnitude,
        num_control_points=cfg.num_control_points,
        max_displacement=cfg.max_displacement,
        device=cfg.device
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    template_vertices, template_faces, template_edge_indices = get_mesh_template(
        mesh_template_radius=cfg.mesh_template_radius,
        mesh_template_subdivisions=cfg.mesh_template_subdivisions,
        batch_size=cfg.batch_size,
        device=cfg.device
    )

    voxel2mesh_model = ShapeEstimationNet(
        unet_num_classes=cfg.unet_num_classes,
        gcn_out_channels=cfg.gcn_out_channels,
        gcn_amplification_factor=cfg.gcn_amplification_factor,
        unet_dropout_p=cfg.unet_dropout_p,
    ).to(cfg.device)
    mesh2voxel_model = DeepRepresentationNet(
        in_channels=cfg.gcn_out_channels,
        hidden_channels=cfg.mesh2voxel_hidden_channels,
        out_channels=cfg.mesh2voxel_out_channels,
        volume_size=cfg.mesh2voxel_volume_size,
        num_stages=cfg.mesh2voxel_num_stages
    ).to(cfg.device)

    # Load pretrained mesh2voxel model if provided
    if cfg.mesh2voxel_config_path is not None:
        with open(cfg.mesh2voxel_config_path, "r", encoding="utf-8") as f:
            mesh2voxel_config = DeepRepresentationTrainingConfig(**json.load(f))
        snapshot = torch.load(mesh2voxel_config.snapshot_path, map_location="cpu")
        mesh2voxel_model.load_state_dict(snapshot["state_dict"])
        mesh2voxel_model = mesh2voxel_model.to(cfg.device)
        print(f"Loaded mesh2voxel model from {mesh2voxel_config.snapshot_path}")

    if cfg.crop_size[0] != cfg.mesh2voxel_volume_size:
        raise ValueError(
            f"Crop size {cfg.crop_size} does not match mesh2voxel volume size {cfg.mesh2voxel_volume_size}"
        )

    optimizer_voxel2mesh = torch.optim.AdamW(
        voxel2mesh_model.parameters(),
        lr=cfg.voxel2mesh_lr,
        weight_decay=cfg.weight_decay
    )
    optimizer_mesh2voxel = torch.optim.AdamW(
        mesh2voxel_model.parameters(),
        lr=cfg.mesh2voxel_lr,
        weight_decay=cfg.weight_decay
    )

    lr_scheduler_voxel2mesh = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_voxel2mesh,
        T_max=num_epochs,
        eta_min=cfg.voxel2mesh_lr_min
    )

    mesh2sdf = Mesh2SDF(cfg.mesh2voxel_volume_size)

    # lap_loss_fn = LaplacianLoss(template_edge_indices)
    lap_loss_fn = GraphLaplacianLoss(
        edge_index=template_edge_indices,
        num_vertices=template_vertices.shape[1],
        normalization=None
    )

    # GridFeatureProjectionTrilinear guidance
    grid_feature_projector = GridFeatureProjectionTrilinear(cfg.mesh2voxel_volume_size)

    config_dir = cfg.config_dir
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, f"{cfg.start_time}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=4)

    os.makedirs(os.path.dirname(cfg.snapshot_path), exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.tensorboard_dir)
    losses_voxel2mesh_train = []
    losses_mesh2voxel_train = []

    laplacian_loss_weight = cfg.laplacian_loss_weight_start
    pbar1 = tqdm(range(cfg.num_epochs), position=0, ncols=80)
    for epoch in pbar1:
        pbar2 = tqdm(dataloader, position=1, leave=False, ncols=80)
        voxel2mesh_model.train()
        losses_voxel2mesh_train_epoch = []
        losses_voxel2mesh_train_sdf_epoch = []
        losses_voxel2mesh_train_seg_epoch = []
        losses_voxel2mesh_train_surface_epoch = []
        losses_voxel2mesh_train_lap_epoch = []
        losses_mesh2voxel_train_epoch = []
        # cosine annealing
        laplacian_loss_weight = cosine_annealing(
            epoch,
            cfg.laplacian_loss_weight_max_epoch,
            start=cfg.laplacian_loss_weight_start,
            end=cfg.laplacian_loss_weight_end
        )
        for image, label in pbar2:
            # 1. train voxel2mesh
            mesh2voxel_model.eval()
            optimizer_voxel2mesh.zero_grad()

            # get mesh from input image
            meshes, segmentation = voxel2mesh_model(
                image,
                mesh_template=template_vertices,
                edge_index=template_edge_indices
            )

            # get sdf from mesh
            sdf_pred = mesh2voxel_model(meshes[-1], edge_index=template_edge_indices)[0]

            # soft-binarize sdf to get an occupancy map
            seg_pred = torch.sigmoid(-cfg.sigmoid_temperature * sdf_pred[:,0])

            # surface loss
            if cfg.surface_loss_weight > 0:
                 # project mesh to grid
                projected_mesh = grid_feature_projector(meshes[-1], torch.ones_like(meshes[-1][...,:1]))
                projected_mesh = (projected_mesh > 0.2).float()
                with torch.no_grad():
                    border = label.unsqueeze(1) - binary_voxel_erosion(label)

                loss_surface = surface_loss(projected_mesh, border, projected_mesh.detach())
            else:
                projected_mesh = None
                loss_surface = torch.tensor(0.0, device=cfg.device)

            # sdf loss
            loss_sdf = torch.nn.functional.binary_cross_entropy(seg_pred, label)

            # segmentation loss
            loss_segmentation = torch.nn.functional.cross_entropy(segmentation, label.long())

            # laplacian loss
            loss_laplacian = sum(lap_loss_fn(mesh) for mesh in meshes)
            loss_voxel2mesh = loss_sdf \
                + cfg.segmentation_loss_weight * loss_segmentation \
                + laplacian_loss_weight * loss_laplacian \
                + cfg.surface_loss_weight * loss_surface

            loss_voxel2mesh.backward()
            optimizer_voxel2mesh.step()

            # 2. train mesh2voxel
            mesh2voxel_model.train()
            optimizer_mesh2voxel.zero_grad()
            y_list = mesh2voxel_model(meshes[-1].detach(), edge_index=template_edge_indices)

            with torch.no_grad():
                sdf_list = []
                for mesh in meshes[-1]:  # Mesh2SDF does not support batching yet
                    sdf_list.append(mesh2sdf(mesh.detach(), template_faces)[None,])
                sdf = torch.stack(sdf_list).detach()

                # increase loss weight near the surface (sqrt(12) is the max theoretical distance for a unit cube)
                sdf_pos_weight = (-sdf.abs() + np.sqrt(12))/np.sqrt(12)
                sdf_pos_weight = sdf_pos_weight*sdf_pos_weight

            loss_mesh2voxel = sum(torch.nn.functional.mse_loss(y, sdf, weight=sdf_pos_weight) for y in y_list)

            loss_mesh2voxel.backward()
            optimizer_mesh2voxel.step()

            losses_voxel2mesh_train_epoch.append(loss_voxel2mesh.item())
            losses_voxel2mesh_train_sdf_epoch.append(loss_sdf.item())
            losses_voxel2mesh_train_seg_epoch.append(loss_segmentation.item())
            losses_voxel2mesh_train_lap_epoch.append(loss_laplacian.item())
            losses_mesh2voxel_train_epoch.append(loss_mesh2voxel.item())
            losses_voxel2mesh_train_surface_epoch.append(loss_surface.item())

        lr_scheduler_voxel2mesh.step()
        writer.add_scalar('train/voxel2mesh_lr', lr_scheduler_voxel2mesh.get_last_lr()[0], epoch)

        losses_voxel2mesh_train.append(np.mean(losses_voxel2mesh_train_epoch))
        losses_mesh2voxel_train.append(np.mean(losses_mesh2voxel_train_epoch))

        writer.add_scalar("train/loss_voxel2mesh", losses_voxel2mesh_train[-1], epoch)
        writer.add_scalar("train/loss_voxel2mesh_sdf", np.mean(losses_voxel2mesh_train_sdf_epoch), epoch)
        writer.add_scalar("train/loss_voxel2mesh_seg", np.mean(losses_voxel2mesh_train_seg_epoch), epoch)
        writer.add_scalar("train/loss_voxel2mesh_lap", np.mean(losses_voxel2mesh_train_lap_epoch), epoch)
        writer.add_scalar("train/loss_voxel2mesh_surface", np.mean(losses_voxel2mesh_train_surface_epoch), epoch)
        writer.add_scalar("train/loss_mesh2voxel", losses_mesh2voxel_train[-1], epoch)

        tqdm.write(f"epoch {epoch}:")
        tqdm.write(f"  loss_voxel2mesh: {losses_voxel2mesh_train[-1]:.5f}")
        tqdm.write(f"  loss_mesh2voxel: {losses_mesh2voxel_train[-1]:.5f}")

        seg_pred = seg_pred.detach().cpu().numpy()
        sdf = sdf.detach().cpu().numpy()
        sdf_pred = sdf_pred[:,0].detach().cpu().numpy()
        vertices = (meshes[-1].detach().cpu().numpy() + 1) * (cfg.mesh2voxel_volume_size-1)/2

        s = cfg.crop_size[0] // 2
        fig, ax = plt.subplots(6, 3, figsize=(10, 16))
        ax[0, 0].imshow(image[0, 0, s].cpu().numpy(), cmap="gray")
        ax[0, 1].imshow(image[0, 0, :, s].cpu().numpy(), cmap="gray")
        ax[0, 2].imshow(image[0, 0, :, :, s].cpu().numpy(), cmap="gray")
        ax[1, 0].imshow(label[0, s].cpu().numpy(), interpolation=None)
        ax[1, 1].imshow(label[0, :, s].cpu().numpy(), interpolation=None)
        ax[1, 2].imshow(label[0, :, :, s].cpu().numpy(), interpolation=None)
        ax[2, 0].imshow(seg_pred[0, s], vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        ax[2, 1].imshow(seg_pred[0, :, s], vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        ax[2, 2].imshow(seg_pred[0, :, :, s], vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        ax[2, 0].plot(vertices[0,:,2], vertices[0,:,1], 'k.', markersize=1, alpha=0.5)
        ax[2, 1].plot(vertices[0,:,2], vertices[0,:,0], 'k.', markersize=1, alpha=0.5)
        ax[2, 2].plot(vertices[0,:,1], vertices[0,:,0], 'k.', markersize=1, alpha=0.5)
        ax[3, 0].imshow(sdf_pred[0, s], vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        ax[3, 1].imshow(sdf_pred[0, :, s], vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        ax[3, 2].imshow(sdf_pred[0, :, :, s], vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        ax[4, 0].imshow(sdf[0, 0, s], vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        ax[4, 1].imshow(sdf[0, 0, :, s], vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        ax[4, 2].imshow(sdf[0, 0, :, :, s], vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        if projected_mesh is not None:
            proj_mesh = projected_mesh.detach().cpu().numpy()
            ax[5, 0].imshow(proj_mesh[0, 0, s], vmin=0, vmax=1, cmap="gray", interpolation=None)
            ax[5, 1].imshow(proj_mesh[0, 0, :, s], vmin=0, vmax=1, cmap="gray", interpolation=None)
            ax[5, 2].imshow(proj_mesh[0, 0, :, :, s], vmin=0, vmax=1, cmap="gray", interpolation=None)
        fig.tight_layout()
        plt.close("all")
        writer.add_figure("train/prediction", fig, epoch)

        mesh = trimesh.Trimesh(
            vertices=meshes[-1].detach().cpu().numpy()[0],
            faces=template_faces.cpu().numpy(),
            process=False
        )

        voxel2mesh_model.eval()
        mesh2voxel_model.eval()

        if losses_voxel2mesh_train[-1] == min(losses_voxel2mesh_train):
            torch.save({
                    "epoch": epoch,
                    "voxel2mesh_state_dict": voxel2mesh_model.state_dict(),
                    "mesh2voxel_state_dict": mesh2voxel_model.state_dict(),
                    "voxel2mesh_optimizer": optimizer_voxel2mesh.state_dict(),
                    "mesh2voxel_optimizer": optimizer_mesh2voxel.state_dict(),
                    "cfg": asdict(cfg),
                    "losses_voxel2mesh_train": losses_voxel2mesh_train,
                    "losses_mesh2voxel_train": losses_mesh2voxel_train,
                },
                cfg.snapshot_path
            )


if __name__ == "__main__":
    Fire(train)
