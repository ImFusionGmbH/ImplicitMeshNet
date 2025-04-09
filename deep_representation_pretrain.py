import os
from datetime import datetime
import json
from dataclasses import dataclass, asdict
import subprocess

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from fire import Fire
import torch
from torch.utils.tensorboard import SummaryWriter

from deep_representation_model import DeepRepresentationNet
from deep_representation_dataset import DeepRepresentationDataset
from utils import check_for_uncommitted_changes


@dataclass
class DeepRepresentationTrainingConfig:
    git_commit: str = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
    ).decode("utf-8").strip()
    start_time: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_path: str = f"snapshots_mesh2voxel/model_{start_time}.pth"

    volume_size: int = 128
    learning_rate: float = 1e-4
    learning_rate_min: float = 1e-8
    weight_decay: float = 1e-6
    batch_size: int = 1
    num_epochs: int = 200
    random_seed: int = 0

    hidden_channels: int = 128
    num_stages: int = 3

    mesh_subdivisions: int = 4

    augmentation_frequency: float = 0.9
    advanced_augmentation_frequency: float = 0.9
    noise_augmentation_frequency: float = 0.1
    num_control_points: int = 16
    max_displacement: float = 0.2
    repeat: int = 10


def main(
    volume_size: int = 128,
    learning_rate: float = 1e-4,
    learning_rate_min: float = 1e-8,
    weight_decay: float = 1e-6,
    batch_size: int = 2,
    num_epochs: int = 500,
    hidden_channels: int = 128,
    num_stages: int = 3,
    mesh_subdivisions: int = 4,
    augmentation_frequency: float = 0.9,
    advanced_augmentation_frequency: float = 0.9,
    noise_augmentation_frequency: float = 0.1,
    num_control_points: int = 16,
    max_displacement: float = 0.2,
    repeat: int = 10,
    device: str = "mps",
    random_seed: int = 0,
    ignore_uncommitted_changes: bool = False
):
    if not ignore_uncommitted_changes and check_for_uncommitted_changes():
        return

    device = torch.device(device)
    cfg = DeepRepresentationTrainingConfig(
        volume_size=volume_size,
        learning_rate=learning_rate,
        learning_rate_min=learning_rate_min,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_epochs=num_epochs,
        hidden_channels=hidden_channels,
        num_stages=num_stages,
        mesh_subdivisions=mesh_subdivisions,
        augmentation_frequency=augmentation_frequency,
        advanced_augmentation_frequency=advanced_augmentation_frequency,
        noise_augmentation_frequency=noise_augmentation_frequency,
        num_control_points=num_control_points,
        max_displacement=max_displacement,
        repeat=repeat,
        random_seed=random_seed
    )

    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    matplotlib.use("Agg")

    dataset = DeepRepresentationDataset(
        volume_size=cfg.volume_size,
        subdivisions=cfg.mesh_subdivisions,
        augmentation_frequency=cfg.augmentation_frequency,
        advanced_augmentation_frequency=cfg.advanced_augmentation_frequency,
        noise_augmentation_frequency=cfg.noise_augmentation_frequency,
        num_control_points=cfg.num_control_points,
        max_displacement=cfg.max_displacement,
        device=device,
        repeat=cfg.repeat
    )

    if len(dataset) == 0:
        raise RuntimeError("No images found")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = DeepRepresentationNet(
        hidden_channels=cfg.hidden_channels,
        volume_size=cfg.volume_size,
        num_stages=cfg.num_stages
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=cfg.learning_rate_min
    )

    writer = SummaryWriter(log_dir=f"tensorboard_mesh2voxel/{cfg.start_time}")
    os.makedirs("configs_mesh2voxel", exist_ok=True)
    os.makedirs("snapshots_mesh2voxel", exist_ok=True)
    with open(f"configs_mesh2voxel/{cfg.start_time}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=4)

    losses_epoch = []
    for epoch in tqdm(range(num_epochs), position=0, ncols=80):
        loss_epoch = []
        pbar_batch = tqdm(dataloader, position=1, leave=False, ncols=80)
        for sdf, vertices, _, edge_index in pbar_batch:
            optimizer.zero_grad()
            y_list = model(vertices, edge_index)

            # increase loss weight near the surface (sqrt(12) is the max theoretical distance for a unit cube)
            with torch.no_grad():
                sdf_pos_weight = (-sdf.abs() + np.sqrt(12))/np.sqrt(12)
                sdf_pos_weight = sdf_pos_weight*sdf_pos_weight

            loss = sum(torch.nn.functional.mse_loss(y, sdf, weight=sdf_pos_weight) for y in y_list)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())

        lr_scheduler.step()
        writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)

        # Log epoch loss to TensorBoard
        mean_epoch_loss = torch.tensor(loss_epoch).mean().item()
        losses_epoch.append(mean_epoch_loss)
        writer.add_scalar('train/mse', mean_epoch_loss, epoch)
        # writer.add_hparams(asdict(config), {'hparam/validation_metric': mean_epoch_loss}, run_name=".")

        # Add segmentation and prediction images directly to TensorBoard
        # Normalize to [0, 1] range for TensorBoard visualization
        pred_slice = y_list[-1].cpu().detach().numpy()[0, 0, volume_size//2].clip(-1, 1)
        # pylint: disable=undefined-loop-variable
        gt_slice = sdf.cpu().detach().numpy()[0, 0, volume_size//2].clip(-1, 1)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(pred_slice, vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        ax[1].imshow(gt_slice, vmin=-1, vmax=1, cmap="seismic", interpolation=None)
        writer.add_figure('train/prediction', fig, epoch)
        plt.close("all")

        tqdm.write(f"epoch {epoch} mean loss: {mean_epoch_loss:.5f}")

        if losses_epoch[-1] == min(losses_epoch):
            torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "cfg": asdict(cfg),
                    "losses_epoch": losses_epoch
                },
                cfg.snapshot_path
            )

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    Fire(main)
