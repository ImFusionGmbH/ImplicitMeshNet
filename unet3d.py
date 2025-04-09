from typing import Sequence
from itertools import pairwise
import torch


class UNet3D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        num_filters_down: Sequence[int] = (16, 32, 64, 128),
        num_filters_bottleneck: int = 256,
        num_filters_up: Sequence[int] = (4, 16, 32, 64),
        dropout_p: float = 0.1
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._num_classes = num_classes
        self._num_filters_bottleneck = num_filters_bottleneck
        self._num_filters_down = tuple(num_filters_down)
        self._num_filters_up = tuple(num_filters_up) + (num_filters_bottleneck,)

        self._init_conv = UNet3DResidualBlock(self._in_channels, self._num_filters_down[0], dropout_p, stride=1)

        self._encoder_list = torch.nn.ModuleList([
            UNet3DResidualBlock(in_f, out_f, dropout_p, stride=2)
        for in_f, out_f in pairwise(self._num_filters_down + (num_filters_bottleneck,))])

        self._decoder_list = torch.nn.ModuleList([
            UNet3DUpBlock(in_f, skip_f, out_f, dropout_p)
        for (in_f, out_f), skip_f in zip(pairwise(reversed(self._num_filters_up)), reversed(self._num_filters_down))])

        self._final_conv = torch.nn.Conv3d(self._num_filters_up[0], num_classes, kernel_size=1)

    def encoder(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        skips = [self._init_conv(x)]
        for enc in self._encoder_list:
            skips.append(enc(skips[-1]))

        return skips

    def decoder(self, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        current_feature = skips[-1]
        encoder_skips = skips[:-1][::-1]

        for current_skip, decoder_block in zip(encoder_skips, self._decoder_list):
            current_feature = decoder_block(current_feature, current_skip)

        output = self._final_conv(current_feature)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        y = self.decoder(skips)

        return y


class UNet3DResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
        stride: int = 1
    ) -> None:
        super().__init__()
        self._conv0 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self._inorm0 = torch.nn.InstanceNorm3d(out_channels, affine=True)
        self._conv1 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self._inorm1 = torch.nn.InstanceNorm3d(out_channels, affine=True)
        self._conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self._inorm2 = torch.nn.InstanceNorm3d(out_channels, affine=True)
        self._dropout = torch.nn.Dropout3d(dropout_p) if dropout_p > 0 else torch.nn.Identity()
        self._act_fn = torch.nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_0 = self._act_fn(self._inorm0(self._conv0(x)))
        y_1 = self._act_fn(self._inorm1(self._conv1(y_0)))
        y_1_d = self._dropout(y_1)
        y_2 = self._act_fn(self._inorm2(self._conv2(y_1_d)))
        y = y_0 + y_2

        return y


class UNet3DUpBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self._tconv = torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self._res_block = UNet3DResidualBlock(skip_channels+out_channels, out_channels, dropout_p)

    def forward(self, x, skip):
        x = self._tconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self._res_block(x)
        return x


def smoke_test():
    torch.manual_seed(0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = UNet3D(num_classes=1).to(device)
    x = torch.randn(1, 1, 128, 128, 128).to(device)
    y = net(x)
    loss = torch.nn.functional.mse_loss(y, x)
    loss.backward()
    print(loss)


def benchmark():
    from time import perf_counter  # pylint: disable=import-outside-toplevel

    torch.manual_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = UNet3D(num_classes=1).to(device)
    x = torch.randn(1, 1, 128, 128, 128).to(device)

    # Warmup
    y = net(x)

    # Benchmark
    num_iters = 10
    start = perf_counter()
    for _ in range(num_iters):
        net.zero_grad()
        y = net(x)
        loss = torch.nn.functional.mse_loss(y, x)
        loss.backward()
    end = perf_counter()
    print(f"Time per training iteration: {(end - start) / num_iters:.3f} seconds")


if __name__ == "__main__":
    smoke_test()
    benchmark()
