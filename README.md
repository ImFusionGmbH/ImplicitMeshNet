# ImplicitMeshNet

Deep Implicit Neural Representations for End-to-End Anatomical Shape Estimation from Volumetric Images

## Abstract

We present an end-to-end approach for anatomical shape estimation from volumetric images using deep implicit neural
representations. Our neural network directly reconstructs shapes as 3D meshes and is trained on voxel-based segmentation
maps by utilizing a deep signed distance field transform, eliminating the need for ground truth meshes. Evaluated on
cardiac CT scans, our method achieves a Dice score of 0.87 for the extraction of the left atrium and ventricle, while
maintaining anatomical fidelity. This enables more accurate cardiac modeling for visualization and downstream analysis
in clinical settings.

## Installation

### Prerequisites

```bash
python3 -m venv env  # create virtual environment
source env/bin/activate
pip install -r requirements.txt  # install dependencies
```

### Custom PyTorch layers

This repository contains custom C++ extension layers for PyTorch (`GridFeatureProjectionTrilinear` and
`Mesh2SDF`), inculding GPU implementations for both CUDA and Metal. Compilation of these layers needs CUDA 12.8 or
Metal 3.2 (macoS 15 and newer).

#### macOS

The UNet architecture makes use of `ConvTranspose3D`, which does not come with MPS support for GPU acceleration in
PyTorch 2.6.0. You can install Pytorch from
[this personal branch](https://github.com/mlaves/pytorch/tree/convtranspose_mps_remove_check) to get full MPS
acceleration for `ConvTranspose3D`, or fall back to CPU inference. Hopefully, my
[my PR](https://github.com/pytorch/pytorch/pull/145366) will get merged soon.

## Inference

This repository contains all model snapshots that were used to generate the results reported in the paper. As of now,
the model architecture does not support multi-mesh inference and therefore, two separate networks have been trained for
LA and LV shape estimation. To reproduce the results in the paper, download the
[MMWHS test set](https://zmiclab.github.io/zxh/0/mmwhs/) and run

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # only needed for MPS on macOS
python implicitmeshnet_eval.py \
    --implicitmeshnet_config=configs_implicitmeshnet/2025-03-31_13-45-41.json \
    --test_dir_in=/path/to/mmwhs_test \
    --test_dir_out=test_out_lv \
    --device=cpu  # or <cuda|mps>

python implicitmeshnet_eval.py \
    --implicitmeshnet_config=configs_implicitmeshnet/2025-03-30_10-03-29.json \
    --test_dir_in=/path/to/mmwhs_test \
    --test_dir_out=test_out_la \
    --device=cpu  # or <cuda|mps>
```

If you also want to generate the Dice scores for the test set, you need to use the Windows `.exe` that comes with the
test set and decrypts the private ground truth segmentations. We successfully used that on macOS using `wine`:

```bash
brew install --cask wine-stable  # or use apt on Linux
python mmwhs_create_submission.py --input_dir_lv=test_out_lv --input_dir_la=test_out_la test_out
```

## Training

Please have a look at `train.slurm` to see how both the pretraining of `DeepRepresentationNetwork`, as well as the
actual training of `ImplicitMeshNet` are invoked.

## Contact

Max-Heinrich Laves, ImFusion, Munich, Germany
