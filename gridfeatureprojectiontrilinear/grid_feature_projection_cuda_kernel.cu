#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

template <typename scalar_t>
__global__ void grid_feature_projection_cuda_forward_kernel(
    const scalar_t* __restrict__ vertices,
    const scalar_t* __restrict__ features,
    scalar_t* __restrict__ volume,
    const int num_vertices,
    const int num_features,
    const int volume_size) {
  const int vertex_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (vertex_idx < num_vertices) {
    // Get vertex coordinates (scaled to [-1, 1])
    scalar_t x = vertices[vertex_idx * 3];
    scalar_t y = vertices[vertex_idx * 3 + 1];
    scalar_t z = vertices[vertex_idx * 3 + 2];

    // Clamp to [-1, 1]
    x = x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x);
    y = y < -1.0 ? -1.0 : (y > 1.0 ? 1.0 : y);
    z = z < -1.0 ? -1.0 : (z > 1.0 ? 1.0 : z);

    // Convert to volume indices [0, volume_size-1]
    scalar_t idx_x = (x + 1.0) * (volume_size - 1) / 2.0;
    scalar_t idx_y = (y + 1.0) * (volume_size - 1) / 2.0;
    scalar_t idx_z = (z + 1.0) * (volume_size - 1) / 2.0;

    // Get floor and ceil indices
    int x0 = static_cast<int>(floorf(idx_x));
    int y0 = static_cast<int>(floorf(idx_y));
    int z0 = static_cast<int>(floorf(idx_z));

    // Clamp to valid range
    x0 = x0 < 0 ? 0 : (x0 > volume_size - 2 ? volume_size - 2 : x0);
    y0 = y0 < 0 ? 0 : (y0 > volume_size - 2 ? volume_size - 2 : y0);
    z0 = z0 < 0 ? 0 : (z0 > volume_size - 2 ? volume_size - 2 : z0);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    // Compute interpolation weights
    scalar_t wx = idx_x - x0;
    scalar_t wy = idx_y - y0;
    scalar_t wz = idx_z - z0;

    // Compute the 8 weights for trilinear interpolation
    scalar_t w000 = (1 - wx) * (1 - wy) * (1 - wz);
    scalar_t w001 = (1 - wx) * (1 - wy) * wz;
    scalar_t w010 = (1 - wx) * wy * (1 - wz);
    scalar_t w011 = (1 - wx) * wy * wz;
    scalar_t w100 = wx * (1 - wy) * (1 - wz);
    scalar_t w101 = wx * (1 - wy) * wz;
    scalar_t w110 = wx * wy * (1 - wz);
    scalar_t w111 = wx * wy * wz;

    // For each feature channel
    for (int c = 0; c < num_features; c++) {
      scalar_t feature_val = features[vertex_idx * num_features + c];

      // Add weighted features to the 8 surrounding voxels
      // Modified to match the Metal implementation: volume has shape [C, X, Y,
      // Z] with indices [c, x, y, z] instead of [c, z, y, x]
      atomicAdd(
          &volume
              [c * volume_size * volume_size * volume_size +
               x0 * volume_size * volume_size + y0 * volume_size + z0],
          feature_val * w000);
      atomicAdd(
          &volume
              [c * volume_size * volume_size * volume_size +
               x0 * volume_size * volume_size + y0 * volume_size + z1],
          feature_val * w001);
      atomicAdd(
          &volume
              [c * volume_size * volume_size * volume_size +
               x0 * volume_size * volume_size + y1 * volume_size + z0],
          feature_val * w010);
      atomicAdd(
          &volume
              [c * volume_size * volume_size * volume_size +
               x0 * volume_size * volume_size + y1 * volume_size + z1],
          feature_val * w011);
      atomicAdd(
          &volume
              [c * volume_size * volume_size * volume_size +
               x1 * volume_size * volume_size + y0 * volume_size + z0],
          feature_val * w100);
      atomicAdd(
          &volume
              [c * volume_size * volume_size * volume_size +
               x1 * volume_size * volume_size + y0 * volume_size + z1],
          feature_val * w101);
      atomicAdd(
          &volume
              [c * volume_size * volume_size * volume_size +
               x1 * volume_size * volume_size + y1 * volume_size + z0],
          feature_val * w110);
      atomicAdd(
          &volume
              [c * volume_size * volume_size * volume_size +
               x1 * volume_size * volume_size + y1 * volume_size + z1],
          feature_val * w111);
    }
  }
}

template <typename scalar_t>
__global__ void grid_feature_projection_cuda_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ vertices,
    const scalar_t* __restrict__ features,
    scalar_t* __restrict__ grad_vertices,
    scalar_t* __restrict__ grad_features,
    const int num_vertices,
    const int num_features,
    const int volume_size) {
  const int vertex_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (vertex_idx < num_vertices) {
    // Get vertex coordinates (scaled to [-1, 1])
    scalar_t x = vertices[vertex_idx * 3];
    scalar_t y = vertices[vertex_idx * 3 + 1];
    scalar_t z = vertices[vertex_idx * 3 + 2];

    // Clamp to [-1, 1]
    x = x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x);
    y = y < -1.0 ? -1.0 : (y > 1.0 ? 1.0 : y);
    z = z < -1.0 ? -1.0 : (z > 1.0 ? 1.0 : z);

    // Convert to volume indices [0, volume_size-1]
    scalar_t idx_x = (x + 1.0) * (volume_size - 1) / 2.0;
    scalar_t idx_y = (y + 1.0) * (volume_size - 1) / 2.0;
    scalar_t idx_z = (z + 1.0) * (volume_size - 1) / 2.0;

    // Get floor and ceil indices
    int x0 = static_cast<int>(floorf(idx_x));
    int y0 = static_cast<int>(floorf(idx_y));
    int z0 = static_cast<int>(floorf(idx_z));

    // Clamp to valid range
    x0 = x0 < 0 ? 0 : (x0 > volume_size - 2 ? volume_size - 2 : x0);
    y0 = y0 < 0 ? 0 : (y0 > volume_size - 2 ? volume_size - 2 : y0);
    z0 = z0 < 0 ? 0 : (z0 > volume_size - 2 ? volume_size - 2 : z0);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    // Compute interpolation weights
    scalar_t wx = idx_x - x0;
    scalar_t wy = idx_y - y0;
    scalar_t wz = idx_z - z0;

    // Compute the 8 weights for trilinear interpolation
    scalar_t w000 = (1 - wx) * (1 - wy) * (1 - wz);
    scalar_t w001 = (1 - wx) * (1 - wy) * wz;
    scalar_t w010 = (1 - wx) * wy * (1 - wz);
    scalar_t w011 = (1 - wx) * wy * wz;
    scalar_t w100 = wx * (1 - wy) * (1 - wz);
    scalar_t w101 = wx * (1 - wy) * wz;
    scalar_t w110 = wx * wy * (1 - wz);
    scalar_t w111 = wx * wy * wz;

    // Compute gradients for features
    for (int c = 0; c < num_features; c++) {
      scalar_t grad_feature = 0;

      // Accumulate gradients from the 8 surrounding voxels
      // Modified to match Metal implementation layout [C, X, Y, Z]
      grad_feature +=
          grad_output
              [c * volume_size * volume_size * volume_size +
               x0 * volume_size * volume_size + y0 * volume_size + z0] *
          w000;
      grad_feature +=
          grad_output
              [c * volume_size * volume_size * volume_size +
               x0 * volume_size * volume_size + y0 * volume_size + z1] *
          w001;
      grad_feature +=
          grad_output
              [c * volume_size * volume_size * volume_size +
               x0 * volume_size * volume_size + y1 * volume_size + z0] *
          w010;
      grad_feature +=
          grad_output
              [c * volume_size * volume_size * volume_size +
               x0 * volume_size * volume_size + y1 * volume_size + z1] *
          w011;
      grad_feature +=
          grad_output
              [c * volume_size * volume_size * volume_size +
               x1 * volume_size * volume_size + y0 * volume_size + z0] *
          w100;
      grad_feature +=
          grad_output
              [c * volume_size * volume_size * volume_size +
               x1 * volume_size * volume_size + y0 * volume_size + z1] *
          w101;
      grad_feature +=
          grad_output
              [c * volume_size * volume_size * volume_size +
               x1 * volume_size * volume_size + y1 * volume_size + z0] *
          w110;
      grad_feature +=
          grad_output
              [c * volume_size * volume_size * volume_size +
               x1 * volume_size * volume_size + y1 * volume_size + z1] *
          w111;

      grad_features[vertex_idx * num_features + c] = grad_feature;
    }

    // Initialize vertex gradients to zero
    scalar_t grad_x = 0, grad_y = 0, grad_z = 0;

    // Only compute vertex gradients if within valid range
    if (x > -1.0 && x < 1.0 && y > -1.0 && y < 1.0 && z > -1.0 && z < 1.0) {
      float scale_factor = (volume_size - 1) / 2.0f;

      for (int c = 0; c < num_features; c++) {
        scalar_t feature_val = features[vertex_idx * num_features + c];

        // Get gradients from output - modified to match Metal layout [C, X, Y,
        // Z]
        scalar_t g000 = grad_output
            [c * volume_size * volume_size * volume_size +
             x0 * volume_size * volume_size + y0 * volume_size + z0];
        scalar_t g001 = grad_output
            [c * volume_size * volume_size * volume_size +
             x0 * volume_size * volume_size + y0 * volume_size + z1];
        scalar_t g010 = grad_output
            [c * volume_size * volume_size * volume_size +
             x0 * volume_size * volume_size + y1 * volume_size + z0];
        scalar_t g011 = grad_output
            [c * volume_size * volume_size * volume_size +
             x0 * volume_size * volume_size + y1 * volume_size + z1];
        scalar_t g100 = grad_output
            [c * volume_size * volume_size * volume_size +
             x1 * volume_size * volume_size + y0 * volume_size + z0];
        scalar_t g101 = grad_output
            [c * volume_size * volume_size * volume_size +
             x1 * volume_size * volume_size + y0 * volume_size + z1];
        scalar_t g110 = grad_output
            [c * volume_size * volume_size * volume_size +
             x1 * volume_size * volume_size + y1 * volume_size + z0];
        scalar_t g111 = grad_output
            [c * volume_size * volume_size * volume_size +
             x1 * volume_size * volume_size + y1 * volume_size + z1];

        // Compute gradients for each corner with respect to x, y, z
        scalar_t dw000_dx = -(1 - wy) * (1 - wz) * scale_factor;
        scalar_t dw000_dy = -(1 - wx) * (1 - wz) * scale_factor;
        scalar_t dw000_dz = -(1 - wx) * (1 - wy) * scale_factor;

        scalar_t dw001_dx = -(1 - wy) * wz * scale_factor;
        scalar_t dw001_dy = -(1 - wx) * wz * scale_factor;
        scalar_t dw001_dz = (1 - wx) * (1 - wy) * scale_factor;

        scalar_t dw010_dx = -(wy) * (1 - wz) * scale_factor;
        scalar_t dw010_dy = (1 - wx) * (1 - wz) * scale_factor;
        scalar_t dw010_dz = -(1 - wx) * wy * scale_factor;

        scalar_t dw011_dx = -(wy)*wz * scale_factor;
        scalar_t dw011_dy = (1 - wx) * wz * scale_factor;
        scalar_t dw011_dz = (1 - wx) * wy * scale_factor;

        scalar_t dw100_dx = (1 - wy) * (1 - wz) * scale_factor;
        scalar_t dw100_dy = -wx * (1 - wz) * scale_factor;
        scalar_t dw100_dz = -wx * (1 - wy) * scale_factor;

        scalar_t dw101_dx = (1 - wy) * wz * scale_factor;
        scalar_t dw101_dy = -wx * wz * scale_factor;
        scalar_t dw101_dz = wx * (1 - wy) * scale_factor;

        scalar_t dw110_dx = wy * (1 - wz) * scale_factor;
        scalar_t dw110_dy = wx * (1 - wz) * scale_factor;
        scalar_t dw110_dz = -wx * wy * scale_factor;

        scalar_t dw111_dx = wy * wz * scale_factor;
        scalar_t dw111_dy = wx * wz * scale_factor;
        scalar_t dw111_dz = wx * wy * scale_factor;

        // Accumulate gradients
        grad_x += feature_val *
            (dw000_dx * g000 + dw001_dx * g001 + dw010_dx * g010 +
             dw011_dx * g011 + dw100_dx * g100 + dw101_dx * g101 +
             dw110_dx * g110 + dw111_dx * g111);

        grad_y += feature_val *
            (dw000_dy * g000 + dw001_dy * g001 + dw010_dy * g010 +
             dw011_dy * g011 + dw100_dy * g100 + dw101_dy * g101 +
             dw110_dy * g110 + dw111_dy * g111);

        grad_z += feature_val *
            (dw000_dz * g000 + dw001_dz * g001 + dw010_dz * g010 +
             dw011_dz * g011 + dw100_dz * g100 + dw101_dz * g101 +
             dw110_dz * g110 + dw111_dz * g111);
      }
    }

    grad_vertices[vertex_idx * 3] = grad_x;
    grad_vertices[vertex_idx * 3 + 1] = grad_y;
    grad_vertices[vertex_idx * 3 + 2] = grad_z;
  }
}

torch::Tensor grid_feature_projection_cuda_forward(
    const torch::Tensor& vertices,
    const torch::Tensor& features,
    int volume_size) {
  const int num_vertices = vertices.size(0);
  const int num_features = features.size(1);

  // Create output volume
  auto volume = torch::zeros(
      {1, num_features, volume_size, volume_size, volume_size},
      vertices.options());

  const int threads = 256;
  const int blocks = (num_vertices + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(
      vertices.scalar_type(), "grid_feature_projection_forward_cuda", ([&] {
        grid_feature_projection_cuda_forward_kernel<scalar_t>
            <<<blocks, threads>>>(
                vertices.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                volume.data_ptr<scalar_t>(),
                num_vertices,
                num_features,
                volume_size);
      }));

  return volume;
}

std::vector<torch::Tensor> grid_feature_projection_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& vertices,
    const torch::Tensor& features,
    int volume_size) {
  const int num_vertices = vertices.size(0);
  const int num_features = features.size(1);

  // Create output gradients
  auto grad_vertices = torch::zeros_like(vertices);
  auto grad_features = torch::zeros_like(features);

  const int threads = 256;
  const int blocks = (num_vertices + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(
      vertices.scalar_type(), "grid_feature_projection_backward_cuda", ([&] {
        grid_feature_projection_cuda_backward_kernel<scalar_t>
            <<<blocks, threads>>>(
                grad_output.data_ptr<scalar_t>(),
                vertices.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(),
                grad_vertices.data_ptr<scalar_t>(),
                grad_features.data_ptr<scalar_t>(),
                num_vertices,
                num_features,
                volume_size);
      }));

  return {grad_vertices, grad_features};
}