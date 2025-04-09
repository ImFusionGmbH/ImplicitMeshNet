#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
torch::Tensor grid_feature_projection_cuda_forward(
    const torch::Tensor& vertices,
    const torch::Tensor& features,
    int volume_size);

std::vector<torch::Tensor> grid_feature_projection_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& vertices,
    const torch::Tensor& features,
    int volume_size);

// Python-visible functions
torch::Tensor grid_feature_projection_forward(
    const torch::Tensor& vertices,
    const torch::Tensor& features,
    int volume_size) {
  // Check inputs
  TORCH_CHECK(vertices.dim() == 3, "vertices must be a 3D tensor");
  TORCH_CHECK(vertices.size(2) == 3, "vertices must have shape (B, N, 3)");
  TORCH_CHECK(features.dim() == 3, "features must be a 3D tensor");
  TORCH_CHECK(
      features.size(0) == vertices.size(0) &&
          features.size(1) == vertices.size(1),
      "features and vertices must have same batch size and number of vertices");

  // Get tensor info
  const int batch_size = vertices.size(0);

  // Create output volume with batch dimension
  auto volume = torch::zeros(
      {batch_size, features.size(2), volume_size, volume_size, volume_size},
      vertices.options());

  // Process each batch separately
  for (int b = 0; b < batch_size; b++) {
    auto vertices_batch = vertices[b];
    auto features_batch = features[b];

    // Call CUDA kernel for this batch
    auto batch_result = grid_feature_projection_cuda_forward(
        vertices_batch, features_batch, volume_size);

    // Copy result to output tensor
    volume[b].copy_(batch_result.squeeze(0));
  }

  return volume;
}

std::vector<torch::Tensor> grid_feature_projection_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& vertices,
    const torch::Tensor& features,
    int volume_size) {
  // Get tensor info
  const int batch_size = vertices.size(0);

  // Create output gradients with batch dimension
  auto grad_vertices = torch::zeros_like(vertices);
  auto grad_features = torch::zeros_like(features);

  // Process each batch separately
  for (int b = 0; b < batch_size; b++) {
    auto grad_output_batch =
        grad_output[b].unsqueeze(0); // Add batch dim for CUDA kernel
    auto vertices_batch = vertices[b];
    auto features_batch = features[b];

    // Call CUDA kernel for this batch
    auto batch_result = grid_feature_projection_cuda_backward(
        grad_output_batch, vertices_batch, features_batch, volume_size);

    // Copy results to output tensors
    grad_vertices[b].copy_(batch_result[0]);
    grad_features[b].copy_(batch_result[1]);
  }

  return {grad_vertices, grad_features};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &grid_feature_projection_forward,
      "Grid Feature Projection forward (CUDA)");
  m.def(
      "backward",
      &grid_feature_projection_backward,
      "Grid Feature Projection backward (CUDA)");
}