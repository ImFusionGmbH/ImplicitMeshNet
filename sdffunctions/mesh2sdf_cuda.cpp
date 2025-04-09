#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA function
torch::Tensor mesh2sdf_cuda_forward(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    int volume_size);

// Python-visible function
torch::Tensor mesh2sdf_forward(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    int volume_size) {
  // Check inputs
  TORCH_CHECK(vertices.dim() == 2, "vertices must be a 2D tensor");
  TORCH_CHECK(vertices.size(1) == 3, "vertices must have shape (N, 3)");
  TORCH_CHECK(faces.dim() == 2, "faces must be a 2D tensor");
  TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
  TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
  TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
  TORCH_CHECK(vertices.is_contiguous(), "vertices must be contiguous");
  TORCH_CHECK(faces.is_contiguous(), "faces must be contiguous");
  TORCH_CHECK(
      vertices.dtype() == torch::kFloat32, "vertices must be a float32 tensor");
  TORCH_CHECK(faces.dtype() == torch::kInt32, "faces must be a int32 tensor");

  // Call CUDA kernel
  return mesh2sdf_cuda_forward(vertices, faces, volume_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mesh2sdf_forward, "Mesh to SDF forward (CUDA)");
}
