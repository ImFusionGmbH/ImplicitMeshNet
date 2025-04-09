#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <vector>

// Helper function to compute point-to-triangle distance
__device__ float
pointTriangleDistance(float3 p, float3 v0, float3 v1, float3 v2) {
  // Edge vectors
  float3 e0 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
  float3 e1 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
  float3 e2 = make_float3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);

  // Normal vector of the triangle
  float3 normal;
  normal.x = e0.y * e1.z - e0.z * e1.y;
  normal.y = e0.z * e1.x - e0.x * e1.z;
  normal.z = e0.x * e1.y - e0.y * e1.x;
  float normal_length =
      sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
  normal.x /= normal_length;
  normal.y /= normal_length;
  normal.z /= normal_length;

  // Check if point is above/below the triangle plane
  float3 v0p = make_float3(p.x - v0.x, p.y - v0.y, p.z - v0.z);
  float dist_to_plane = v0p.x * normal.x + v0p.y * normal.y + v0p.z * normal.z;

  // Project point onto the triangle plane
  float3 p_proj = make_float3(
      p.x - dist_to_plane * normal.x,
      p.y - dist_to_plane * normal.y,
      p.z - dist_to_plane * normal.z);

  // Check if projected point is inside the triangle
  // Using barycentric coordinates for more robust inside/outside test
  float3 c = make_float3(
      e0.y * e1.z - e0.z * e1.y,
      e0.z * e1.x - e0.x * e1.z,
      e0.x * e1.y - e0.y * e1.x);
  float area = sqrtf(c.x * c.x + c.y * c.y + c.z * c.z) / 2.0f;

  float3 vp0 = make_float3(p_proj.x - v0.x, p_proj.y - v0.y, p_proj.z - v0.z);
  float3 vp1 = make_float3(p_proj.x - v1.x, p_proj.y - v1.y, p_proj.z - v1.z);
  float3 vp2 = make_float3(p_proj.x - v2.x, p_proj.y - v2.y, p_proj.z - v2.z);

  float3 c0 = make_float3(
      e0.y * vp0.z - e0.z * vp0.y,
      e0.z * vp0.x - e0.x * vp0.z,
      e0.x * vp0.y - e0.y * vp0.x);
  float3 c1 = make_float3(
      e1.y * vp2.z - e1.z * vp2.y,
      e1.z * vp2.x - e1.x * vp2.z,
      e1.x * vp2.y - e1.y * vp2.x);
  float3 c2 = make_float3(
      -e2.y * vp1.z + e2.z * vp1.y,
      -e2.z * vp1.x + e2.x * vp1.z,
      -e2.x * vp1.y + e2.y * vp1.x);

  float s0 =
      (c0.x * normal.x + c0.y * normal.y + c0.z * normal.z) / (2.0f * area);
  float s1 =
      (c1.x * normal.x + c1.y * normal.y + c1.z * normal.z) / (2.0f * area);
  float s2 =
      (c2.x * normal.x + c2.y * normal.y + c2.z * normal.z) / (2.0f * area);

  bool inside =
      (s0 >= 0.0f && s1 >= 0.0f && s2 >= 0.0f &&
       (s0 + s1 + s2) <= 1.0f + 1e-4f);

  if (inside) {
    // If inside, return signed distance to the plane
    return dist_to_plane;
  }

  // If outside, compute distance to the nearest edge or vertex
  // Edge v0-v1
  float t0 = fmaxf(
      0.0f,
      fminf(
          1.0f,
          ((p.x - v0.x) * e0.x + (p.y - v0.y) * e0.y + (p.z - v0.z) * e0.z) /
              (e0.x * e0.x + e0.y * e0.y + e0.z * e0.z)));
  float3 proj0 =
      make_float3(v0.x + t0 * e0.x, v0.y + t0 * e0.y, v0.z + t0 * e0.z);
  float d0 = sqrtf(
      (p.x - proj0.x) * (p.x - proj0.x) + (p.y - proj0.y) * (p.y - proj0.y) +
      (p.z - proj0.z) * (p.z - proj0.z));

  // Edge v1-v2
  float t1 = fmaxf(
      0.0f,
      fminf(
          1.0f,
          ((p.x - v1.x) * e2.x + (p.y - v1.y) * e2.y + (p.z - v1.z) * e2.z) /
              (e2.x * e2.x + e2.y * e2.y + e2.z * e2.z)));
  float3 proj1 =
      make_float3(v1.x + t1 * e2.x, v1.y + t1 * e2.y, v1.z + t1 * e2.z);
  float d1 = sqrtf(
      (p.x - proj1.x) * (p.x - proj1.x) + (p.y - proj1.y) * (p.y - proj1.y) +
      (p.z - proj1.z) * (p.z - proj1.z));

  // Edge v2-v0
  float3 e3 = make_float3(v0.x - v2.x, v0.y - v2.y, v0.z - v2.z);
  float t2 = fmaxf(
      0.0f,
      fminf(
          1.0f,
          ((p.x - v2.x) * e3.x + (p.y - v2.y) * e3.y + (p.z - v2.z) * e3.z) /
              (e3.x * e3.x + e3.y * e3.y + e3.z * e3.z)));
  float3 proj2 =
      make_float3(v2.x + t2 * e3.x, v2.y + t2 * e3.y, v2.z + t2 * e3.z);
  float d2 = sqrtf(
      (p.x - proj2.x) * (p.x - proj2.x) + (p.y - proj2.y) * (p.y - proj2.y) +
      (p.z - proj2.z) * (p.z - proj2.z));

  float d = fminf(d0, fminf(d1, d2));

  // Return unsigned distance (sign will be determined later)
  return d;
}

// Ray-triangle intersection test
__device__ bool rayTriangleIntersect(
    float3 origin,
    float3 dir,
    float3 v0,
    float3 v1,
    float3 v2) {
  float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
  float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
  float3 h = make_float3(
      dir.y * e2.z - dir.z * e2.y,
      dir.z * e2.x - dir.x * e2.z,
      dir.x * e2.y - dir.y * e2.x);
  float a = e1.x * h.x + e1.y * h.y + e1.z * h.z;

  // Ray parallel to triangle
  if (fabsf(a) < 1e-6f)
    return false;

  float f = 1.0f / a;
  float3 s = make_float3(origin.x - v0.x, origin.y - v0.y, origin.z - v0.z);
  float u = f * (s.x * h.x + s.y * h.y + s.z * h.z);

  if (u < 0.0f || u > 1.0f)
    return false;

  float3 q = make_float3(
      s.y * e1.z - s.z * e1.y,
      s.z * e1.x - s.x * e1.z,
      s.x * e1.y - s.y * e1.x);
  float v = f * (dir.x * q.x + dir.y * q.y + dir.z * q.z);

  if (v < 0.0f || u + v > 1.0f)
    return false;

  float t = f * (e2.x * q.x + e2.y * q.y + e2.z * q.z);

  return (t > 0.0f);
}

// Improved inside/outside test using ray casting
template <typename scalar_t>
__device__ bool isPointInside(
    float3 p,
    const scalar_t* vertices,
    const int* triangles,
    int numTriangles) {
  // Cast a ray in a fixed direction (e.g., positive x)
  float3 rayDir = make_float3(1.0f, 0.0f, 0.0f);

  // Count intersections
  int intersections = 0;

  for (int i = 0; i < numTriangles; i++) {
    int idx0 = triangles[i * 3];
    int idx1 = triangles[i * 3 + 1];
    int idx2 = triangles[i * 3 + 2];

    float3 v0 = make_float3(
        vertices[idx0 * 3], vertices[idx0 * 3 + 1], vertices[idx0 * 3 + 2]);
    float3 v1 = make_float3(
        vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]);
    float3 v2 = make_float3(
        vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]);

    if (rayTriangleIntersect(p, rayDir, v0, v1, v2)) {
      intersections++;
    }
  }

  // Odd number of intersections means inside
  return (intersections % 2) == 1;
}

template <typename scalar_t>
__global__ void compute_distance_volume_kernel(
    const scalar_t* __restrict__ vertices,
    const int* __restrict__ triangles,
    scalar_t* __restrict__ output,
    const int numVertices,
    const int numTriangles,
    const int volumeSize) {
  // Get the voxel position
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  // Check if this thread is within the volume bounds
  if (x >= volumeSize || y >= volumeSize || z >= volumeSize) {
    return;
  }

  // shift to -1 to 1
  float3 pos = make_float3(
      (float)z / (volumeSize - 1) * 2.0f - 1.0f,
      (float)y / (volumeSize - 1) * 2.0f - 1.0f,
      (float)x / (volumeSize - 1) * 2.0f - 1.0f);

  // Compute the minimum distance to any triangle
  float minDist = INFINITY;
  float3 closestNormal;
  bool foundClosest = false;

  for (int i = 0; i < numTriangles; i++) {
    int idx0 = triangles[i * 3];
    int idx1 = triangles[i * 3 + 1];
    int idx2 = triangles[i * 3 + 2];

    float3 v0 = make_float3(
        vertices[idx0 * 3], vertices[idx0 * 3 + 1], vertices[idx0 * 3 + 2]);
    float3 v1 = make_float3(
        vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]);
    float3 v2 = make_float3(
        vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]);

    float dist = pointTriangleDistance(pos, v0, v1, v2);

    if (dist < minDist) {
      minDist = dist;
      foundClosest = true;
    }
  }

  // Determine if the point is inside or outside
  bool inside = isPointInside<scalar_t>(pos, vertices, triangles, numTriangles);

  // Apply sign based on inside/outside test
  float signedDist = inside ? -minDist : minDist;

  // Set the voxel value
  int idx = z * volumeSize * volumeSize + y * volumeSize + x;
  output[idx] = signedDist;
}

torch::Tensor mesh2sdf_cuda_forward(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    int volume_size) {
  // Get dimensions
  const int num_vertices = vertices.size(0);
  const int num_faces = faces.size(0);

  // Create output tensor
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device());
  auto sdf_volume =
      torch::full({volume_size, volume_size, volume_size}, INFINITY, options);

  // Define the block and grid sizes
  dim3 threads_per_block(8, 8, 8);
  dim3 num_blocks(
      (volume_size + threads_per_block.x - 1) / threads_per_block.x,
      (volume_size + threads_per_block.y - 1) / threads_per_block.y,
      (volume_size + threads_per_block.z - 1) / threads_per_block.z);

  // Launch the kernel with scalar_type() instead of type()
  AT_DISPATCH_FLOATING_TYPES(
      vertices.scalar_type(), "compute_distance_volume_kernel", ([&] {
        compute_distance_volume_kernel<scalar_t>
            <<<num_blocks, threads_per_block>>>(
                vertices.data_ptr<scalar_t>(),
                faces.data_ptr<int>(),
                sdf_volume.data_ptr<scalar_t>(),
                num_vertices,
                num_faces,
                volume_size);
      }));

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  return sdf_volume;
}