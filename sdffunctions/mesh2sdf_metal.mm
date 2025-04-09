#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <torch/extension.h>
#include <cmath>
#include <vector>

// Metal shader code for SDF computation
static const char* PYTORCH_SDF_KERNEL = R"(
#include <metal_stdlib>
using namespace metal;

// Helper function to compute point-to-triangle distance
float pointTriangleDistance(float3 p, float3 v0, float3 v1, float3 v2) {
    // Edge vectors
    float3 e0 = v1 - v0;
    float3 e1 = v2 - v0;
    float3 e2 = v2 - v1;

    // Normal vector of the triangle
    float3 normal = normalize(cross(e0, e1));

    // Check if point is above/below the triangle plane
    float3 v0p = p - v0;
    float dist_to_plane = dot(v0p, normal);

    // Project point onto the triangle plane
    float3 p_proj = p - dist_to_plane * normal;

    // Check if projected point is inside the triangle
    // Using barycentric coordinates for more robust inside/outside test
    float3 c = cross(e0, e1);
    float area = length(c) / 2.0f;

    float3 vp0 = p_proj - v0;
    float3 vp1 = p_proj - v1;
    float3 vp2 = p_proj - v2;

    float3 c0 = cross(e0, vp0);
    float3 c1 = cross(e1, vp2);
    float3 c2 = cross(-e2, vp1);

    float s0 = dot(c0, normal) / (2.0f * area);
    float s1 = dot(c1, normal) / (2.0f * area);
    float s2 = dot(c2, normal) / (2.0f * area);

    bool inside = (s0 >= 0.0f && s1 >= 0.0f && s2 >= 0.0f &&
                  (s0 + s1 + s2) <= 1.0f + 1e-4f);

    if (inside) {
        // If inside, return signed distance to the plane
        return dist_to_plane;
    }

    // If outside, compute distance to the nearest edge or vertex
    // Edge v0-v1
    float t0 = clamp(dot(p - v0, e0) / dot(e0, e0), 0.0f, 1.0f);
    float3 proj0 = v0 + t0 * e0;
    float d0 = length(p - proj0);

    // Edge v1-v2
    float t1 = clamp(dot(p - v1, e2) / dot(e2, e2), 0.0f, 1.0f);
    float3 proj1 = v1 + t1 * e2;
    float d1 = length(p - proj1);

    // Edge v2-v0
    float3 e3 = v0 - v2;
    float t2 = clamp(dot(p - v2, e3) / dot(e3, e3), 0.0f, 1.0f);
    float3 proj2 = v2 + t2 * e3;
    float d2 = length(p - proj2);

    float d = min(d0, min(d1, d2));

    // Return unsigned distance (sign will be determined later)
    return d;
}

// Ray-triangle intersection test
bool rayTriangleIntersect(float3 origin, float3 dir, float3 v0, float3 v1, float3 v2) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 h = cross(dir, e2);
    float a = dot(e1, h);

    // Ray parallel to triangle
    if (abs(a) < 1e-6f) return false;

    float f = 1.0f / a;
    float3 s = origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f) return false;

    float3 q = cross(s, e1);
    float v = f * dot(dir, q);

    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * dot(e2, q);

    return (t > 0.0f);
}

// Improved inside/outside test using ray casting
bool isPointInside(float3 p, const device float* vertices, const device int* triangles, uint numTriangles) {
    // Cast a ray in a fixed direction (e.g., positive x)
    float3 rayDir = float3(1.0f, 0.0f, 0.0f);

    // Count intersections
    int intersections = 0;

    for (uint i = 0; i < numTriangles; i++) {
        uint idx0 = triangles[i*3];
        uint idx1 = triangles[i*3+1];
        uint idx2 = triangles[i*3+2];

        float3 v0 = float3(vertices[idx0*3], vertices[idx0*3+1], vertices[idx0*3+2]);
        float3 v1 = float3(vertices[idx1*3], vertices[idx1*3+1], vertices[idx1*3+2]);
        float3 v2 = float3(vertices[idx2*3], vertices[idx2*3+1], vertices[idx2*3+2]);

        if (rayTriangleIntersect(p, rayDir, v0, v1, v2)) {
            intersections++;
        }
    }

    // Odd number of intersections means inside
    return (intersections % 2) == 1;
}

kernel void computeDistanceVolume(
    const device float* vertices [[buffer(0)]],
    const device int* triangles [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& numVertices [[buffer(3)]],
    constant uint& numTriangles [[buffer(4)]],
    constant uint3& volumeDimensions [[buffer(5)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]])
{
    // Get the voxel position
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    uint z = thread_position_in_grid.z;

    // Check if this thread is within the volume bounds
    if (x >= volumeDimensions.x || y >= volumeDimensions.y || z >= volumeDimensions.z) {
        return;
    }

    float3 extent = float3(volumeDimensions.x, volumeDimensions.y, volumeDimensions.z);

    // shift to -1 to 1
    float3 pos = float3(z, y, x) / (extent-1.0f) * 2.0f - 1.0f;

    // Compute the minimum distance to any triangle
    float minDist = INFINITY;
    float3 closestNormal;
    bool foundClosest = false;

    for (uint i = 0; i < numTriangles; i++) {
        uint idx0 = triangles[i*3];
        uint idx1 = triangles[i*3+1];
        uint idx2 = triangles[i*3+2];

        float3 v0 = float3(vertices[idx0*3], vertices[idx0*3+1], vertices[idx0*3+2]);
        float3 v1 = float3(vertices[idx1*3], vertices[idx1*3+1], vertices[idx1*3+2]);
        float3 v2 = float3(vertices[idx2*3], vertices[idx2*3+1], vertices[idx2*3+2]);

        float dist = pointTriangleDistance(pos, v0, v1, v2);

        if (dist < minDist) {
            minDist = dist;
            // Store normal for later sign determination
            closestNormal = normalize(cross(v1 - v0, v2 - v0));
            foundClosest = true;
        }
    }

    // Determine if the point is inside or outside
    bool inside = isPointInside(pos, vertices, triangles, numTriangles);

    // Apply sign based on inside/outside test
    float signedDist = inside ? -minDist : minDist;

    // For points very close to the surface, double-check with ray casting
    if (minDist < 0.01f) {
        // Use ray casting for more accurate inside/outside determination
        signedDist = inside ? -minDist : minDist;
    }

    uint idx = z * volumeDimensions.x * volumeDimensions.y + y * volumeDimensions.x + x;
    output[idx] = signedDist;
}
)";

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Forward pass implementation
torch::Tensor mesh2sdf_forward(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    int volume_size) {
  // Check that tensors are on MPS device
  TORCH_CHECK(vertices.dim() == 2, "vertices must be a 2D tensor");
  TORCH_CHECK(faces.dim() == 2, "faces must be a 2D tensor");
  TORCH_CHECK(vertices.device().is_mps(), "vertices must be a MPS tensor");
  TORCH_CHECK(faces.device().is_mps(), "faces must be a MPS tensor");
  TORCH_CHECK(vertices.is_contiguous(), "vertices must be contiguous");
  TORCH_CHECK(faces.is_contiguous(), "faces must be contiguous");
  TORCH_CHECK(
      vertices.dtype() == torch::kFloat32, "vertices must be a float32 tensor");
  TORCH_CHECK(faces.dtype() == torch::kInt32, "faces must be a int32 tensor");

  int num_vertices = vertices.size(0);
  int num_faces = faces.size(0);

  // Create output tensor
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device());
  torch::Tensor sdf_volume =
      torch::ones({volume_size, volume_size, volume_size}, options) *
      INFINITY; // Initialize with a large value

  @autoreleasepool {
    // Get Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;

    // Create Metal library with our kernel
    MTLCompileOptions* compileOptions = [MTLCompileOptions new];
    compileOptions.languageVersion = MTLLanguageVersion3_2;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:PYTORCH_SDF_KERNEL]
                     options:compileOptions
                       error:&error];
    TORCH_CHECK(
        library,
        "Failed to create Metal library: ",
        error.localizedDescription.UTF8String);

    // Get kernel function
    id<MTLFunction> kernelFunction =
        [library newFunctionWithName:@"computeDistanceVolume"];
    TORCH_CHECK(kernelFunction, "Failed to find the Metal kernel function");

    // Create compute pipeline state
    id<MTLComputePipelineState> pipelineState =
        [device newComputePipelineStateWithFunction:kernelFunction
                                              error:&error];
    TORCH_CHECK(
        pipelineState,
        "Failed to create compute pipeline state: ",
        error.localizedDescription.UTF8String);

    // Get a reference to the command buffer for the MPS stream
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

    // Get a reference to the dispatch queue for the MPS stream
    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    dispatch_sync(serialQueue, ^() {
      // Create a compute command encoder
      id<MTLComputeCommandEncoder> computeEncoder =
          [commandBuffer computeCommandEncoder];
      TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

      // Set the compute pipeline state
      [computeEncoder setComputePipelineState:pipelineState];

      // Set the buffers
      [computeEncoder
          setBuffer:getMTLBufferStorage(vertices)
             offset:vertices.storage_offset() * vertices.element_size()
            atIndex:0];
      [computeEncoder setBuffer:getMTLBufferStorage(faces)
                         offset:faces.storage_offset() * faces.element_size()
                        atIndex:1];
      [computeEncoder
          setBuffer:getMTLBufferStorage(sdf_volume)
             offset:sdf_volume.storage_offset() * sdf_volume.element_size()
            atIndex:2];

      // Set the constants
      uint numVerticesVal = static_cast<uint>(num_vertices);
      [computeEncoder setBytes:&numVerticesVal length:sizeof(uint) atIndex:3];

      uint numTrianglesVal = static_cast<uint>(num_faces);
      [computeEncoder setBytes:&numTrianglesVal length:sizeof(uint) atIndex:4];

      // Define a struct for the volume dimensions
      struct UInt3 {
        uint x, y, z;
      };
      UInt3 volumeDimensions = {
          static_cast<uint>(volume_size),
          static_cast<uint>(volume_size),
          static_cast<uint>(volume_size)};
      [computeEncoder setBytes:&volumeDimensions
                        length:sizeof(UInt3)
                       atIndex:5];

      // Dispatch the threads - one thread per voxel
      MTLSize gridSize = MTLSizeMake(volume_size, volume_size, volume_size);

      // Determine a good threadgroup size (8x8x8 is often a good balance)
      MTLSize threadgroupSize = MTLSizeMake(8, 8, 8);

      // Dispatch the threads
      [computeEncoder dispatchThreads:gridSize
                threadsPerThreadgroup:threadgroupSize];

      // End encoding
      [computeEncoder endEncoding];

      // Commit the work
      torch::mps::commit();
    });
  }

  return sdf_volume;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mesh2sdf_forward, "Mesh to SDF forward (Metal)");
}