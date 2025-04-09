#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <torch/extension.h>
#include <vector>

// Metal shader code for trilinear interpolation
static const char* GRID_FEATURE_PROJECTION_KERNEL = R"(
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void grid_feature_projection_forward(
    constant float* vertices [[buffer(0)]],
    constant float* features [[buffer(1)]],
    device atomic_float* volume [[buffer(2)]],
    constant int& num_vertices [[buffer(3)]],
    constant int& num_features [[buffer(4)]],
    constant int& volume_size [[buffer(5)]],
    uint vertex_idx [[thread_position_in_grid]])
{
    if (vertex_idx >= (uint)num_vertices) return;

    // Get vertex coordinates (scaled to [-1, 1])
    float x = vertices[vertex_idx * 3];
    float y = vertices[vertex_idx * 3 + 1];
    float z = vertices[vertex_idx * 3 + 2];

    // Clamp to [-1, 1]
    x = clamp(x, -1.0f, 1.0f);
    y = clamp(y, -1.0f, 1.0f);
    z = clamp(z, -1.0f, 1.0f);

    // Convert to volume indices [0, volume_size-1]
    float idx_x = (x + 1.0f) * (volume_size - 1) / 2.0f;
    float idx_y = (y + 1.0f) * (volume_size - 1) / 2.0f;
    float idx_z = (z + 1.0f) * (volume_size - 1) / 2.0f;

    // Get floor and ceil indices
    int x0 = int(floor(idx_x));
    int y0 = int(floor(idx_y));
    int z0 = int(floor(idx_z));

    // Clamp to valid range
    x0 = clamp(x0, 0, volume_size - 2);
    y0 = clamp(y0, 0, volume_size - 2);
    z0 = clamp(z0, 0, volume_size - 2);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    // Compute interpolation weights
    float wx = idx_x - float(x0);
    float wy = idx_y - float(y0);
    float wz = idx_z - float(z0);

    // Compute the 8 weights for trilinear interpolation
    float w000 = (1 - wx) * (1 - wy) * (1 - wz);
    float w001 = (1 - wx) * (1 - wy) * wz;
    float w010 = (1 - wx) * wy * (1 - wz);
    float w011 = (1 - wx) * wy * wz;
    float w100 = wx * (1 - wy) * (1 - wz);
    float w101 = wx * (1 - wy) * wz;
    float w110 = wx * wy * (1 - wz);
    float w111 = wx * wy * wz;

    // For each feature channel
    for (int c = 0; c < num_features; c++) {
        float feature_val = features[vertex_idx * num_features + c];

        // Add weighted features to the 8 surrounding voxels
        // Swap x and z to match SDF implementation
        // Original: volume has shape [C, D, H, W] with indices [c, z, y, x]
        // Modified: volume has shape [C, D, H, W] with indices [c, x, y, z]
        atomic_fetch_add_explicit((volume + c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y0 * volume_size + z0), feature_val * w000, memory_order_relaxed);
        atomic_fetch_add_explicit((volume + c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y0 * volume_size + z1), feature_val * w001, memory_order_relaxed);
        atomic_fetch_add_explicit((volume + c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y1 * volume_size + z0), feature_val * w010, memory_order_relaxed);
        atomic_fetch_add_explicit((volume + c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y1 * volume_size + z1), feature_val * w011, memory_order_relaxed);
        atomic_fetch_add_explicit((volume + c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y0 * volume_size + z0), feature_val * w100, memory_order_relaxed);
        atomic_fetch_add_explicit((volume + c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y0 * volume_size + z1), feature_val * w101, memory_order_relaxed);
        atomic_fetch_add_explicit((volume + c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y1 * volume_size + z0), feature_val * w110, memory_order_relaxed);
        atomic_fetch_add_explicit((volume + c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y1 * volume_size + z1), feature_val * w111, memory_order_relaxed);
    }
}

kernel void grid_feature_projection_backward(
    constant float* grad_output [[buffer(0)]],
    constant float* vertices [[buffer(1)]],
    constant float* features [[buffer(2)]],
    device float* grad_vertices [[buffer(3)]],
    device float* grad_features [[buffer(4)]],
    constant int& num_vertices [[buffer(5)]],
    constant int& num_features [[buffer(6)]],
    constant int& volume_size [[buffer(7)]],
    uint vertex_idx [[thread_position_in_grid]])
{
    if (vertex_idx >= (uint)num_vertices) return;

    // Get vertex coordinates (scaled to [-1, 1])
    float x = vertices[vertex_idx * 3];
    float y = vertices[vertex_idx * 3 + 1];
    float z = vertices[vertex_idx * 3 + 2];

    // Clamp to [-1, 1]
    x = clamp(x, -1.0f, 1.0f);
    y = clamp(y, -1.0f, 1.0f);
    z = clamp(z, -1.0f, 1.0f);

    // Convert to volume indices [0, volume_size-1]
    float idx_x = (x + 1.0f) * (volume_size - 1) / 2.0f;
    float idx_y = (y + 1.0f) * (volume_size - 1) / 2.0f;
    float idx_z = (z + 1.0f) * (volume_size - 1) / 2.0f;

    // Get floor and ceil indices
    int x0 = int(floor(idx_x));
    int y0 = int(floor(idx_y));
    int z0 = int(floor(idx_z));

    // Clamp to valid range
    x0 = clamp(x0, 0, volume_size - 2);
    y0 = clamp(y0, 0, volume_size - 2);
    z0 = clamp(z0, 0, volume_size - 2);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    // Compute interpolation weights
    float wx = idx_x - float(x0);
    float wy = idx_y - float(y0);
    float wz = idx_z - float(z0);

    // Compute the 8 weights for trilinear interpolation
    float w000 = (1 - wx) * (1 - wy) * (1 - wz);
    float w001 = (1 - wx) * (1 - wy) * wz;
    float w010 = (1 - wx) * wy * (1 - wz);
    float w011 = (1 - wx) * wy * wz;
    float w100 = wx * (1 - wy) * (1 - wz);
    float w101 = wx * (1 - wy) * wz;
    float w110 = wx * wy * (1 - wz);
    float w111 = wx * wy * wz;

    // Compute gradients for features
    for (int c = 0; c < num_features; c++) {
        float grad_feature = 0.0f;

        // Accumulate gradients from the 8 surrounding voxels
        // Swap x and z to match SDF implementation
        grad_feature += grad_output[c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y0 * volume_size + z0] * w000;
        grad_feature += grad_output[c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y0 * volume_size + z1] * w001;
        grad_feature += grad_output[c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y1 * volume_size + z0] * w010;
        grad_feature += grad_output[c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y1 * volume_size + z1] * w011;
        grad_feature += grad_output[c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y0 * volume_size + z0] * w100;
        grad_feature += grad_output[c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y0 * volume_size + z1] * w101;
        grad_feature += grad_output[c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y1 * volume_size + z0] * w110;
        grad_feature += grad_output[c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y1 * volume_size + z1] * w111;

        grad_features[vertex_idx * num_features + c] = grad_feature;
    }

    // Initialize vertex gradients to zero
    float grad_x = 0.0f;
    float grad_y = 0.0f;
    float grad_z = 0.0f;

    // Only compute vertex gradients if within valid range
    if (x > -1.0f && x < 1.0f && y > -1.0f && y < 1.0f && z > -1.0f && z < 1.0f) {
        float scale_factor = (volume_size - 1) / 2.0f;

        for (int c = 0; c < num_features; c++) {
            float feature_val = features[vertex_idx * num_features + c];

            // Get gradients from output
            // Swap x and z to match SDF implementation
            float g000 = grad_output[c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y0 * volume_size + z0];
            float g001 = grad_output[c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y0 * volume_size + z1];
            float g010 = grad_output[c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y1 * volume_size + z0];
            float g011 = grad_output[c * volume_size * volume_size * volume_size + x0 * volume_size * volume_size + y1 * volume_size + z1];
            float g100 = grad_output[c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y0 * volume_size + z0];
            float g101 = grad_output[c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y0 * volume_size + z1];
            float g110 = grad_output[c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y1 * volume_size + z0];
            float g111 = grad_output[c * volume_size * volume_size * volume_size + x1 * volume_size * volume_size + y1 * volume_size + z1];

            // Compute gradients for each weight with respect to x, y, z
            // dw/dx
            float dw000_dx = -(1 - wy) * (1 - wz) * scale_factor;
            float dw001_dx = -(1 - wy) * wz * scale_factor;
            float dw010_dx = -(wy) * (1 - wz) * scale_factor;
            float dw011_dx = -(wy) * wz * scale_factor;
            float dw100_dx = (1 - wy) * (1 - wz) * scale_factor;
            float dw101_dx = (1 - wy) * wz * scale_factor;
            float dw110_dx = wy * (1 - wz) * scale_factor;
            float dw111_dx = wy * wz * scale_factor;

            // dw/dy
            float dw000_dy = -(1 - wx) * (1 - wz) * scale_factor;
            float dw001_dy = -(1 - wx) * wz * scale_factor;
            float dw010_dy = (1 - wx) * (1 - wz) * scale_factor;
            float dw011_dy = (1 - wx) * wz * scale_factor;
            float dw100_dy = -wx * (1 - wz) * scale_factor;
            float dw101_dy = -wx * wz * scale_factor;
            float dw110_dy = wx * (1 - wz) * scale_factor;
            float dw111_dy = wx * wz * scale_factor;

            // dw/dz
            float dw000_dz = -(1 - wx) * (1 - wy) * scale_factor;
            float dw001_dz = (1 - wx) * (1 - wy) * scale_factor;
            float dw010_dz = -(1 - wx) * wy * scale_factor;
            float dw011_dz = (1 - wx) * wy * scale_factor;
            float dw100_dz = -wx * (1 - wy) * scale_factor;
            float dw101_dz = wx * (1 - wy) * scale_factor;
            float dw110_dz = -wx * wy * scale_factor;
            float dw111_dz = wx * wy * scale_factor;

            // Accumulate gradients for x
            grad_x += feature_val * (
                dw000_dx * g000 + dw001_dx * g001 + dw010_dx * g010 + dw011_dx * g011 +
                dw100_dx * g100 + dw101_dx * g101 + dw110_dx * g110 + dw111_dx * g111
            );

            // Accumulate gradients for y
            grad_y += feature_val * (
                dw000_dy * g000 + dw001_dy * g001 + dw010_dy * g010 + dw011_dy * g011 +
                dw100_dy * g100 + dw101_dy * g101 + dw110_dy * g110 + dw111_dy * g111
            );

            // Accumulate gradients for z
            grad_z += feature_val * (
                dw000_dz * g000 + dw001_dz * g001 + dw010_dz * g010 + dw011_dz * g011 +
                dw100_dz * g100 + dw101_dz * g101 + dw110_dz * g110 + dw111_dz * g111
            );
        }
    }

    // Store the gradients
    grad_vertices[vertex_idx * 3] = grad_x;
    grad_vertices[vertex_idx * 3 + 1] = grad_y;
    grad_vertices[vertex_idx * 3 + 2] = grad_z;
}
)";

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Forward pass implementation
torch::Tensor grid_feature_projection_forward(
    const torch::Tensor& vertices,
    const torch::Tensor& features,
    int volume_size) {
  // Check that tensors are on MPS device
  TORCH_CHECK(vertices.device().is_mps(), "vertices must be a MPS tensor");
  TORCH_CHECK(features.device().is_mps(), "features must be a MPS tensor");
  TORCH_CHECK(vertices.is_contiguous(), "vertices must be contiguous");
  TORCH_CHECK(features.is_contiguous(), "features must be contiguous");

  // Get tensor info
  const int batch_size = vertices.size(0);
  const int num_vertices_per_batch = vertices.size(1);
  const int num_features = features.size(2);

  // Create output volume directly on MPS with batch dimension
  auto volume =
      torch::zeros(
          {batch_size, num_features, volume_size, volume_size, volume_size},
          torch::TensorOptions()
              .dtype(vertices.dtype())
              .device(vertices.device()))
          .contiguous();

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;

    // Load the custom kernel
    MTLCompileOptions* compileOptions = [MTLCompileOptions new];
    compileOptions.languageVersion = MTLLanguageVersion3_2;
    id<MTLLibrary> customKernelLibrary = [device
        newLibraryWithSource:
            [NSString stringWithUTF8String:GRID_FEATURE_PROJECTION_KERNEL]
                     options:compileOptions
                       error:&error];
    TORCH_CHECK(
        customKernelLibrary,
        "Failed to create custom kernel library, error: ",
        error.localizedDescription.UTF8String);

    id<MTLFunction> forwardFunction = [customKernelLibrary
        newFunctionWithName:@"grid_feature_projection_forward"];
    TORCH_CHECK(forwardFunction, "Failed to create forward function");

    // Create compute pipeline state
    id<MTLComputePipelineState> forwardPipelineState =
        [device newComputePipelineStateWithFunction:forwardFunction
                                              error:&error];
    TORCH_CHECK(
        forwardPipelineState,
        "Failed to create forward pipeline state: ",
        error.localizedDescription.UTF8String);

    // Process each batch separately
    for (int b = 0; b < batch_size; b++) {
      // Get the vertices and features for this batch
      auto vertices_batch = vertices[b];
      auto features_batch = features[b];
      auto volume_batch = volume[b];

      // Get a reference to the command buffer for the MPS stream
      id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
      TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

      // Get a reference to the dispatch queue for the MPS stream
      dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

      dispatch_sync(serialQueue, ^() {
        // Start a compute pass
        id<MTLComputeCommandEncoder> computeEncoder =
            [commandBuffer computeCommandEncoder];
        TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

        // Set pipeline state and buffers
        [computeEncoder setComputePipelineState:forwardPipelineState];
        [computeEncoder setBuffer:getMTLBufferStorage(vertices_batch)
                           offset:vertices_batch.storage_offset() *
                           vertices_batch.element_size()
                          atIndex:0];
        [computeEncoder setBuffer:getMTLBufferStorage(features_batch)
                           offset:features_batch.storage_offset() *
                           features_batch.element_size()
                          atIndex:1];
        [computeEncoder setBuffer:getMTLBufferStorage(volume_batch)
                           offset:volume_batch.storage_offset() *
                           volume_batch.element_size()
                          atIndex:2];

        // Set scalar parameters
        [computeEncoder setBytes:&num_vertices_per_batch
                          length:sizeof(int)
                         atIndex:3];
        [computeEncoder setBytes:&num_features length:sizeof(int) atIndex:4];
        [computeEncoder setBytes:&volume_size length:sizeof(int) atIndex:5];

        // Dispatch threads
        MTLSize gridSize = MTLSizeMake(num_vertices_per_batch, 1, 1);
        NSUInteger threadGroupSize =
            forwardPipelineState.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > num_vertices_per_batch) {
          threadGroupSize = num_vertices_per_batch;
        }
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [computeEncoder dispatchThreads:gridSize
                  threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];

        // Commit the work
        torch::mps::commit();
      });
    }
  }

  return volume;
}

// Backward pass implementation
std::vector<torch::Tensor> grid_feature_projection_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& vertices,
    const torch::Tensor& features,
    int volume_size) {
  // Check that tensors are on MPS device
  TORCH_CHECK(
      grad_output.device().is_mps(), "grad_output must be a MPS tensor");
  TORCH_CHECK(vertices.device().is_mps(), "vertices must be a MPS tensor");
  TORCH_CHECK(features.device().is_mps(), "features must be a MPS tensor");

  // Ensure vertices and features are contiguous
  TORCH_CHECK(vertices.is_contiguous(), "vertices must be contiguous");
  TORCH_CHECK(features.is_contiguous(), "features must be contiguous");

  // Get tensor info
  const int batch_size = vertices.size(0);
  const int num_vertices_per_batch = vertices.size(1);
  const int num_features = features.size(2);

  // Create output gradients directly on MPS with batch dimension
  auto grad_vertices = torch::zeros_like(vertices).contiguous();
  auto grad_features = torch::zeros_like(features).contiguous();

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;

    // Load the custom kernel
    MTLCompileOptions* compileOptions = [MTLCompileOptions new];
    compileOptions.languageVersion = MTLLanguageVersion3_2;
    id<MTLLibrary> customKernelLibrary = [device
        newLibraryWithSource:
            [NSString stringWithUTF8String:GRID_FEATURE_PROJECTION_KERNEL]
                     options:compileOptions
                       error:&error];
    TORCH_CHECK(
        customKernelLibrary,
        "Failed to create custom kernel library, error: ",
        error.localizedDescription.UTF8String);

    id<MTLFunction> backwardFunction = [customKernelLibrary
        newFunctionWithName:@"grid_feature_projection_backward"];
    TORCH_CHECK(backwardFunction, "Failed to create backward function");

    // Create compute pipeline state
    id<MTLComputePipelineState> backwardPipelineState =
        [device newComputePipelineStateWithFunction:backwardFunction
                                              error:&error];
    TORCH_CHECK(
        backwardPipelineState,
        "Failed to create backward pipeline state: ",
        error.localizedDescription.UTF8String);

    // Process each batch separately
    for (int b = 0; b < batch_size; b++) {
      // Get the tensors for this batch
      auto grad_output_batch = grad_output[b];
      auto vertices_batch = vertices[b];
      auto features_batch = features[b];
      auto grad_vertices_batch = grad_vertices[b];
      auto grad_features_batch = grad_features[b];

      // Get a reference to the command buffer for the MPS stream
      id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
      TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

      // Get a reference to the dispatch queue for the MPS stream
      dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

      dispatch_sync(serialQueue, ^() {
        // Start a compute pass
        id<MTLComputeCommandEncoder> computeEncoder =
            [commandBuffer computeCommandEncoder];
        TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

        // Set pipeline state and buffers
        [computeEncoder setComputePipelineState:backwardPipelineState];
        [computeEncoder setBuffer:getMTLBufferStorage(grad_output_batch)
                           offset:grad_output_batch.storage_offset() *
                           grad_output_batch.element_size()
                          atIndex:0];
        [computeEncoder setBuffer:getMTLBufferStorage(vertices_batch)
                           offset:vertices_batch.storage_offset() *
                           vertices_batch.element_size()
                          atIndex:1];
        [computeEncoder setBuffer:getMTLBufferStorage(features_batch)
                           offset:features_batch.storage_offset() *
                           features_batch.element_size()
                          atIndex:2];
        [computeEncoder setBuffer:getMTLBufferStorage(grad_vertices_batch)
                           offset:grad_vertices_batch.storage_offset() *
                           grad_vertices_batch.element_size()
                          atIndex:3];
        [computeEncoder setBuffer:getMTLBufferStorage(grad_features_batch)
                           offset:grad_features_batch.storage_offset() *
                           grad_features_batch.element_size()
                          atIndex:4];

        // Set scalar parameters
        [computeEncoder setBytes:&num_vertices_per_batch
                          length:sizeof(int)
                         atIndex:5];
        [computeEncoder setBytes:&num_features length:sizeof(int) atIndex:6];
        [computeEncoder setBytes:&volume_size length:sizeof(int) atIndex:7];

        // Dispatch threads
        MTLSize gridSize = MTLSizeMake(num_vertices_per_batch, 1, 1);
        NSUInteger threadGroupSize =
            backwardPipelineState.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > num_vertices_per_batch) {
          threadGroupSize = num_vertices_per_batch;
        }
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [computeEncoder dispatchThreads:gridSize
                  threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];

        // Commit the work
        torch::mps::commit();
      });
    }
  }

  return {grad_vertices, grad_features};
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &grid_feature_projection_forward,
      "Grid Feature Projection forward (Metal)");
  m.def(
      "backward",
      &grid_feature_projection_backward,
      "Grid Feature Projection backward (Metal)");
}