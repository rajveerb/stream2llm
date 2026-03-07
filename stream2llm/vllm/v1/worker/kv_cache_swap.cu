#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Helper to compute the offset between for the value tensor in a KV cache tensor
static inline int64_t get_val_stride(const torch::Tensor& kv_tensor) {
    return kv_tensor.stride(0) * kv_tensor.element_size();
}

// Helper to compute the offset between consecutive blocks for a KV cache tensor slice (key or value)
static inline int64_t get_block_stride(const torch::Tensor& kv_tensor) {
    // kv_tensor shape: (2, num_blocks, block_size, num_kv_heads, head_size)
    // stride(1) gives number of ELEMENTS between consecutive blocks for a single slice (key or value)
    return kv_tensor.stride(1) * kv_tensor.element_size();
}

void swap_blocks_host(const torch::Tensor& src_kv_cache,
                      const torch::Tensor& dst_kv_cache,
                      const int* src_block_ids,
                      const int* dst_block_ids,
                      int64_t num_blocks,
                      cudaMemcpyKind memcpy_type,
                      cudaStream_t stream) {
    const char* src_base_ptr = static_cast<const char*>(src_kv_cache.data_ptr());
    char* dst_base_ptr = static_cast<char*>(dst_kv_cache.data_ptr());

    const int64_t val_stride = get_val_stride(src_kv_cache);
    const int64_t block_stride = get_block_stride(src_kv_cache);

    for (int64_t i = 0; i < num_blocks; ++i) {
        int64_t src_block_id = static_cast<int64_t>(src_block_ids[i]);
        int64_t dst_block_id = static_cast<int64_t>(dst_block_ids[i]);

        // TODO: rbachkaniwala3: the memcpy of k and v cache can be parallelized

        // Copy KEY slice (slice 0)
        const char* key_src_ptr = src_base_ptr + src_block_id * block_stride;
        char* key_dst_ptr = dst_base_ptr + dst_block_id * block_stride;
        cudaMemcpyAsync(key_dst_ptr, key_src_ptr, block_stride, memcpy_type, stream);

        // Copy VALUE slice (slice 1)
        const char* val_src_base = src_base_ptr + val_stride;
        char* val_dst_base = dst_base_ptr + val_stride;
        const char* val_src_ptr = val_src_base + src_block_id * block_stride;
        char* val_dst_ptr = val_dst_base + dst_block_id * block_stride;
        cudaMemcpyAsync(val_dst_ptr, val_src_ptr, block_stride, memcpy_type, stream);
    }
}

void swap_blocks(std::vector<torch::Tensor> src_kv_caches,
                 std::vector<torch::Tensor> dst_kv_caches,
                 torch::Tensor src_block_ids,
                 torch::Tensor dst_block_ids,
                 int num_layers,
                 int num_blocks,
                 int /*block_size not used in host impl*/) {
    TORCH_CHECK(src_block_ids.dtype() == torch::kInt32);
    TORCH_CHECK(dst_block_ids.dtype() == torch::kInt32);
    TORCH_CHECK(src_block_ids.device().is_cuda(), "block id tensors must be on CUDA");
    TORCH_CHECK(dst_block_ids.device().is_cuda(), "block id tensors must be on CUDA");

    TORCH_CHECK(static_cast<int>(src_kv_caches.size()) == num_layers);
    TORCH_CHECK(static_cast<int>(dst_kv_caches.size()) == num_layers);

    // For simplicity, use current stream.
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Move block id tensors to CPU (synchronous) for safe host access
    torch::Tensor src_block_ids_cpu_tensor = src_block_ids.to(torch::kCPU);
    torch::Tensor dst_block_ids_cpu_tensor = dst_block_ids.to(torch::kCPU);
    const int* src_block_ids_cpu = src_block_ids_cpu_tensor.data_ptr<int>();
    const int* dst_block_ids_cpu = dst_block_ids_cpu_tensor.data_ptr<int>();

    for (int layer = 0; layer < num_layers; ++layer) {
        const torch::Tensor& src_kv_cache = src_kv_caches[layer];
        const torch::Tensor& dst_kv_cache = dst_kv_caches[layer];

        torch::Device src_dev = src_kv_cache.device();
        torch::Device dst_dev = dst_kv_cache.device();

        cudaMemcpyKind memcpy_type;
        if (src_dev.is_cuda() && dst_dev.is_cpu()) {
            memcpy_type = cudaMemcpyDeviceToHost;
        } else if (src_dev.is_cpu() && dst_dev.is_cuda()) {
            memcpy_type = cudaMemcpyHostToDevice;
        } else {
            TORCH_CHECK(false, "Unsupported device combination for swap_blocks (both on CPU or both on GPU is not supported)");
        }

        // Launch host-side async memcpy per block
        swap_blocks_host(src_kv_cache, dst_kv_cache,
                         src_block_ids_cpu,
                         dst_block_ids_cpu,
                         num_blocks, memcpy_type, stream);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swap_blocks", &swap_blocks, "Swap KV cache blocks between tensors (CPU/GPU)");
}