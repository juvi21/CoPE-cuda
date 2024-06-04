#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void cumsum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t size,
    int64_t dim_size,
    int64_t stride) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64_t d = (idx / stride) % dim_size;
        float cumsum = 0.0;
        for (int i = 0; i <= d; ++i) {
            int current_idx = idx - d * stride + i * stride;
            cumsum += input[current_idx];
        }
        output[idx] = cumsum;
    }
}

void cumsum_launcher(
    const torch::Tensor& input,
    torch::Tensor& output,
    int64_t dim) {

    int64_t numel = input.numel();
    int64_t dim_size = input.size(dim);
    int64_t stride = input.stride(dim);

    int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    cumsum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        dim_size,
        stride);
}
