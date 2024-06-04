#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void cumsum_launcher(
    const torch::Tensor& input,
    torch::Tensor& output,
    int64_t dim);

torch::Tensor cumsum(const torch::Tensor& input, int64_t dim) {
    CHECK_INPUT(input);
    auto output = torch::empty_like(input);

    dim = dim < 0 ? input.dim() + dim : dim;

    cumsum_launcher(input, output, dim);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum", &cumsum, "Cumulative Sum (CUDA)");
}
