#include "mathutil_cuda_kernel.h"
#include <torch/extension.h>


torch::Tensor NMSFilter3d(torch::Tensor a_tensor, int kernel_size, int padding)
{   
    
    int h=0, w=0, z=0;
    //int ndim = THCudaTensor_nDimension(state, a_tensor);
    int ndim = 3;

    if (ndim == 3) {    
         z = a_tensor.size(0);
         h = a_tensor.size(1);
         w = a_tensor.size(2);
    } else {
        fprintf(stderr, "3 dim required, got %d\n", ndim);
        exit(-1);
    }
    //torch::Tensor b_tensor = torch::zeros({z + (padding*2), h + (padding*2), w + (padding*2)}, at::CUDA(at::kFloat));
    torch::Tensor c_tensor = torch::zeros({z-2, h-2, w-2}, at::CUDA(at::kFloat));
   
    float *a = a_tensor.data<float>();
    //float *b = b_tensor.data<float>();
    float *c = c_tensor.data<float>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    NMSFilter3d_cuda(a, c, z, h, w, kernel_size, padding, stream);
    //THCudaFree(state, b_tensor); 
    //return a_tensor;
    return c_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("NMSFilter3d", &NMSFilter3d, "NMSFilter3d");
}
