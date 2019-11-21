#ifndef _MATHUTIL_CUDA_KERNEL
#define _MATHUTIL_CUDA_KERNEL

#define BLOCK_SIZE 256

#define MAX_STREAMS 256

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#ifdef __cplusplus
extern "C" {
#endif

void NMSFilter3d_cuda(const float *a, float *c, const int z, const int h, const int w, const int ker, const int pad, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
