
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "mathutil_cuda_kernel.h"

int cuda_gridsize(int n)
{
    int k = (n / BLOCK_SIZE) + 1;
    if(k > 65535) {
        k = 65535;
    };
    return k;
}



__global__ void Copy(const float *a, float *b, const int z, const int h, const int w, const int pad=1){

    const int w_b = w + (pad*2);
    const int h_b = h + (pad*2);
    const int z_b = z + (pad*2);

    int d = 0, r = 0, c = 0;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int index_b = 0;
    const int wh = w * h;

    while(index < ( z * h * w ) ){
        d  = index / (wh);
        r = (index - (d * wh)) / w;
        c = index - w * (r + h * d);
        index_b = (c + pad) + w_b * ((r + pad) + h_b * (d + pad));
        b[index_b] = a[index];
        index += blockDim.x * gridDim.x;
     }
}


__global__ void NMSFilter3d_kernel(const float *a, float *b, const int z, const int h, const int w, const int ker, const int pad){
    
    bool stop = false;
    const int w_a = w + (pad*2);
    const int h_a = h + (pad*2);
    const int z_a = z + (pad*2);

    int d = 0, r = 0, c = 0, d_a = 0, r_a = 0, c_a = 0;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int index_a = 0;
    const int wh = w * h;

    int k_d = 0, k_r = 0, k_c = 0;
    const int ker_ = ker / 2;

    while(index < ( z * h * w ) ){
        d  = index / (wh);
        r = (index - (d * wh)) / w;
        c = index - w * (r + h * d);
        d_a = d + 1;
        r_a = r + 1;
        c_a = c + 1;
        index_a = c_a + w_a * (r_a + h_a * d_a);
//        b[index] = index;
//        index += blockDim.x * gridDim.x;

        stop = false;
        for(k_d=-ker_; k_d<=ker_&&!stop; k_d++){
            for(k_r=-ker_; k_r<=ker_&&!stop; k_r++){
                for(k_c=-ker_; k_c<=ker_&&!stop; k_c++){
                    if (index_a == (c_a + k_c) + w_a * ((r_a + k_r) + h_a * (d_a + k_d)) )
                        continue;
                    if (a[index_a] <= a[ (c_a + k_c) + w_a * ((r_a + k_r) + h_a * (d_a + k_d)) ])
                        stop = true;
                }
            }
        }
                if (!stop)
                       b[index] = a[index_a];
        index += blockDim.x * gridDim.x;
     }
}


void NMSFilter3d_cuda(const float *a, float *c, const int z, const int h, const int w, const int ker, const int pad, cudaStream_t stream)
{
    //int b_size = (h + pad*2) * (w +  pad*2) * (z + pad*2);
    int a_size = h * w * z;
    cudaError_t err;

    //size_t ss = (size_t)(b_size*sizeof(float));
    

   //cudaMemset(b, 0, ss);

   //Copy<<<cuda_gridsize(a_size), BLOCK_SIZE, 0, stream>>>(a, b, z, h, w, pad);

   //cudaMemset(c, 0, a_size*sizeof(float));


   NMSFilter3d_kernel<<<cuda_gridsize(a_size), BLOCK_SIZE, 0, stream>>>(a, c, z-2, h-2, w-2, ker, pad);
      err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "4CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    //err = cudaGetLastError();
    //if (cudaSuccess != err)
    //{
    //    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    //    exit(-1);
    //}

}

