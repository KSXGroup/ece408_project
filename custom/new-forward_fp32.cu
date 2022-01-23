#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define CONSTANT_MEMORY 10
//#define MATRIX_MULTIPLY
#define TILE_X 16
#define TILE_Y 32
__constant__ float kernel[4704];




__global__ void conv_matrix_multi_tiled(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K){
#define x_half4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define kernel_half4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]
    extern __shared__ float shmem_3[];
    float *X = &shmem_3[0];
    float *Y = &shmem_3[K * K * blockDim.x * C];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int b = blockIdx.x;
    int m = blockIdx.y * blockDim.x + tidx;
    int w = blockIdx.z * blockDim.y + tidy;
    int kernel_len = K * K;
    int outlen = W_out * H_out;
    int col_width = C * kernel_len;
    int w_out = w % W_out;
    int h_out = w / W_out;
    float value = 0.0;
#pragma unroll 2
    for(int i = 0; i < col_width; i += blockDim.y){
        if(m < M && i + tidy < col_width)
            X[tidx * col_width + tidy + i] = k[m * col_width + tidy + i];
    }
#pragma unroll 2
    for(int i = 0; i < col_width; i += blockDim.x){
        if (w < outlen && i + tidx < col_width) {
            int cur_x = i + tidx;
            //int cur_y = i + tidy;
            int c = cur_x / kernel_len;
            int tmp_idx = cur_x - c * kernel_len;
            int p = tmp_idx / K;
            int q = tmp_idx % K;
            int idx1 = cur_x * blockDim.y + tidy;
            Y[idx1] = x_half4d(b, c, h_out + p, w_out + q);
        }
    }
    __syncthreads();
    if(m < M && w < outlen) {
#pragma unroll 4
        for(int i = 0; i < col_width; ++i)
            value += X[tidx * col_width + i] * Y[i * blockDim.y + tidy];
        y[b * (M * H_out * W_out) + m * (H_out * W_out) + w] = value;
    }
}


__global__ void conv_forward_kernel_tiled(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__  k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]
#define Xshare2d(i1, i0) X[(i1) * (TILE_WIDTH + K - 1) + i0]
#define kshared4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]
#define kshared2d(i1, i0) K_shared[(i1) * (K) +    i0]


    // Insert your GPU convolution kernel code here
    int X_tile_size = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float *X = &shmem[0];
    //float *K_shared = &shmem[X_tile_size * X_tile_size];

    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * blockDim.y;
    int w = (blockIdx.z % W_grid) * blockDim.x;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    float value = 0.0;
    for(int c = 0; c <  C; ++c){
        for (int i = tidx; i < X_tile_size; i += TILE_WIDTH) {
            for (int j = tidy; j < X_tile_size; j += TILE_WIDTH) {
                if(h+i < H && w + j < W)
                    Xshare2d(i, j) = x4d(b, c, h + i, w + j);
                else
                    Xshare2d(i, j) = 0.0;
            }
        }
        __syncthreads();
#pragma unroll 7
        for(int p = 0; p < K; ++p) {
#pragma unroll 7
            for (int q = 0; q < K; ++q) {
                value += Xshare2d(tidx + p, tidy + q) * k4d(m, c, p, q);
            }
        }
        __syncthreads();
    }
    if(h+tidx < H_out && w+tidy < W_out)
        y4d(b, m, h + tidx, w + tidy) = value;
#undef y4d
#undef x4d
#undef k4d
#undef Xshare2d
#undef kshared4d
}


__global__ void conv_forward_kernel_baseline(float *y, const float* x, const float* k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#ifndef CONSTANT_MEMORY
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]
#else
#define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]
#endif

    // Insert your GPU convolution kernel code here

    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
    int w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;
    float value = 0.0;
    if(h < H_out && w < W_out){
        for(int c = 0; c < C; ++c)
                for(int p=0;p < K; ++p)
                        for(int q = 0; q < K; ++q)
                            value += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
        y4d(b, m, h, w) = value;
    }
#undef y4d
#undef x4d
#undef k4d
}




__global__ void conv_forward_kernel(float *y, const float* __restrict__ x, const float* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#ifndef CONSTANT_MEMORY
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]
#else
#define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]
#endif

    // Insert your GPU convolution kernel code here

    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
    int w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;
    float value = 0.0;
    if(h < H_out && w < W_out){
#pragma unroll 2
        for(int c = 0; c < C; ++c)
#pragma unroll 7
                for(int p=0;p < K; ++p)
#pragma unroll 7
                        for(int q = 0; q < K; ++q)
                            value += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
        y4d(b, m, h, w) = value;
    }
#undef y4d
#undef x4d
#undef k4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int input_length = B * H * W * C;
    int output_length = B * H_out * W_out * M;
    int kernel_length = C * K * K * M;
    cudaMalloc(device_x_ptr, input_length * sizeof(float));
    cudaMalloc(device_y_ptr, output_length * sizeof(float));
#ifndef CONSTANT_MEMORY
    cudaMalloc(device_k_ptr, kernel_length * sizeof(float));
    cudaMemcpy(*device_k_ptr, host_k, kernel_length * sizeof(float), cudaMemcpyHostToDevice);
#else
    cudaMemcpyToSymbol(kernel, host_k, kernel_length * sizeof(float));
#endif
    cudaMemcpy(*device_x_ptr, host_x, input_length * sizeof(float), cudaMemcpyHostToDevice);
;
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    int H_out = H - K + 1;
    int W_out = W - K + 1;
#ifndef MATRIX_MULTIPLY
    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int H_grid = ceil((float)H_out/TILE_WIDTH);
    dim3 DimGrid(B , M, W_grid * H_grid);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    //conv_forward_kernel<<<DimGrid, DimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    //conv_forward_kernel<<<DimGrid, DimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    conv_forward_kernel_tiled<<<DimGrid, DimBlock, (C * (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1)) * sizeof(float) >>>(device_y, device_x, device_k, B, M, C, H, W, K);
#else
    int a = ceil((float) M / TILE_X);
    int b = ceil((float) (W_out * H_out) / TILE_Y);
    dim3 DimGrid(B, a, b);
    dim3 DimBlock(TILE_X,TILE_Y, 1);
    conv_matrix_multi_tiled<<<DimGrid, DimBlock, (TILE_X * C * K * K + TILE_Y * C * K * K) * sizeof(float)>>>(device_y, device_x, device_k, B, M, C, H, W, K);
#endif
     cudaError_t error = cudaGetLastError();
     if(error != cudaSuccess)
     {
         std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
         exit(-1);
     }

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int len = B * H_out * W_out * M ;
    cudaMemcpy(host_y, device_y, len * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
#ifndef CONSTANT_MEMORY
    cudaFree(device_k);
#endif
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}