#pragma O3
#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH 32
#define TILE_WIDTH_2 32
#define NUM_STREAM_1 4
#define NUM_STREAM_2 50


__constant__ float kernel[4704];
__constant__ half kernel_0[4704];
half kernel_host[4704];
cudaStream_t streams_1[NUM_STREAM_1];
cudaStream_t streams_2[NUM_STREAM_2];

float *host_y_b, *host_x_b;

__global__ void to_half(const float* __restrict__ in, half* __restrict__ out, const int len){
    int a = blockDim.x;
    int c = threadIdx.x;
    int b = blockIdx.x;
    int idx = a * b + c;
    if(idx < len) out[idx] = __float2half(in[idx]);
}


__global__ void conv_forward_kernel_mod(float* __restrict__ y, const float* __restrict__ x, const half* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{
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
#define x_half4d(i3, i2, i1, i0) k[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define kernel_half4d(i3, i2, i1, i0) kernel_0[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]

    // Insert your GPU convolution kernel code here

    int m = blockIdx.z;
    int b = blockIdx.y * blockDim.x + threadIdx.y;
    int coor = blockIdx.x* blockDim.x + threadIdx.x;
    int w = coor % W_out;
    int h = coor / W_out;
    half value = 0.0;
    int kernel_len = K * K;
    if(h < H_out && w < W_out && b < B){
#pragma unroll 2
        for(int c = 0; c < C; ++c)
#pragma unroll 7
                for(int p=0;p < K; ++p)
#pragma unroll 7
                        for(int q = 0; q < K; ++q)
                            value = __hadd(__hmul(x_half4d(b, c, h+p, w+q), kernel_0[m * (C * kernel_len) + c * kernel_len + p * (K) +    q]), value);
        y4d(b, m, h, w) =  __half2float(value);
    }
#undef y4d
#undef x4d
#undef k4d
}




__global__ void conv_forward_kernel_0(float* __restrict__ y, const float* __restrict__ x, half *x_half, const int B, const int M, const int C, const int H, const int W, const int K)
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
#define x4d(i3, i2, i1, i0) x_half[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel_0[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]

    // Insert your GPU convolution kernel code here

    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
    int w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;
    half value = 0.0;
    if(h < H_out && w < W_out){
        #pragma unroll 4
        for(int c = 0; c < C; ++c)
            #pragma unroll 7
            for(int p=0;p < K; ++p)
                #pragma unroll 7
                for(int q = 0; q < K; ++q)
                    value = __hadd(__hmul(x4d(b, c, h+p, w+q) , k4d(m, c, p, q)), value);
        y4d(b, m, h, w) = __half2float(value);
    }
#undef y4d
#undef x4d
#undef k4d
}


__global__ void conv_forward_kernel_1(float* __restrict__ y, const float* __restrict__ x, half *x_half, const int B, const int M, const int C, const int H, const int W, const int K)
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
#define x4d(i3, i2, i1, i0) x_half[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel_0[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]

    // Insert your GPU convolution kernel code here

    int W_grid = ceil((float)W_out/blockDim.x);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
    int w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;
    half value = 0.0;
    if(h < H_out && w < W_out){
#pragma unroll 2
        for(int c = 0; c < C; ++c)
#pragma unroll 7
                for(int p=0;p < K; ++p)
#pragma unroll 7
                        for(int q = 0; q < K; ++q)
                            value = __hadd(__hmul(x4d(b, c, h+p, w+q) , k4d(m, c, p, q)), value);
        y4d(b, m, h, w) = __half2float(value);
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
//     cudaError_t error = cudaGetLastError();
//     if(error != cudaSuccess)
//     {
//         std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//         exit(-1);
//     }
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int input_length = B * H * W * C;
    int output_length = B * H_out * W_out * M;
    int kernel_length = C * K * K * M;
    //std::cout << H << " " << W << " " << H_out << " " << W_out << " " <<  C << " " << M  << " " << K << "\n";
    //get_device_properties();

    cudaMalloc(device_x_ptr, input_length * sizeof(float));
    cudaMalloc(device_y_ptr, output_length * sizeof(float));
    cudaMalloc(device_k_ptr, input_length * sizeof(half));
    host_x_b = (float*)host_x;
    cudaHostRegister((float*)host_x, input_length * sizeof(float), cudaHostRegisterDefault);
    host_y_b = (float*)host_y;
    cudaHostRegister((float*)host_y, output_length * sizeof(float), cudaHostRegisterDefault);
    for(int i = 0; i < kernel_length; ++i) kernel_host[i] = __float2half(host_k[i]);
    cudaMemcpyToSymbol(kernel_0, kernel_host, kernel_length * sizeof(half));
    //cudaMemcpyToSymbol(kernel, host_k, kernel_length * sizeof(float));

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    //std::cout << "what 2!!!" << std::endl;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    if(H > 80) {
        int W_grid = ceil((float)W_out/TILE_WIDTH);
        int H_grid = ceil((float)H_out/TILE_WIDTH);
        int offset = B / NUM_STREAM_1 * H_out * W_out * M ;
        int xoffset = B / NUM_STREAM_1 * H * W * C;
        dim3 DimGrid(B / NUM_STREAM_1, M, W_grid * H_grid);
        dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 to_fp16_block(1024, 1, 1);
        dim3 to_fp16_grid(ceil(B * H * W * C / NUM_STREAM_1 / 1024.0), 1, 1);
        half *device_x_half = reinterpret_cast<half*>(const_cast<float*>(device_k));
        for(int i = 0; i < NUM_STREAM_1; ++i){
            cudaStreamCreate(&streams_1[i]);
            cudaMemcpyAsync(const_cast<float*>(&device_x[i*xoffset]), &host_x_b[i * xoffset], xoffset * sizeof(float), cudaMemcpyHostToDevice, streams_1[i]);
            to_half<<<to_fp16_grid, to_fp16_block, 0, streams_1[i]>>>(device_x + i * xoffset,device_x_half+ i * xoffset, xoffset);
            conv_forward_kernel_0<<<DimGrid, DimBlock, 0, streams_1[i]>>>(device_y + i * offset, device_x + i * xoffset, device_x_half  + i * xoffset, B / NUM_STREAM_1, M, C, H, W, K);
            cudaMemcpyAsync(&host_y_b[i*offset], device_y + i * offset, offset * sizeof(float), cudaMemcpyDeviceToHost, streams_1[i]);
        }


    }
    else {
//        int W_grid = ceil((float)W_out/TILE_WIDTH_2);
//        int H_grid = ceil((float)H_out/TILE_WIDTH_2);
//        int offset = B / NUM_STREAM_2 * H_out * W_out * M ;
//        int xoffset = B / NUM_STREAM_2 * H * W * C;
//        dim3 DimGrid(B / NUM_STREAM_2, M, W_grid * H_grid);
//        dim3 DimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);
//        dim3 to_fp16_block(1024, 1, 1);
//        dim3 to_fp16_grid(ceil(B * H * W * C / NUM_STREAM_2 / 1024.0), 1, 1);
//        half *device_x_half =  reinterpret_cast<half*>(const_cast<float*>(device_k));
//        for(int i = 0; i < NUM_STREAM_2; ++i){
//            cudaStreamCreate(&streams_2[i]);
//            cudaMemcpyAsync(const_cast<float*>(&device_x[i*xoffset]), &host_x_b[i * xoffset], xoffset * sizeof(float), cudaMemcpyHostToDevice, streams_2[i]);
//            to_half<<<to_fp16_grid, to_fp16_block, 0, streams_2[i]>>>(device_x + i * xoffset,device_x_half+ i * xoffset, xoffset);
//            conv_forward_kernel_1<<<DimGrid, DimBlock, 0, streams_2[i]>>>(device_y + i * offset, device_x + i * xoffset, device_x_half + i * xoffset, B / NUM_STREAM_2, M, C, H, W, K);
//            cudaMemcpyAsync(&host_y_b[i*offset], device_y + i * offset, offset * sizeof(float), cudaMemcpyDeviceToHost, streams_2[i]);
//        }
////
////

        int tmpa = ceil(W_out * H_out * 1.0 / TILE_WIDTH_2);
        int tmpb = ceil( 1.0 * B / NUM_STREAM_2  / TILE_WIDTH_2 );
        int tmpc = M;
        int xoffset = B * H * W * C / NUM_STREAM_2;
        int offset = B / NUM_STREAM_2 * H_out * W_out * M ;
        dim3 DimGrid(tmpa, tmpb, tmpc);
        dim3 DimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);
        dim3 to_fp16_block(1024, 1, 1);
        dim3 to_fp16_grid(ceil(B * H * W * C / NUM_STREAM_2 / 1024.0), 1, 1);
        half *device_x_half =  reinterpret_cast<half*>(const_cast<float*>(device_k));
#pragma unroll NUM_STREAM_2
        for(int i = 0; i < NUM_STREAM_2; ++i){
            cudaStreamCreate(&streams_2[i]);
            cudaMemcpyAsync(const_cast<float*>(&device_x[i*xoffset]), &host_x_b[i * xoffset], xoffset * sizeof(float), cudaMemcpyHostToDevice, streams_2[i]);
            to_half<<<to_fp16_grid, to_fp16_block, 0, streams_2[i]>>>(device_x + i * xoffset,device_x_half+ i * xoffset, xoffset);
            conv_forward_kernel_mod<<<DimGrid, DimBlock, 0, streams_2[i]>>>(device_y + i * offset,
                                                                          device_x + i * xoffset,
                                                                          &device_x_half[i*xoffset],
                                                                          B / NUM_STREAM_2,
                                                                          M, C, H, W, K );
            cudaMemcpyAsync(&host_y_b[i*offset], device_y + i * offset, offset * sizeof(float), cudaMemcpyDeviceToHost, streams_2[i]);
        }

    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    cudaDeviceSynchronize();
    cudaFree(device_y);
    cudaFree(device_x);
    //cudaFree(matrx_unrolled);
    //cudaFree(device_k);
   // get_device_properties();
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
