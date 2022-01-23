#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define TILE_WIDTH_2 32
#define MASK_WIDTH 5
#define KERNEL_SIZE 28
#define NUM_STREAM_1 4
#define NUM_STREAM_2 50
#define SHARED_HEIGHT 196
#define SHARED_WIDTH 16
#define TILE_X 16
#define TILE_Y 32
#define TILE_X_0 4
#define TILE_Y_0 64



__constant__ float kernel[4704];
__constant__ half kernel_0[4704];
half kernel_host[4704];
cudaStream_t streams_1[NUM_STREAM_1], streams_2[NUM_STREAM_2];


__global__ void conv_matrix_multi(float* __restrict__ y, const float* __restrict__ x, const half* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K, const int tw){
#define x_half4d(i3, i2, i1, i0) k[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define kernel_half4d(i3, i2, i1, i0) kernel_0[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int b = blockIdx.x;
    int m = blockIdx.y * blockDim.x + threadIdx.x;
    int w = blockIdx.z * blockDim.y + threadIdx.y;
    //if(blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 2)
        //printf("block idx y %d, block idx z %d, block dim x %d, block dim y %d, threadIdx x %d, threadIdx y %d, m %d, w %d\n",blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y ,m, w);
    int col_width = C * K * K;
    int w_out = w % W_out;
    int h_out = w / W_out;
    half value = 0.0;
    if(m < M && w < W_out * H_out) {
#pragma unroll 8
        for (int i = 0; i < col_width; ++i) {
            int c = i / (K * K);
            int tmp_idx = i - c * (K * K);
            int p = tmp_idx / K;
            int q = tmp_idx % K;
            //value = __hadd(__hmul(x_half4d(b, c, h_out + p, w_out + q), value), value);
            value = __hadd(__hmul(x_half4d(b, c, h_out + p, w_out + q), kernel_0[m * (C * K * K ) + i]), value);
            //value = __hadd(__hmul(value, kernel_0[m * (C * K * K ) + i]), value);
        }
        y[b * (M * H_out * W_out) + m * (H_out * W_out) + w] = __half2float(value);
    }
}

__global__ void conv_matrix_multi_tiled(float* __restrict__ y, const float* __restrict__ x, const half* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K, const int tw, const int kernel_offset){
#define x_half4d(i3, i2, i1, i0) k[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define kernel_half4d(i3, i2, i1, i0) kernel_0[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]
    extern __shared__ half shmem_3[];
    half *X = &shmem_3[0];
    half *Y = &shmem_3[K * K * blockDim.x * C];
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
    const half* kernel_cur = &k[kernel_offset];
    half value = 0.0;
#pragma unroll 2
    for(int i = 0; i < col_width; i += blockDim.y){
        if(m < M && i + tidy < col_width)
            X[tidx * col_width + tidy + i] = kernel_cur[m * col_width + tidy + i];
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
            value = __hadd(__hmul(X[tidx * col_width + i], Y[i * blockDim.y + tidy]), value);
        y[b * (M * H_out * W_out) + m * (H_out * W_out) + w] = __half2float(value);
    }
}

__global__ void to_half(const float* __restrict__ in, half* __restrict__ out, const int len){
    int a = blockDim.x;
    int c = threadIdx.x;
    int b = blockIdx.x;
    int idx = a * b + c;
    if(idx < len) out[idx] = __float2half(in[idx]);
}


__global__ void conv_forward_kernel_large(float* __restrict__ y, const float* __restrict__ x, const half* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K, const int tw)
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
#define x4d(i3, i2, i1, i0) k[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel_0[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]

    // Insert your GPU convolution kernel code here

    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
    int w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;
    half value = 0.0;
    if(h < H_out && w < W_out){
//#pragma unroll 4
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


__global__ void conv_forward_kernel_0(float* __restrict__ y, const float* __restrict__ x, const half* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K, const int tw)
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
#define x_half4d(i3, i2, i1, i0) k[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define kernel_half4d(i3, i2, i1, i0) kernel_0[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +    i0]

    // Insert your GPU convolution kernel code here

    int m = blockIdx.z;
    int b = blockIdx.y*tw + threadIdx.y;
    int coor = blockIdx.x*tw + threadIdx.x;
    int w = coor % W_out;
    int h = coor / W_out;

//    int b =  bc % B;
//    int c = bc / B;
    half value;
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


__global__ void conv_forward_kernel_1(float* __restrict__ y, const float* __restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K)
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

__global__ void conv_forward_kernel_tiled(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
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
    float *K_shared = &shmem[X_tile_size * X_tile_size];

    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * blockDim.y;
    int w = (blockIdx.z % W_grid) * blockDim.x;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    float value = 0.0;
    for(int c = 0; c <  C; ++c){
        if(tidx < K && tidy < K) kshared2d(tidx, tidy) = kshared4d(m, c, tidx, tidy);
        __syncthreads();

        for (int i = tidx; i < X_tile_size; i += TILE_WIDTH) {
            for (int j = tidy; j < X_tile_size; j += TILE_WIDTH) {
                Xshare2d(i, j) = x4d(b, c, h + i, w + j);
            }
        }
        __syncthreads();
        for(int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                value += Xshare2d(tidx + p, tidy + q) * kshared2d(p, q);
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
    //get_device_properties();
    cudaHostRegister((float*)host_x, input_length * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister((float*)host_y, output_length * sizeof(float), cudaHostRegisterDefault);
    cudaMalloc(device_x_ptr, input_length * sizeof(float));
    cudaMalloc(device_y_ptr, output_length * sizeof(float));
    if(H > 80) {
        cudaMalloc(device_k_ptr, (input_length + kernel_length) * sizeof(half));
        half* device_x_half = reinterpret_cast<half*>(*device_k_ptr);
        dim3 to_fp16_block(1024, 1, 1);
        dim3 to_fp16_grid(ceil(B * H * W * C / NUM_STREAM_1 / 1024.0), 1, 1);
        int xoffset = input_length / NUM_STREAM_1;
        #pragma unroll NUM_STREAM_1
        for(int i = 0; i < NUM_STREAM_1; ++i){
            cudaStreamCreate(&streams_1[i]);
            cudaMemcpyAsync(&((*device_x_ptr)[i*xoffset]), &host_x[i * xoffset], xoffset * sizeof(float), cudaMemcpyHostToDevice, streams_1[i]);
            to_half<<<to_fp16_grid, to_fp16_block, 0, streams_1[i]>>>(*device_x_ptr + i * xoffset,&device_x_half[i*xoffset], xoffset);
        }
        for(int i = 0; i < kernel_length; ++i) kernel_host[i] = __float2half(host_k[i]);
        cudaMemcpyToSymbol(kernel_0, kernel_host, kernel_length * sizeof(half), 0, cudaMemcpyHostToDevice);
        //cudaMemcpy(&device_x_half[input_length], kernel_host, kernel_length * sizeof(half), cudaMemcpyHostToDevice);
    }else {
        cudaMalloc(device_k_ptr, (input_length + kernel_length)* sizeof(half));
        half* device_x_half = reinterpret_cast<half*>(*device_k_ptr);
        dim3 to_fp16_block(1024, 1, 1);
        dim3 to_fp16_grid(ceil(B * H * W * C / NUM_STREAM_2 / 1024.0), 1, 1);
        int xoffset = input_length / NUM_STREAM_2;
        #pragma unroll NUM_STREAM_2
        for(int i = 0; i < NUM_STREAM_2; ++i){
            cudaStreamCreate(&streams_2[i]);
            cudaMemcpyAsync(&((*device_x_ptr)[i*xoffset]), &host_x[i * xoffset], xoffset * sizeof(float), cudaMemcpyHostToDevice, streams_2[i]);
            to_half<<<to_fp16_grid, to_fp16_block, 0, streams_2[i]>>>(*device_x_ptr + i * xoffset,&device_x_half[i*xoffset], xoffset);
        }
        for(int i = 0; i < kernel_length; ++i) kernel_host[i] = __float2half(host_k[i]);
       cudaMemcpyToSymbol(kernel_0, kernel_host, kernel_length * sizeof(half), 0, cudaMemcpyHostToDevice);
       //cudaMemcpy(&device_x_half[input_length], kernel_host, kernel_length * sizeof(half), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

     cudaError_t error = cudaGetLastError();
     if(error != cudaSuccess)
     {
         std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
         exit(-1);
     }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    const half* device_x_half = reinterpret_cast<const half*>(device_k);
    if(H > 80) {
        if(1){
        int W_grid = ceil((float)W_out/TILE_WIDTH);
        int H_grid = ceil((float)H_out/TILE_WIDTH);
        int offset = B / NUM_STREAM_1 * H_out * W_out * M ;
        int xoffset = B / NUM_STREAM_1 * H * W * C;
        //int kernel_offset = B * H * W * C;
        dim3 DimGrid(B / NUM_STREAM_1, M, W_grid * H_grid);
        dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        half *device_x_half = reinterpret_cast<half*>(const_cast<float*>(device_k));
        for(int i = 0; i < NUM_STREAM_1; ++i){
            conv_forward_kernel_large<<<DimGrid, DimBlock, 0, streams_1[i]>>>(device_y + i * offset,
                                                                          device_x + i * xoffset,
                                                                          &device_x_half[i*xoffset],
                                                                          B / NUM_STREAM_1,
                                                                          M, C, H, W, K, TILE_WIDTH);
        }
        }else {
            int xoffset = B * H * W * C / NUM_STREAM_1;
            int offset = B / NUM_STREAM_1 * H_out * W_out * M;
            int a = ceil((float) M / TILE_X_0);
            int b = ceil((float) (W_out * H_out) / TILE_Y_0);
            dim3 DimGrid(B / NUM_STREAM_1, a, b);
            dim3 DimBlock(TILE_X_0,TILE_Y_0, 1);
            int kernel_offset = B * H * W * C;
            for (int i = 0; i < NUM_STREAM_1; ++i) {
                conv_matrix_multi_tiled<<<DimGrid, DimBlock, (TILE_X_0 * C * K * K + TILE_Y_0 * C * K * K) * sizeof(half), streams_1[i]>>>(device_y + i * offset,
                                                                                device_x + i * xoffset,
                                                                                &device_x_half[i * xoffset],
                                                                                B / NUM_STREAM_1,
                                                                                M, C, H, W, K, TILE_WIDTH, kernel_offset - xoffset * i);
            }
        }

    }
    else if(1){
        int W_grid = ceil((float)W_out/TILE_WIDTH_2);
        int H_grid = ceil((float)H_out/TILE_WIDTH_2);
        int tmpa = ceil(W_out * H_out * 1.0 / TILE_WIDTH_2);
        int tmpb = ceil( 1.0 * B / NUM_STREAM_2  / TILE_WIDTH_2 );
        int tmpc = M;
        int xoffset = B * H * W * C / NUM_STREAM_2;
        int offset = B / NUM_STREAM_2 * H_out * W_out * M ;
        dim3 DimGrid(tmpa, tmpb, tmpc);
        dim3 DimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);
        #pragma unroll NUM_STREAM_2
        for(int i = 0; i < NUM_STREAM_2; ++i){

            conv_forward_kernel_0<<<DimGrid, DimBlock, 0, streams_2[i]>>>(device_y + i * offset,
                                                                          device_x + i * xoffset,
                                                                          &device_x_half[i*xoffset],
                                                                          B / NUM_STREAM_2,
                                                                          M, C, H, W, K , TILE_WIDTH_2);
        }
    }else{
        int xoffset = B * H * W * C / NUM_STREAM_2;
        int offset = B / NUM_STREAM_2 * H_out * W_out * M ;
        int kernel_offset = B * H * W * C;
        int a = ceil((float)M / TILE_X);
        int b = ceil((float)(W_out * H_out) / TILE_Y);
        dim3 DimGrid(B / NUM_STREAM_2, a, b);
        dim3 DimBlock(TILE_X, TILE_Y , 1);
        for(int i = 0; i < NUM_STREAM_2; ++i){
            conv_matrix_multi_tiled<<<DimGrid, DimBlock, (TILE_X * C * K * K + TILE_Y * C * K * K) * sizeof(half), streams_2[i]>>>(device_y + i * offset,
                                                                          device_x + i * xoffset,
                                                                          &device_x_half[i*xoffset],
                                                                          B / NUM_STREAM_2,
                                                                          M, C, H, W, K , TILE_WIDTH_2, kernel_offset  - xoffset * i);
        }
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    // cudaMemcpy(host_y, device_y,  B * H_out * W_out * M * sizeof(float), cudaMemcpyDeviceToHost);
    if(H > 80) {
        int offset = B / NUM_STREAM_1 * H_out * W_out * M ;
        #pragma unroll NUM_STREAM_1
        for(int i = 0; i < NUM_STREAM_1; ++i){
            cudaMemcpyAsync(&host_y[i*offset], device_y + i * offset, offset * sizeof(float), cudaMemcpyDeviceToHost, streams_1[i]);
        }
    }else{
        int offset = B / NUM_STREAM_2 * H_out * W_out * M ;
        #pragma unroll NUM_STREAM_2
        for(int i = 0; i < NUM_STREAM_2; ++i){
            cudaMemcpyAsync(&host_y[i*offset], device_y + i * offset, offset * sizeof(float), cudaMemcpyDeviceToHost, streams_2[i]);
        }
    }
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);

    cudaDeviceSynchronize();

    cudaError_t error1 = cudaGetLastError();
    if(error1 != cudaSuccess)
    {
        std::cout<<"final CUDA error: "<<cudaGetErrorString(error1)<<std::endl;
        exit(-1);
    }


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
