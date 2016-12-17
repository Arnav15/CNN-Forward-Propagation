#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <time.h>
#include <valarray>
#include <string>

#include <hdf5.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

#define NUM_STREAMS 32

#define TILEDIM 32
#define POOL_SIZE 2
#define KERNEL_WIDTH 5
#define TILE_WIDTH 12

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

__constant__ float filter1[KERNEL_WIDTH * KERNEL_WIDTH * TILEDIM];

static int loadData(float *x, float *y) {
// Open the data file
    const auto file_id =
    H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

// Open the dataset x and y
    const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
    const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

// Get the dataset x dimensions
    const auto xspace = H5Dget_space(x_id);
    const auto xndims = H5Sget_simple_extent_ndims(xspace);
    assert(xndims == 4);

    hsize_t *input_dims = allocate<hsize_t>(xdims);
    H5Sget_simple_extent_dims(xspace, input_dims, NULL);
    if (input_dims[0] != FLAGS_batch_size) {
        std::cout << "data size does not match batch size specified!\n";
        delete[] input_dims;
return 1; // return error
}
std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
<< " x " << input_dims[2] << " x " << input_dims[3] << "\n";

// Read the dataset x and y
check_success(
    H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
check_success(
    H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

// Close the dataset x and y
check_success(H5Dclose(x_id));
check_success(H5Dclose(y_id));

// Close the file
check_success(H5Fclose(file_id));

// return success
delete[] input_dims;
return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
// Open the model file
    const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

// Open the dataset
    const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
    const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
    const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
    const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

// Read the dataset
    check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
        H5P_DEFAULT, conv1));
    check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
        H5P_DEFAULT, conv2));
    check_success(
        H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
    check_success(
        H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

// Close the dataset x and y
    check_success(H5Dclose(conv1_id));
    check_success(H5Dclose(conv2_id));
    check_success(H5Dclose(fc1_id));
    check_success(H5Dclose(fc2_id));

// Close the file
    check_success(H5Fclose(file_id));
}

/*----------------------------START OF KERNELS---------------------------------*/

/*
* convolution
*   DESCRIPTION: Performs the convolution of the input map with the given filters
*   INPUTS: in_width, out_width, C, out_channel, W_grid, X, W, Y
*   OUTPUTS: none
*   RETURN VALUE: none
*/

__global__ void convolution(int in_width, int out_width, int C, int out_channel, 
    int W_grid, float* X, float* W, float* Y) 
{
    int i, j, n, m, h0, w0, h_base, w_base, h, w;
    int X_tile_width = TILE_WIDTH + KERNEL_WIDTH-1;
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base+ h0;
    w = w_base+ w0;
    float acc = 0.0;

    for (int c = 0; c < C; c++) 
    {
        if (( h0 < KERNEL_WIDTH) && ( w0 < KERNEL_WIDTH))
            W_shared[h0 * KERNEL_WIDTH + w0]= W[ (h0 * KERNEL_WIDTH * C * out_channel) + (w0 * C * out_channel) + (c * out_channel) + m]; 
        __syncthreads();

        for (i = h; i < h_base+ X_tile_width; i += TILE_WIDTH) 
        {
            for (j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
            {
                if(i < in_width && j < in_width)
                    X_shared[(i-h_base) * X_tile_width + (j-w_base)] = X[(n * in_width * in_width * C) + (i * in_width * C) + (j * C) + c]; 
            }
        } 
        __syncthreads();

        for (i = 0; i < KERNEL_WIDTH; i++) 
        {
            for (j = 0; j < KERNEL_WIDTH; j++) 
            { 
                if(h < out_width && w < out_width)
                    acc = acc + X_shared[(h0 + i) * X_tile_width + (w0 + j)] * W_shared[i * KERNEL_WIDTH + j];
            }
        }
        __syncthreads();
    }
    if(h < out_width && w < out_width)
    {
        int Yoffset = ((n * out_width + h) * out_width + w) * out_channel + m;
        Y[Yoffset] = (int)(acc > 0) * acc;
    }
}

/*
* unroll_input
*   DESCRIPTION: The kernel to perform the unrolling of the input map.
*   INPUTS: H_out, W_out, C, H, M, unrolled_height, unrolled_width, in, unrolled
*   OUTPUTS: none
*   RETURN VALUE: none
*/

__global__ void unroll_input(int H_out, int W_out, int C, int H, int W, int unrolled_height, int unrolled_width, const float * in, float * unrolled) {

    int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int W_unroll = H_out * W_out; 

    if (t < C * W_unroll) {
        c = t / W_unroll;
        s = t % W_unroll;

        h_out = s / W_out;
        w_out = s % W_out;

        h_unroll = h_out * W_out + w_out;
        w_base = c * KERNEL_WIDTH * KERNEL_WIDTH;

        for(p = 0; p < KERNEL_WIDTH; p++) 
        {
            for(q = 0; q < KERNEL_WIDTH; q++) 
            {
                w_unroll = w_base + p * KERNEL_WIDTH + q;
                unrolled[w_unroll * W_unroll + h_unroll] = in[(h_out + p) * W * C + (w_out + q) * C + c];
            }
        }
    }
}

/*
* reroll
*   DESCRIPTION: The kernel to perform the rerolling of the output map.
*   INPUTS: y_unroll, y_reroll, H_out, W_out, M
*   OUTPUTS: none
*   RETURN VALUE: none
*/

__global__ void reroll(float* y_unroll, float* y_reroll, int H_out, int W_out, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = idx / (H_out*W_out);
    int position = idx % (H_out*W_out);
    int row = position / W_out;
    int col = position % W_out;

    if(idx < H_out * W_out * M)
    {
        y_reroll[row * W_out * M + col * M + m] = y_unroll[m * H_out * W_out + row * W_out + col];
    }
}

/*
* pooling
*   DESCRIPTION: The kernel to perform the average pooling required for the forward step
*   INPUTS: in_width, out_width, in_channel, out_channel, X, Y 
*   OUTPUTS: none
*   RETURN VALUE: none
*/

__global__ void pooling(int in_width, int out_width, int in_channel, int out_channel, float* X, float* Y) 
{
    int p, q, bx, by, tx, ty;
    bx = blockIdx.x; 
    by = blockIdx.y;
    tx = threadIdx.x;
    ty = threadIdx.y;
    float acc = 0.0;
    int Yoffset = (bx * out_width * out_width * out_channel) + (ty * out_width * out_channel) + (tx * out_channel) + by;
    int pool_s = 2;

    for (p = 0; p < POOL_SIZE; p++) 
    {
        for (q = 0; q < POOL_SIZE; q++) 
            acc += X[(bx * in_width * in_width * in_channel) + (((POOL_SIZE * ty) + p) * in_width * in_channel)
                + ((POOL_SIZE * tx + q) * in_channel) + by]/(1.0f * pool_s * pool_s);

    }
    Y[Yoffset] = acc;
}

/*
* matrixMultiplyShared
*   DESCRIPTION: The kernel to perform matrix multiplication using shared memory
*   INPUTS: A, B, C, numARows, numAColumns, numCRows, numBColumns 
*   OUTPUTS: none
*   RETURN VALUE: none
*/

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns) 
{

    float CValue = 0;

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float subTileM[TILEDIM][TILEDIM];
    __shared__ float subTileN[TILEDIM][TILEDIM];

    for (int i = 0; i < (ceil((float)numBRows/TILEDIM)); i++) 
    {
        if (i*TILEDIM + threadIdx.x < numAColumns && Row < numARows)   
            subTileM[threadIdx.y][threadIdx.x] = A[Row * numAColumns + i * TILEDIM + threadIdx.x];
        else                                                   
            subTileM[threadIdx.y][threadIdx.x] = 0.0;

        if (i*TILEDIM + threadIdx.y < numBRows && Col < numBColumns)   
            subTileN[threadIdx.y][threadIdx.x] = B[(i * TILEDIM + threadIdx.y) * numBColumns + Col];
        else                                                   
            subTileN[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int j = 0; j < TILEDIM; j++) 
            CValue += subTileM[threadIdx.y][j] * subTileN[j][threadIdx.x];

        __syncthreads();
    }

    if (Row < numARows && Col < numBColumns) 
        C[((blockIdx.y * blockDim.y + threadIdx.y) * numBColumns)+(blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
}

/*
* matrixMultiplyShared1
*   DESCRIPTION: The kernel to perform matrix multiplication using shared memory and global memory
*   INPUTS: B, C, numARows, numAColumns, numCRows, numBColumns
*   OUTPUTS: none
*   RETURN VALUE: none
*/

__global__ void matrixMultiplyShared1(float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns) 
{

    float CValue = 0;

    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ float subTileM[TILEDIM][TILEDIM];
    __shared__ float subTileN[TILEDIM][TILEDIM];

    for (int i = 0; i < (ceil((float)numBRows/TILEDIM)); i++) 
    {
        if (i*TILEDIM + threadIdx.x < numAColumns && Row < numARows)   
            subTileM[threadIdx.y][threadIdx.x] = filter1[Row * numAColumns + i * TILEDIM + threadIdx.x];
        else                                                   
            subTileM[threadIdx.y][threadIdx.x] = 0.0;

        if (i*TILEDIM + threadIdx.y < numBRows && Col < numBColumns)   
            subTileN[threadIdx.y][threadIdx.x] = B[(i * TILEDIM + threadIdx.y) * numBColumns + Col];
        else                                                   
            subTileN[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int j = 0; j < TILEDIM; j++) 
            CValue += subTileM[threadIdx.y][j] * subTileN[j][threadIdx.x];


        __syncthreads();
    }

    if (Row < numARows && Col < numBColumns) 
        C[((blockIdx.y * blockDim.y + threadIdx.y) * numBColumns) + (blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
}

/*-----------------------------END OF KERNELS---------------------------------*/

/*----------------------START OF SEQUENTIAL FUNCTIONS-------------------------*/
void unroll_filter(const float * W, float * w, int M, int C) {

    for(int row = 0; row < KERNEL_WIDTH; row++) 
    {
        for(int col = 0; col< KERNEL_WIDTH; col++) 
        {
            for(int i = 0; i < C; i++)
            {
                for(int j = 0; j < M; j++) 
                    w[j * C * KERNEL_WIDTH * KERNEL_WIDTH + i * KERNEL_WIDTH * KERNEL_WIDTH + row * KERNEL_WIDTH + col] = W[row * KERNEL_WIDTH * C * M + col * C * M + i * M + j];
            }
        }
    }
}

// Recified linear unit 4d
static void relu4(float *X, const int xdims[4]) 
{
    for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3]))
        X[i] = (X[i] < 0) ? 0 : X[i];
}

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) 
{
    for (const auto i : range(0, xdims[0] * xdims[1]))
        X[i] = (X[i] < 0) ? 0 : X[i];
}

// Choose the guess with largest score
void argmax(const float *X, const int xdims[2], int *Y) 
{
    for (const auto i : range(0, xdims[0])) {
        auto max_idx = 0;
        auto max     = X[i * xdims[1]];
        for (const auto j : range(0, xdims[1])) {
            const auto elem = X[(i * xdims[1]) + j];
            if (elem > max) {
                max_idx = j;
                max     = elem;
            }
        }
        Y[i] = max_idx;
    }
}
/*------------------------END OF SEQUENTIAL FUNCTIONS-------------------------*/

/*------------------------- START OF KERNEL CALLS ----------------------------*/

/*
* conv_forward_valid
*   DESCRIPTION: Calls the kernel for unrolling the input map, matrix multiplication 
*                and rerolling of the output map to perform the convolution.
*                Only used if the input map is above a given threshold.
*   INPUTS: X, xdims, W, wdims, Y, ydims 
*   OUTPUTS: none
*   RETURN VALUE: none
*/

static void conv_forward_valid(const float *X, const int xdims[4],
    const float *W, const int wdims[4], float *Y,
    const int ydims[4]) 
{
    int i, j;

    int C = wdims[2]; 
    int M = wdims[3]; 
    int H = xdims[1];  
    int w = xdims[2];
    int H_out = ydims[1];
    int W_out = ydims[2];

    int x_batch_size = xdims[1] * xdims[2] * xdims[3];
    int y_batch_size = ydims[1] * ydims[2] * ydims[3];
    int x_unrolled_height = C * KERNEL_WIDTH * KERNEL_WIDTH;
    int x_unrolled_width = H_out * W_out;
    int filter_height = M;
    int filter_width = C * KERNEL_WIDTH * KERNEL_WIDTH;

    float * device_input[NUM_STREAMS];
    float * device_output;
    float * device_X[NUM_STREAMS];
    float * device_Y[NUM_STREAMS];
    float * device_roll_Y[NUM_STREAMS];
    float * filter;

    cudaStream_t streams[NUM_STREAMS];

    for(i=0; i < NUM_STREAMS; i++) 
    {
        cudaStreamCreate(&streams[i]);

        cudaMalloc((void **) &device_X[i], x_unrolled_height * x_unrolled_width * sizeof(float));
        cudaMalloc((void **) &device_Y[i], y_batch_size * sizeof(float));
        cudaMalloc((void **) &device_roll_Y[i], y_batch_size * sizeof(float));
        cudaMalloc((void **) &device_input[i], x_batch_size * sizeof(float));
    }
    
    cudaMalloc((void**) &device_output, ydims[0] * y_batch_size * sizeof(float));
    cudaMalloc((void **) &filter, filter_height * filter_width * sizeof(float));

    float * w_unrolled = (float *) malloc(filter_height * filter_width * sizeof(float));

    unroll_filter(W, w_unrolled, M, C);

    if(wdims[2] == 1)
        cudaMemcpyToSymbol(filter1, w_unrolled, filter_height * filter_width * sizeof(float));
    else
        cudaMemcpy(filter, w_unrolled, filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);

    int unroll_block = 1024;
    int unroll_grid = ceil( (float) (C * x_unrolled_width) / 1024);

    dim3 mult_block(TILEDIM, TILEDIM, 1);
    dim3 mult_grid(ceil( (float) (H_out * W_out) / TILEDIM), ceil((float) M / TILEDIM), 1);  

    int reroll_block = 1024;
    int reroll_grid = ceil( (float) y_batch_size / 1024);

    for (i = 0; i < ydims[0]; i += NUM_STREAMS) 
    {  
        for(j = 0; (i + j < ydims[0]) && (j < NUM_STREAMS); j++)
        {
            int xoffset = (i + j) * x_batch_size;

            cudaMemcpyAsync(device_input[j], &X[xoffset], x_batch_size * sizeof(float), cudaMemcpyHostToDevice, streams[j]);

            unroll_input<<< unroll_grid, unroll_block, 0, streams[j] >>> (H_out, W_out, C, H, w, 
                x_unrolled_height, x_unrolled_width, 
                device_input[j], device_X[j]); 
        }
        
        for(j=0; (i + j < ydims[0]) && (j < NUM_STREAMS); j++) 
        {
            if(wdims[2] == 1)
                matrixMultiplyShared1<<< mult_grid, mult_block, 0, streams[j] >>>(device_X[j], device_Y[j], 
                    filter_height, filter_width, 
                    x_unrolled_height, x_unrolled_width);
            else
                matrixMultiplyShared<<< mult_grid, mult_block, 0, streams[j] >>>(filter, device_X[j], device_Y[j], 
                    filter_height, filter_width, 
                    x_unrolled_height, x_unrolled_width);
        }
        
        for(j=0; (i + j < ydims[0]) && (j < NUM_STREAMS); j++) 
        {
            device_roll_Y[j] = device_output + (i + j) * y_batch_size;

            reroll<<< reroll_grid, reroll_block, 0, streams[j] >>> (device_Y[j], device_roll_Y[j], 
                ydims[1], ydims[2], ydims[3]);
        }
    }  

    cudaMemcpy(Y, device_output, ydims[0] * y_batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    free(w_unrolled);

    for(i = 0; i < NUM_STREAMS; i++) 
    {
        cudaFree(device_input[i]);
        cudaFree(device_X[i]);
        cudaFree(device_Y[i]);
        cudaFree(device_roll_Y[i]);
    }
    cudaFree(filter);
}

/*
* conv_forward_valid2
*   DESCRIPTION: Calls the kernel for basic convolution for the forward step.
*                Only used if the input map is below a given threshold.
*   INPUTS: X, xdims, W, wdims, Y, ydims 
*   OUTPUTS: none
*   RETURN VALUE: none
*/

void conv_forward_valid2(const float *X, const int xdims[4],
    const float *W, const int wdims[4], float *Y,
    const int ydims[4]) 
{
    float* device_input;
    float* device_output;
    float* filter_conv;

    cudaMalloc((void **) &filter_conv, sizeof(float) * conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3]);
    cudaMalloc((void **) &device_input, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3]);
    cudaMalloc((void **) &device_output, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3]);

    cudaMemcpy(device_input, X, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3], cudaMemcpyHostToDevice);

    dim3 dimGrid(xdims[0], ydims[3], ceil(ydims[1]/(float)TILE_WIDTH) * ceil(ydims[1]/(float)TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    size_t shmem_size = sizeof(float) * ( (TILE_WIDTH + KERNEL_WIDTH-1)*(TILE_WIDTH + KERNEL_WIDTH-1) + KERNEL_WIDTH*KERNEL_WIDTH );

    cudaMemcpy(filter_conv, W, sizeof(float) * conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3], cudaMemcpyHostToDevice);

    convolution<<<dimGrid, dimBlock, shmem_size>>>(xdims[1], ydims[1], xdims[3], ydims[3],
        2, device_input, filter_conv, device_output);

    cudaMemcpy(Y, device_output, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3], cudaMemcpyDeviceToHost);

    cudaFree(filter_conv);
    cudaFree(device_output);
    cudaFree(device_input);
    return;
}

/*
* average_pool
*   DESCRIPTION: Calls the kernel for pooling
*   INPUTS: X, xdims, Y, ydims 
*   OUTPUTS: none
*   RETURN VALUE: none
*/

void average_pool(const float *X, const int xdims[4],
    float *Y, const int ydims[4]) 
{
    float* device_input;
    float* device_output;
    cudaMalloc((void **) &device_input, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3]);
    cudaMalloc((void **) &device_output, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3]);
    cudaMemcpy(device_input, X, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3], cudaMemcpyHostToDevice);

    dim3 dimGrid(xdims[0], ydims[3], 1);
    dim3 dimBlock(ydims[1], ydims[1]);
    pooling<<<dimGrid, dimBlock>>>(xdims[1], ydims[1], xdims[3], ydims[3], device_input, device_output);

    cudaMemcpy(Y, device_output, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3], cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    cudaFree(device_input);
    return;

}

/*
* fully_forward
*   DESCRIPTION: Calls the kernel for matrix multiplication
*   INPUTS: X, xdims, W, wdims, Y, ydims 
*   OUTPUTS: none
*   RETURN VALUE: none
*/

void fully_forward(const float *X, const int xdims[2], float *W,
    const int wdims[2], float *Y, const int ydims[2]) 
{
    float* device_input;
    float* device_output;
    float* device_w;

    cudaMalloc((void **) &device_input, sizeof(float) * xdims[0] * xdims[1]);
    cudaMalloc((void **) &device_output, sizeof(float) * ydims[0] * ydims[1]);
    cudaMalloc((void **) &device_w, sizeof(float)*wdims[0] * wdims[1]);

    cudaMemcpy(device_input, X, sizeof(float) * xdims[0] * xdims[1], cudaMemcpyHostToDevice);
    cudaMemcpy(device_w, W, sizeof(float) * wdims[0]*wdims[1], cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(ydims[1]/(float)TILEDIM), ceil(ydims[0]/(float)TILEDIM));
    dim3 dimBlock(TILEDIM, TILEDIM);

    matrixMultiplyShared<<<dimGrid,dimBlock>>>(device_input, device_w, device_output,
        xdims[0], xdims[1],
        wdims[0], wdims[1]);

    cudaMemcpy(Y, device_output, sizeof(float) * ydims[0] * ydims[1], cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_w);
}
/*------------------------- END OF KERNEL CALLS ----------------------------*/

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1, float *fc2, int *out) 
{

// conv layer
    const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
    auto a = zeros<float>(adims);

    if(xdims[0] >= 100)
        conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);
    else
        conv_forward_valid2(x, xdims, conv1, conv1dims, a, adims);

// relu layer
    relu4(a, adims);

// average pooling
    const int pool_size = 2;
    const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};
    auto b = zeros<float>(bdims);

    average_pool(a, adims, b, bdims);

// conv layer
    const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
    auto c = zeros<float>(cdims);
    if(bdims[0] >= 100)
        conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);
    else 
        conv_forward_valid2(b, bdims, conv2, conv2dims, c, cdims);

// relu
    relu4(c, cdims);

// average pooling
    const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3]};
    auto d = zeros<float>(ddims);

    average_pool(c, cdims, d, ddims);

// reshape
    const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

// matrix multiplication
    const int edims[] = {ddims[0], fc1dims[1]};
    auto e            = zeros<float>(edims);

    fully_forward(d, ddims2, fc1, fc1dims, e, edims);

// relu
    relu2(e, edims);

// matrix multiplication
    const int fdims[] = {edims[0], fc2dims[1]};
    auto f            = zeros<float>(fdims);

    fully_forward(e, edims, fc2, fc2dims, f, fdims);

    argmax(f, fdims, out);

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    delete[] e;
    delete[] f;
}

int main(int argc, char **argv) 
{

    if (argc != 3 && argc != 4) {
        std::cerr << "\n"
        << "This program performs the forward opertion step for "
        "Convolutional Neural Network(CNN).  "
        "Sample usage: \n"
        << argv[0]
        << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
        return -1;
    }
    FLAGS_testdata = std::string(argv[1]);
    FLAGS_model    = std::string(argv[2]);
    if (argc == 3) {
        const std::map<std::string, int> default_batch_sizes{
            {"../data/test2.hdf5", 2},
            {"../data/test10.hdf5", 10},
            {"../data/test100.hdf5", 100},
            {"../data/testfull.hdf5", 10000}};
            const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
            if (batch_size_in_map == default_batch_sizes.end()) {
                std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
                return -1;
            }
            FLAGS_batch_size = batch_size_in_map->second;
        } else if (argc == 4) {
            FLAGS_batch_size = atoi(argv[3]);
    }
    xdims[0] = FLAGS_batch_size;
    rdims[0] = FLAGS_batch_size;

    // Load data into x and y
    float *x = allocate<float>(xdims);
    float *y = allocate<float>(rdims);
    loadData(x, y);

    // Load model
    float *conv1 = allocate<float>(conv1dims);
    float *conv2 = allocate<float>(conv2dims);
    float *fc1   = allocate<float>(fc1dims);
    float *fc2   = allocate<float>(fc2dims);
    loadModel(conv1, conv2, fc1, fc2);

    // Perform foward opertion
    int *out = zeros<int>(FLAGS_batch_size);

    // get start time
    const auto start = now();

    forward_operation(x, conv1, conv2, fc1, fc2, out);

    // get end time
    const auto end = now();

    // get elapsed time in milliseconds
    const auto elapsed =
    std::chrono::duration<double, std::milli>(end - start).count();

    // Get reference
    int *ref = zeros<int>(FLAGS_batch_size);
    argmax(y, rdims, ref);

    // Calculate correctness
    int num_correct = 0;
    for (const auto i : range(0, FLAGS_batch_size)) {
        if (out[i] == ref[i]) {
            num_correct++;
        }
    }
    std::cout << "Done with " << FLAGS_batch_size << " queries in "
    << "elapsed = " << elapsed << " milliseconds. Correctness: "
    << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

    delete[] x;
    delete[] y;
    delete[] conv1;
    delete[] conv2;
    delete[] fc1;
    delete[] fc2;
    delete[] out;
    delete[] ref;

    return 0;
}