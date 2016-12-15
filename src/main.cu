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

#define NUM_STREAMS 16

#define TILEDIM 16
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
__global__ void convolution(int in_width, int out_width, int in_channel, int out_channel, 
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

    for (int c = 0; c < in_channel; c++) 
    {
        if (( h0 < KERNEL_WIDTH) && ( w0 < KERNEL_WIDTH))
             W_shared[h0*KERNEL_WIDTH+ w0]= W[ (h0 * KERNEL_WIDTH * in_channel* out_channel) + (w0* in_channel*out_channel) +(c*out_channel)+m]; 
         __syncthreads();

        for (i = h; i < h_base+ X_tile_width; i += TILE_WIDTH) 
        {
            for (j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
            {
                if(i < in_width && j < in_width)
                  X_shared[(i-h_base)*X_tile_width+ (j-w_base)] = X[(n*in_width*in_width*in_channel)+(i*in_width*in_channel)+(j*in_channel)+c]; 
            }
        } 
        __syncthreads();

        for (i = 0; i < KERNEL_WIDTH; i++) 
        {
            for (j = 0; j < KERNEL_WIDTH; j++) 
            { 
                if(h < out_width && w < out_width)
                    acc = acc + X_shared[(h0 + i)*X_tile_width + (w0+j)] * W_shared[i *KERNEL_WIDTH + j];
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

void unroll_weights(const float * W, float * w_unrolled, int M, int C, int filter_height, int filter_width) {
  int ip_map, op_map, row, col;
  for(row = 0; row <filter_height; row++) {
    for(col = 0; col< filter_width; col++) {
      for(ip_map = 0; ip_map < C; ip_map++){
        for(op_map = 0; op_map < M; op_map++) {
          w_unrolled[op_map * C * filter_width * filter_height + ip_map * filter_height * filter_width + row * filter_width + col] = W[row * filter_width * C * M + col * C * M + ip_map * M + op_map];
        }
      }
    }
  }
}

__global__ void unroll_Kernel(int H_out, int W_out, int C, int H, int W, int filter_height, int filter_width, int unrolled_height, int unrolled_width, const float * in, float * unrolled) {
  
  int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int W_unroll = H_out * W_out; //width of unrolled matrix

  if (t < C * W_unroll) {
    c = t / W_unroll; //current input map we're looking at in expanded matrix
    s = t % W_unroll; //current element of input map we're looking at

    h_out = s / W_out; //rows iterating through num Output rows
    w_out = s % W_out;  //column iterating through num Output cols
    
    h_unroll = h_out * W_out + w_out; //row of unrolled matrix
    w_base = c * filter_height * filter_width;

    for(p = 0; p < filter_height; p++) {
      for(q = 0; q < filter_width; q++) {
        w_unroll = w_base + p * filter_width + q; //column of unrolled matrix
        unrolled[w_unroll * W_unroll + h_unroll] = in[(h_out + p) * W * C + (w_out + q) * C + c];
      }
    }
  }
}

__global__ void reroll_Y_kernel2(float* y_unroll, float* y_reroll, int H_out, int W_out, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = idx / (H_out*W_out);
    int position = idx % (H_out*W_out);
    int row = position / W_out;
    int col = position % W_out;

    if(idx < H_out * W_out * M)
    {
        y_reroll[row * W_out * M + col * M + m] = y_unroll[m*H_out*W_out + row*W_out + col];
    }
}

__global__ void multiplication_kernel_conv2(const float * unrolled, const float * weights, int unrolled_height, int unrolled_width, int weight_height, int weight_width, float * out) {
  __shared__ float tileW [TILEDIM][TILEDIM];
  __shared__ float tileX [TILEDIM][TILEDIM];

  int out_height = weight_height;
  int out_width = unrolled_width;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y*blockDim.y + ty;
  int col = blockIdx.x*blockDim.x + tx;

  float res = 0;

  for(int tilenum = 0; tilenum < (ceil((float)unrolled_height/TILEDIM)); tilenum++){
    if(row < weight_height && (tilenum*TILEDIM)+tx < weight_width){
        tileW[ty][tx] = weights[row*weight_width+tilenum*TILEDIM+tx];
      }
    else {
      tileW[ty][tx] = 0;
    }


    if((ty+tilenum*TILEDIM)<unrolled_height && col < unrolled_width){
      tileX[ty][tx] = unrolled[col+unrolled_width*(ty+tilenum*TILEDIM)];
    }
    else {
      tileX[ty][tx] = 0;
    }
    __syncthreads();
    for(int i = 0; i < TILEDIM; i++) {
      res += tileW[ty][i] * tileX[i][tx];
    }
    __syncthreads();
  }

  if(row < out_height && col < out_width){
    out[row*out_width+col] = (res<0) ? 0 : res;
  }
}

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  
    float CValue = 0;

    int Row = blockIdx.y*TILEDIM + threadIdx.y;
    int Col = blockIdx.x*TILEDIM + threadIdx.x;

    __shared__ float subTileM[TILEDIM][TILEDIM];
    __shared__ float subTileN[TILEDIM][TILEDIM];

    for (int i = 0; i < (TILEDIM + numAColumns - 1)/TILEDIM; i++) {

         if (i*TILEDIM + threadIdx.x < numAColumns && Row < numARows)   
            subTileM[threadIdx.y][threadIdx.x] = A[Row*numAColumns + i*TILEDIM + threadIdx.x];
         else                                                   
            subTileM[threadIdx.y][threadIdx.x] = 0.0;

         if (i*TILEDIM + threadIdx.y < numBRows && Col < numBColumns)   
            subTileN[threadIdx.y][threadIdx.x] = B[(i*TILEDIM + threadIdx.y)*numBColumns + Col];
         else                                                   
            subTileN[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         if(Row<numARows && Col < numBColumns)
         {
            for (int j = 0; j < TILEDIM; j++) 
                CValue += subTileM[threadIdx.y][j] * subTileN[j][threadIdx.x];
          }

         __syncthreads();
    }

    if (Row < numCRows && Col < numCColumns) 
        C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}

static void conv_forward_valid(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  int filter_height = wdims[0];
  int filter_width = wdims[1];
  int C = wdims[2]; //number input feature maps
  int M = wdims[3]; //number output feature maps
  int H = xdims[1]; //height of input map 
  int w = xdims[2]; //width of input map
  int H_out = ydims[1];  //height of output map
  int W_out = ydims[2];   //width of output map
  
  int x_batch_size = xdims[1] * xdims[2] * xdims[3];
  int y_batch_size = ydims[1] * ydims[2] * ydims[3];
  int x_unrolled_height = C * filter_height * filter_width;
  int x_unrolled_width = H_out * W_out;
  int weight_matrix_height = M;
  int weight_matrix_width = C * filter_height * filter_width;
 
  //dimensions for unroll kernel
  int unroll_num_threads = C * H_out * W_out;
  dim3 unroll_block_dim(512, 1, 1);
  dim3 unroll_grid_dim(ceil( (float) (unroll_num_threads) / unroll_block_dim.x), 1, 1);

  //dimensions for the multiply kernel
  dim3 multiply_block_dim(TILEDIM, TILEDIM, 1);
  dim3 multiply_grid_dim(ceil( (float) (H_out * W_out) / multiply_block_dim.x), ceil((float) M / multiply_block_dim.y), 1);  
  //dim3 multiply_grid_dim(ceil((M * H_out * W_out) / multiply_block_dim.x), 1, 1);

  //Dimensions for reroll kernel
  dim3 reroll_block_dim(1024, 1, 1);
  dim3 reroll_grid_dim( ceil( (float) y_batch_size / reroll_block_dim.x), 1, 1);

  //Initialize our streams
  cudaStream_t streams[NUM_STREAMS];

  //Create device memory for streams
  float * d_xin[NUM_STREAMS];
  float * d_input;
  float * d_output;
  float * d_xunrolled[NUM_STREAMS];
  float * d_yout[NUM_STREAMS];
  float * d_yrolled[NUM_STREAMS];
  float * d_weights;

  int i, j;
  for(i=0; i<NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc((void **) &d_xunrolled[i], x_unrolled_height * x_unrolled_width * sizeof(float));
    cudaMalloc((void **) &d_yout[i], y_batch_size * sizeof(float));
    cudaMalloc((void **) &d_yrolled[i], y_batch_size * sizeof(float));
  }
  cudaMalloc((void**) &d_input, xdims[0] * x_batch_size * sizeof(float));
  cudaMalloc((void**) &d_output, ydims[0] * y_batch_size * sizeof(float));
 
  //Unroll the weights
  cudaMalloc((void **) &d_weights, weight_matrix_height * weight_matrix_width * sizeof(float));
  
  float * w_unrolled = (float *) malloc(weight_matrix_height * weight_matrix_width * sizeof(float));
  
  unroll_weights(W, w_unrolled, M, C, filter_height, filter_width);

  cudaMemcpyAsync(d_weights, w_unrolled, weight_matrix_height * weight_matrix_width * sizeof(float), cudaMemcpyHostToDevice, streams[0]);

  cudaMemcpyAsync(d_input, X, xdims[0] * x_batch_size * sizeof(float), cudaMemcpyHostToDevice, streams[1]);

  cudaDeviceSynchronize();
  
  for (i=0; i<ydims[0]; i+=NUM_STREAMS) {

    for(j=0; (i + j < ydims[0]) && (j < NUM_STREAMS); j++) {
      d_xin[j] = d_input + (i + j) * x_batch_size;
      d_yrolled[j] = d_output + (i+ j) * y_batch_size;
    }
    
    for(j=0; (i + j < ydims[0]) && (j < NUM_STREAMS); j++) {
      unroll_Kernel<<< unroll_grid_dim, unroll_block_dim, 0, streams[j] >>> (H_out, W_out, C, H, w, filter_height, filter_width, x_unrolled_height, x_unrolled_width, d_xin[j], d_xunrolled[j]); 
    }
    for(j=0; (i + j < ydims[0]) && (j < NUM_STREAMS); j++) {
      multiplication_kernel_conv2<<< multiply_grid_dim, multiply_block_dim, 0, streams[j] >>> (d_xunrolled[j], d_weights, x_unrolled_height, x_unrolled_width, weight_matrix_height, weight_matrix_width, d_yout[j]);
    }

    for(j=0; (i + j < ydims[0]) && (j < NUM_STREAMS); j++) {
      reroll_Y_kernel2<<< reroll_grid_dim, reroll_block_dim, 0, streams[j] >>> (d_yout[j], d_yrolled[j], ydims[1], ydims[2], ydims[3]);
    }
  }  
  cudaDeviceSynchronize();
  cudaMemcpy(Y, d_output, ydims[0] * y_batch_size * sizeof(float), cudaMemcpyDeviceToHost);

  //Free the memory that we used
  free(w_unrolled);
  for(i=0; i<3; i++) {
    cudaFree(d_xunrolled[i]);
    cudaFree(d_yout[i]);
    cudaFree(d_yrolled[i]);
  }
  cudaFree(d_weights);
}

// From book chapter Figure 16.4
void conv_forward_valid2(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  // const auto filter_h   = wdims[0];
  // const auto filter_w   = wdims[1];
  // const auto in_channel = wdims[2];

  // for (const auto i : range(0, ydims[0])) {
  //   for (const auto m : range(0, ydims[3])) {
  //     for (const auto w : range(0, ydims[2])) {
  //       for (const auto h : range(0, ydims[1])) {
  //         for (const auto p : range(0, filter_h)) {
  //           for (const auto q : range(0, filter_w)) {
  //             for (const auto c : range(0, in_channel)) {
  //               const auto yoffset =
  //                   ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
  //               const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
  //                                    (h + p) * xdims[2] * xdims[3] +
  //                                    (w + q) * xdims[3] + c;
  //               const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
  //                                    q * wdims[2] * wdims[3] + c * wdims[3] + m;
  //               Y[yoffset] += X[xoffset] * W[woffset];
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

    float* d_input_img;
    float* convout;
    cudaMalloc((void **) &d_input_img, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3]);
    cudaMalloc((void **) &convout, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3]);
    cudaMemcpy(d_input_img, X, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3], cudaMemcpyHostToDevice);

    int W_grid = ceil(ydims[1]/(float)TILE_WIDTH);
    int H_grid = ceil(ydims[1]/(float)TILE_WIDTH);
    int Z = H_grid * W_grid;

    dim3 dimGrid(xdims[0],ydims[3],Z);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

    size_t shmem_size = sizeof(float) * ( (TILE_WIDTH + KERNEL_WIDTH-1)*(TILE_WIDTH + KERNEL_WIDTH-1) + KERNEL_WIDTH*KERNEL_WIDTH );

    float* filter_conv;
    cudaMalloc((void **) &filter_conv, sizeof(float) * conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3]);
    cudaMemcpy(filter_conv, W, sizeof(float) * conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3], cudaMemcpyHostToDevice);
    convolution<<<dimGrid, dimBlock, shmem_size>>>(xdims[1], ydims[1], xdims[3], ydims[3],
                                                   2, d_input_img, filter_conv, convout);
    cudaFree(filter_conv);

    cudaMemcpy(Y, convout, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3], cudaMemcpyDeviceToHost);
    cudaFree(convout);
    cudaFree(d_input_img);
    return;
}

// Recified linear unit 4d
static void relu4(float *X, const int xdims[4]) {
  for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

void average_pool(const float *X, const int xdims[4],
                         float *Y, const int ydims[4]) {
  // for (const auto i : range(0, ydims[0])) {
  //   for (const auto m : range(0, ydims[3])) {
  //     for (const auto w : range(0, ydims[2])) {
  //       for (const auto h : range(0, ydims[1])) {
  //         for (const auto p : range(0, pool_size)) {
  //           for (const auto q : range(0, pool_size)) {
  //             const auto yoffset =
  //                 ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
  //             const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
  //                                  (pool_size * h + p) * xdims[2] * xdims[3] +
  //                                  (pool_size * w + q) * xdims[3] + m;
  //             Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  float* d_input_img;
  float* poolout;
  cudaMalloc((void **) &d_input_img, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3]);
  cudaMalloc((void **) &poolout, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3]);
  cudaMemcpy(d_input_img, X, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3], cudaMemcpyHostToDevice);

  dim3 dimGrid(xdims[0], ydims[3], 1);
  dim3 dimBlock(ydims[1], ydims[1]);
  pooling<<<dimGrid, dimBlock>>>(xdims[1], ydims[1], xdims[3], ydims[3], d_input_img, poolout);

  cudaMemcpy(Y, poolout, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3], cudaMemcpyDeviceToHost);
  cudaFree(poolout);
  cudaFree(d_input_img);
  return;

}

void fully_forward(const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2]) {
  // for (const auto i : range(0, xdims[0])) {
  //   for (const auto j : range(0, wdims[1])) {
  //     float sum = 0;
  //     for (const auto k : range(0, xdims[1])) {
  //       sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
  //     }
  //     Y[i * wdims[1] + j] = sum;
  //   }
  // }

  float* d_input_img;
  float* fc1out;
  float* w_matrix;
  cudaMalloc((void **) &d_input_img, sizeof(float) * xdims[0] * xdims[1]);
  cudaMalloc((void **) &fc1out, sizeof(float) * ydims[0] * ydims[1]);
  cudaMalloc((void **) &w_matrix, sizeof(float)*wdims[0] * wdims[1]);

  cudaMemcpy(d_input_img, X, sizeof(float) * xdims[0] * xdims[1], cudaMemcpyHostToDevice);
  cudaMemcpy(w_matrix, W, sizeof(float) * wdims[0]*wdims[1], cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(ydims[1]/(float)TILEDIM), ceil(ydims[0]/(float)TILEDIM));
  dim3 dimBlock(TILEDIM, TILEDIM);

  matrixMultiplyShared<<<dimGrid,dimBlock>>>(d_input_img, w_matrix, fc1out,
                                            xdims[0], xdims[1],
                                            wdims[0], wdims[1],
                                            ydims[0], ydims[1]);
  cudaMemcpy(Y, fc1out, sizeof(float) * ydims[0] * ydims[1], cudaMemcpyDeviceToHost);

  cudaFree(d_input_img);
  cudaFree(fc1out);
  cudaFree(w_matrix);

}

// Choose the guess with largest score
void argmax(const float *X, const int xdims[2], int *Y) {
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

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1, float *fc2, int *out) {

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

int main(int argc, char **argv) {

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